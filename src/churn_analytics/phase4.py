"""Train-only Logistic Regression development; the final test remains reserved."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

from .cleaning import DEFAULT_CLEAN_PATH, load_cleaned_dataset
from .data import PROJECT_ROOT
from .model_splits import create_split_manifest, development_split, validate_split_manifest
from .modeling import (CATEGORICAL_FEATURES, EXCLUDED_FEATURES, FEATURES, MODEL_NAMES, NUMERIC_FEATURES,
                       THRESHOLD, build_pipeline, coefficient_report, evaluate_pipeline, feature_policy, fit_pipeline)
from .model_reporting import logistic_report, metric_table
from .phase2 import _write_json


def _ids_hash(ids) -> str:
    """Record model-operation customer provenance without duplicating ID lists."""
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def run_phase4(
    processed_path: str | Path = DEFAULT_CLEAN_PATH,
    reports_dir: str | Path = PROJECT_ROOT / "reports",
    models_dir: str | Path = PROJECT_ROOT / "models",
) -> dict:
    """Fit a dummy plus two fixed Logistic Regression candidates on train only.

    Persist the selection rule before fitting. Evaluate train/validation only and
    select by validation AP (ties: standard). Save that already-fitted Pipeline,
    without train+validation/full-data refitting. Existing split identities cannot
    silently change on reruns. There is intentionally no test-evaluation switch.
    """
    processed_path, reports_dir, models_dir = map(Path, (processed_path, reports_dir, models_dir))
    if not processed_path.is_file():
        raise FileNotFoundError(f"Phase 2 processed dataset missing: {processed_path}. Run Phase 2 first; Phase 4 never re-cleans raw data.")
    for output in (reports_dir, models_dir):
        if any(output.resolve().is_relative_to(PROJECT_ROOT / "data" / name) for name in ["raw", "processed", "database"]):
            raise ValueError("Model/report outputs must not overwrite prior data directories")
    source_fingerprint = hashlib.sha256(processed_path.read_bytes()).hexdigest()
    frame = load_cleaned_dataset(processed_path)
    manifest = create_split_manifest(frame, source_fingerprint)
    destination = reports_dir / "phase4"
    destination.mkdir(parents=True, exist_ok=True)
    saved_manifest = destination / "split_manifest.json"
    if saved_manifest.exists():
        existing = json.loads(saved_manifest.read_text())
        validate_split_manifest(frame, existing, source_fingerprint)
        if existing != manifest:
            raise ValueError("Existing split manifest differs; refusing an implicit re-split")
    _write_json(saved_manifest, manifest)
    assignments = pd.DataFrame([{"customerID": identifier, "split": role}
        for role, info in manifest["splits"].items() for identifier in info["customer_ids"]])
    assignments.to_csv(destination / "split_assignments.csv", index=False)
    train = development_split(frame, manifest, "train")
    validation = development_split(frame, manifest, "validation")
    del frame  # Full-data features cannot accidentally enter any following model call.
    policy = feature_policy()
    policy.to_csv(destination / "feature_policy.csv", index=False)
    configuration = {
        "random_state": 42, "target": "ChurnFlag", "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES), "excluded_features": EXCLUDED_FEATURES,
        "candidate_order": list(MODEL_NAMES), "threshold": THRESHOLD,
        "selection_criterion": "Highest validation average precision among standard/balanced Logistic Regression; exact ties favor standard. Dummy is a reference, not a candidate.",
        "classifier": {"solver": "lbfgs", "effective_regularization": "L2", "C": 1.0, "max_iter": 2000, "tol": 1e-6},
        "preprocessing": {"numeric": "train median + StandardScaler", "categorical": "constant __MISSING__ + OneHotEncoder(drop=first,handle_unknown=ignore)", "remainder": "drop"},
        "model_filename": str((models_dir / "logistic_regression_pipeline.joblib").resolve()),
        "test_evaluation": "prohibited in Phase 4", "refit_after_selection": False,
    }
    # This file is intentionally written before Dummy/LR fitting or evaluation.
    _write_json(destination / "run_config.json", configuration)
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "seaborn"]}}
    _write_json(destination / "environment.json", environment)
    fitted, evidence, prediction_frames, events = {}, {}, [], []
    for name in MODEL_NAMES:
        pipeline = build_pipeline(name)
        fit_info = fit_pipeline(pipeline, train, manifest)
        events.append({"operation": "fit_preprocessing_and_classifier", "model": name, "split": "train",
                       "rows": len(train.y), "customer_ids_sha256": _ids_hash(train.X.index)})
        train_metrics, _ = evaluate_pipeline(pipeline, train, manifest)
        validation_metrics, predictions = evaluate_pipeline(pipeline, validation, manifest)
        for subset in [train, validation]:
            events.append({"operation": "transform_and_predict", "model": name, "split": subset.role,
                           "rows": len(subset.y), "customer_ids_sha256": _ids_hash(subset.X.index)})
        predictions["model_name"] = name
        prediction_frames.append(predictions)
        fitted[name] = pipeline
        evidence[name] = {"fit": fit_info, "classifier_parameters": pipeline.named_steps["classifier"].get_params(),
                          "train": train_metrics, "validation": validation_metrics}
    candidates = ["logistic_standard", "logistic_balanced"]
    selected = max(candidates, key=lambda name: evidence[name]["validation"]["average_precision"])
    metrics = {"schema_version": 1, "selected_model": selected, "selection_criterion": configuration["selection_criterion"],
               "models": evidence, "test_evaluated": False, "threshold": THRESHOLD}
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    if set(all_predictions.customerID) != set(manifest["splits"]["validation"]["customer_ids"]):
        raise ValueError("Validation prediction IDs do not match the reserved development partition")
    if set(all_predictions.customerID) & set(manifest["splits"]["test"]["customer_ids"]):
        raise ValueError("Test predictions are prohibited")
    all_predictions.to_csv(destination / "candidate_validation_predictions.csv", index=False)
    all_predictions.loc[all_predictions.model_name.eq(selected)].to_csv(destination / "validation_predictions.csv", index=False)
    coefficients, coefficient_info = coefficient_report(fitted[selected])
    coefficients.to_csv(destination / "coefficients.csv", index=False)
    _write_json(destination / "coefficient_metadata.json", coefficient_info)
    _write_json(destination / "metrics.json", metrics)
    metric_table(metrics).to_csv(destination / "validation_metrics.csv", index=False)
    preprocessor = fitted[selected].named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["numeric"]
    categorical = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    state = {"fit_role": "train", "training_ids_sha256": _ids_hash(train.X.index),
             "median_statistics": numeric.named_steps["imputer"].statistics_.tolist(),
             "scaler_mean": numeric.named_steps["scaler"].mean_.tolist(),
             "scaler_scale": numeric.named_steps["scaler"].scale_.tolist(),
             "scaler_samples_seen": int(numeric.named_steps["scaler"].n_samples_seen_),
             "training_category_vocabularies": {column: levels.tolist() for column, levels in zip(CATEGORICAL_FEATURES, categorical.categories_)},
             "transformed_feature_names": preprocessor.get_feature_names_out().tolist(),
             "class_weight_balanced_training_weights": {str(k): len(train.y) / (2 * int(train.y.eq(k).sum())) for k in [0, 1]}}
    _write_json(destination / "preprocessing_state.json", state)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "logistic_regression_pipeline.joblib"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=models_dir, suffix=".joblib", delete=False) as handle:
            temporary = Path(handle.name)
        joblib.dump(fitted[selected], temporary, compress=3)
        temporary.replace(model_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    # Trust boundary: reload only the artifact created immediately above, never
    # an arbitrary incoming pickle/joblib. Both checks below use validation only.
    reloaded = joblib.load(model_path)
    _, reloaded_predictions = evaluate_pipeline(reloaded, validation, manifest)
    expected = all_predictions.loc[all_predictions.model_name.eq(selected), "churn_probability"].to_numpy()
    np.testing.assert_allclose(reloaded_predictions.churn_probability, expected, rtol=1e-12, atol=1e-12)
    events.append({"operation": "reload_transform_and_predict", "model": selected, "split": "validation",
                   "rows": len(validation.y), "customer_ids_sha256": _ids_hash(validation.X.index)})
    audit = {"events": events,
             "test_feature_rows_transformed": sum(e["rows"] for e in events if e["split"] == "test" and "predict" in e["operation"]),
             "test_predictions": sum(e["rows"] for e in events if e["split"] == "test" and "predict" in e["operation"]),
             "test_evaluated": any(e["split"] == "test" for e in events),
             "train_only_fitting": all(e["split"] == "train" for e in events if e["operation"] == "fit_preprocessing_and_classifier"),
             "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "reload_validation_consistency": "passed"}
    _write_json(destination / "model_audit.json", audit)
    import matplotlib
    matplotlib.use("Agg")
    from .model_plots import create_model_figures
    figures = create_model_figures(all_predictions, coefficients, destination / "figures")
    (reports_dir / "PHASE4_LOGISTIC_REGRESSION_REPORT.md").write_text(
        logistic_report(configuration, manifest, metrics, policy, coefficients, coefficient_info), encoding="utf-8")
    if hashlib.sha256(processed_path.read_bytes()).hexdigest() != source_fingerprint:
        raise RuntimeError("Phase 2 processed CSV changed during modeling")
    result = {"selected_model": selected, "validation_metrics": evidence[selected]["validation"],
              "split_sizes": {role: data["size"] for role, data in manifest["splits"].items()},
              "manifest_id": manifest["manifest_id"], "model_path": str(model_path.resolve()),
              "reload_validation_consistency": "passed", "test_evaluated": False,
              "figure_paths": [str(path.resolve()) for path in figures]}
    _write_json(destination / "phase4_manifest.json", result)
    return result


def main() -> None:
    """Run Phase 4 with no option to fit/predict/evaluate the reserved test set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-path", type=Path, default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    try:
        result = run_phase4(args.processed_path, args.reports_dir, args.models_dir)
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError) as exc:
        parser.exit(1, f"Phase 4 failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
