"""Train/compare/freeze Phase 5 candidates; final evaluation is an explicit command."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from .data import PROJECT_ROOT
from .final_evaluation import evaluate_final_once, freeze_selection
from .inference import ChurnPredictor, load_churn_predictor
from .model_comparison import CANDIDATE_ORDER, THRESHOLDS, calibration_table, probability_metrics, select_model, threshold_table
from .modeling import FEATURES, evaluate_pipeline, guard_development_data
from .phase5_data import load_development, load_membership, read_json, sha256, write_new_json
from .random_forest import SEARCH_SPACE, build_random_forest, fit_random_forest, impurity_importance, tune_random_forest, validation_permutation_importance


def run_phase5(root: str | Path = PROJECT_ROOT, *, n_iter: int = 8, folds: int = 3,
               search_space=None, rf_parameters=None, permutation_repeats: int = 5) -> dict:
    """Develop once on original train/validation, then freeze without test access.

    Test evaluation requires a separate guarded call. Existing output directories
    are never reused: no accidental artifact replacement or implicit new selection.
    Keyword budgets permit fast synthetic tests; real defaults are predeclared.
    """
    root = Path(root)
    membership = load_membership(root)
    destination = root/"reports/phase5"
    models = root/"models/phase5"
    if destination.exists() or models.exists():
        raise ValueError("Phase 5 outputs already exist; refusing redevelopment/overwrite")
    train, validation = load_development(root, membership)
    guard_development_data(train, membership, fitting=True)
    guard_development_data(validation, membership)
    lr_path = root/"models/logistic_regression_pipeline.joblib"
    lr_hash = read_json(root/"reports/phase4/model_audit.json")["model_sha256"]
    predictor = load_churn_predictor(lr_path, expected_sha256=lr_hash)
    lr = predictor._pipeline
    destination.mkdir()
    models.mkdir()
    environment = {"python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "seaborn"]}}
    # Freeze the development protocol BEFORE RF fitting or inspecting its scores.
    protocol = {"random_state": 42, "search_candidates": n_iter, "cv_folds": folds,
        "cv": "StratifiedKFold(shuffle=True, random_state=42), original train only",
        "search_space": SEARCH_SPACE if search_space is None else search_space,
        "n_jobs": 1, "candidate_order": list(CANDIDATE_ORDER),
        "selection_rule": "Highest unrounded validation AP; exact ties prefer LR, then fixed baseline RF, balanced RF, tuned RF",
        "threshold": 0.50, "threshold_policy": "Keep 0.50 for every candidate; grid is descriptive only; no known intervention cost",
        "threshold_grid": list(THRESHOLDS), "calibration": "Assess Brier/log loss and 10 fixed bins; do not fit a calibrator",
        "permutation_repeats": permutation_repeats, "training_policy": "original train only, including CV refit; no train+validation refit",
        "environment": environment}
    write_new_json(destination/"development_protocol.json", protocol)
    fitted = {"logistic_standard": lr}
    training_seconds = {"logistic_standard": None}  # Historical training not timed again.
    for name, weight in [("random_forest_baseline", None), ("random_forest_balanced", "balanced_subsample")]:
        model = build_random_forest(**({} if rf_parameters is None else rf_parameters), class_weight=weight)
        training_seconds[name] = fit_random_forest(model, train, membership)
        fitted[name] = model
    search, search_audit = tune_random_forest(train, membership, n_iter=n_iter, folds=folds, search_space=search_space)
    fitted["random_forest_tuned"] = search.best_estimator_
    training_seconds["random_forest_tuned"] = search_audit["seconds"]
    pd.DataFrame(search.cv_results_).to_csv(destination/"random_forest_cv_results.csv", index=False)
    write_new_json(destination/"search_audit.json", search_audit)
    metrics, predictions, comparison, thresholds, calibration, paths = {}, [], [], [], [], {}
    historical = read_json(root/"reports/phase4/metrics.json")["models"]["logistic_standard"]["validation"]
    for name in CANDIDATE_ORDER:
        model = fitted[name]
        _, pred = evaluate_pipeline(model, validation, membership)
        p = pred.churn_probability.to_numpy()
        result = probability_metrics(validation.y, p)
        result["split_role"] = "validation"
        metrics[name] = result
        if name == "logistic_standard":
            saved = pd.read_csv(root/"reports/phase4/validation_predictions.csv").set_index("customerID").loc[validation.X.index]
            np.testing.assert_allclose(p, saved.churn_probability, rtol=1e-12, atol=1e-12)
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
                np.testing.assert_allclose(result[key], historical[key], rtol=1e-12, atol=1e-12)
            assert result["confusion_matrix"] == historical["confusion_matrix"]
            paths[name] = "models/logistic_regression_pipeline.joblib"
        else:
            filename = "random_forest_pipeline.joblib" if name.endswith("tuned") else name+".joblib"
            paths[name] = "models/phase5/"+filename
            with (root/paths[name]).open("xb") as handle:
                joblib.dump(model, handle, compress=3)
        # Trust only locally generated, checksum-identified artifacts. Reload must
        # reproduce the same full validation predictions before selection freezes.
        adapter = load_churn_predictor(root/paths[name], expected_sha256=sha256(root/paths[name]))
        elapsed = []
        for _ in range(3):
            start = perf_counter()
            reloaded = adapter.predict_proba(validation.X)[:, 1]
            elapsed.append(perf_counter()-start)
            np.testing.assert_allclose(reloaded, p, rtol=1e-12, atol=1e-12)
        row = {"model_name": name, **{k: result[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "brier_score", "log_loss"]},
            "training_seconds": training_seconds[name], "validation_inference_median_seconds": float(np.median(elapsed)),
            "inference_repeats": 3, "inference_customers": len(validation.y)}
        tn, fp, fn, tp = np.asarray(result["confusion_matrix"]).ravel()
        row.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
        comparison.append(row)
        pred["model_name"] = name
        predictions.append(pred)
        for function, collection in [(threshold_table, thresholds), (calibration_table, calibration)]:
            table = function(validation.y, p)
            table["model_name"] = name
            collection.append(table)
    table, pred_table = pd.DataFrame(comparison), pd.concat(predictions, ignore_index=True)
    threshold_data, calibration_data = pd.concat(thresholds, ignore_index=True), pd.concat(calibration, ignore_index=True)
    table.to_csv(destination/"model_comparison.csv", index=False)
    pred_table.to_csv(destination/"validation_predictions.csv", index=False)
    threshold_data.to_csv(destination/"threshold_analysis.csv", index=False)
    calibration_data.to_csv(destination/"calibration_bins.csv", index=False)
    write_new_json(destination/"random_forest_validation_metrics.json", metrics)
    detail, grouped = impurity_importance(fitted["random_forest_tuned"])
    detail.to_csv(destination/"transformed_feature_importance.csv", index=False)
    grouped.to_csv(destination/"feature_importance.csv", index=False)
    perm = validation_permutation_importance(fitted["random_forest_tuned"], validation, membership, repeats=permutation_repeats)
    perm.to_csv(destination/"permutation_importance.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    from .phase5_plots import create_phase5_figures
    figures = create_phase5_figures(pred_table, table, calibration_data, threshold_data, grouped, perm, destination/"figures")
    selected = select_model(metrics)
    write_new_json(destination/"development_audit.json", {"train_ids": train.X.index.tolist(),
        "validation_ids": validation.X.index.tolist(), "test_feature_rows_accessed": 0, "test_predictions": 0,
        "all_candidate_reload_checks": "passed", "existing_lr_sha256": lr_hash,
        "candidate_parameters": {name: model.named_steps["classifier"].get_params() for name, model in fitted.items()},
        "split_sizes": {r: m["size"] for r,m in membership["splits"].items()}, "manifest_id": membership["manifest_id"]})
    evidence_files = [str(path.relative_to(root)) for path in destination.iterdir() if path.is_file()]
    selection = freeze_selection(root, selected, paths[selected], metrics[selected], protocol, evidence_files)
    if sha256(lr_path) != lr_hash:
        raise ValueError("Original LR artifact changed")
    return {"selected_model": selected, "validation_metrics": metrics, "selection_id": selection["selection_id"],
            "test_evaluated": False, "figures": figures}


def main() -> None:
    """Separate development, explicit one-time evaluation, and read-only status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["develop", "final-evaluate", "status"], nargs="?", default="status")
    args = parser.parse_args()
    try:
        if args.action == "develop":
            result = run_phase5()
        elif args.action == "final-evaluate":
            result = evaluate_final_once(PROJECT_ROOT)
        else:
            path = PROJECT_ROOT/"reports/phase5/test_consumption.json"
            result = read_json(path) if path.exists() else {"test_consumed": False}
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError) as exc:
        parser.exit(1, f"Phase 5 stopped: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
