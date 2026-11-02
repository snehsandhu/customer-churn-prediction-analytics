"""Freeze provenance before a durable, one-time held-out evaluation claim.

This is an application guard, not a tamper-proof security boundary. Preserve the
ledger with artifacts. Failed or interrupted attempts remain claimed and block
automatic retries; never delete it to obtain another look at the test set.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from .inference import load_churn_predictor, normalize_inference_input
from .model_comparison import probability_metrics
from .model_splits import validate_target
from .phase5_data import _read_selected_rows, load_membership, read_json, sha256, write_new_json

CODE_FILES = tuple("src/churn_analytics/"+name+".py" for name in (
    "phase5", "phase5_data", "random_forest", "model_comparison", "final_evaluation",
    "phase5_plots", "inference", "modeling", "model_splits", "cleaning"))


def utc_now() -> str:
    """Return an explicit UTC evidence timestamp."""
    return datetime.now(timezone.utc).isoformat()


def selection_identity(selection: dict) -> str:
    """Hash all frozen selection content except its own identity field."""
    payload = {k: v for k, v in selection.items() if k != "selection_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def freeze_selection(root: str | Path, selected_model: str, artifact: str, validation_metrics: dict,
                     protocol: dict, evidence_files: list[str]) -> dict:
    """Exclusively persist the candidate, threshold, inputs, code and provenance."""
    root = Path(root)
    destination = root / "reports/phase5"
    if (destination / "test_consumption.json").exists():
        raise ValueError("Test set already claimed/consumed; selection is locked")
    membership = load_membership(root)
    if validation_metrics.get("split_role") != "validation" or protocol["threshold"] != 0.50:
        raise ValueError("Freeze requires validation evidence and predeclared threshold")
    from .modeling import FEATURES
    protected = ["data/raw/Telco-Customer-Churn.csv", "data/processed/telco_churn_cleaned.csv",
        "reports/phase4/split_manifest.json", "reports/phase4/split_assignments.csv",
        "reports/phase4/validation_predictions.csv", "reports/phase4/metrics.json",
        "models/logistic_regression_pipeline.joblib", artifact, *evidence_files, *CODE_FILES]
    selection = {"schema_version": 1, "model_version": "phase5-v1", "frozen_at_utc": utc_now(),
        "selected_model": selected_model, "artifact_path": artifact, "artifact_sha256": sha256(root/artifact),
        "threshold": protocol["threshold"], "features": list(FEATURES),
        "input_schema": {"numeric": ["tenure", "MonthlyCharges"], "categorical": list(FEATURES[2:])},
        "normalization": "inference.normalize_inference_input -> modeling.select_features; F01 contract",
        "training_policy": "original train only; no train+validation refit", "calibration": "assessment only; no calibrator fitted",
        "split_manifest_id": membership["manifest_id"], "dataset_sha256": membership["dataset_sha256"],
        "training_ids": membership["splits"]["train"]["customer_ids"],
        "validation_metrics": validation_metrics, "protocol": protocol,
        "frozen_files": {rel: sha256(root/rel) for rel in sorted(set(protected))},
        "test_policy": "one exclusive durable claim before label access; failed claims also block retries"}
    selection["selection_id"] = selection_identity(selection)
    write_new_json(destination / "final_selection_manifest.json", selection)
    return selection


def _check_frozen(root: Path, selection: dict, membership: dict) -> None:
    """Reject drift before any test label is read or prediction performed."""
    from .modeling import FEATURES
    if selection["selection_id"] != selection_identity(selection):
        raise ValueError("Frozen selection identity mismatch")
    if selection["split_manifest_id"] != membership["manifest_id"] or selection["dataset_sha256"] != membership["dataset_sha256"]:
        raise ValueError("Frozen dataset/split differs")
    if selection["threshold"] != selection["protocol"]["threshold"] or tuple(selection["features"]) != FEATURES:
        raise ValueError("Frozen threshold/features differ")
    if selection["training_ids"] != membership["splits"]["train"]["customer_ids"]:
        raise ValueError("Frozen training provenance differs")
    for rel, expected in selection["frozen_files"].items():
        path = root/rel
        if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError(f"Frozen file changed: {rel}")
    if sha256(root/selection["artifact_path"]) != selection["artifact_sha256"]:
        raise ValueError("Frozen model identity differs")


def evaluate_final_once(root: str | Path) -> dict:
    """Evaluate only the frozen train-fitted winner; refuse every repeated attempt.

    No call to a fit method or model-selection function exists on this path.
    Claim is fsynced before reading test fields. Predictions/metrics are exclusively
    created, then the ledger records completion. Failure leaves the claim in place.
    """
    root = Path(root)
    destination = root / "reports/phase5"
    ledger = destination / "test_consumption.json"
    if ledger.exists() or (destination / "final_test_metrics.json").exists() or (destination / "final_test_predictions.csv").exists():
        raise ValueError("Final test already claimed/consumed; repeated evaluation prohibited")
    selection = read_json(destination / "final_selection_manifest.json")
    membership = load_membership(root)
    _check_frozen(root, selection, membership)
    predictor = load_churn_predictor(root/selection["artifact_path"],
        expected_sha256=selection["artifact_sha256"], threshold=selection["threshold"])
    model = predictor._pipeline
    if getattr(model, "_phase4_manifest_id", None) != membership["manifest_id"] or list(getattr(model, "_phase4_training_ids", [])) != selection["training_ids"]:
        raise ValueError("Fitted estimator training provenance mismatch")
    claim = {"status": "claimed_before_test_access", "claimed_at_utc": utc_now(),
             "selection_id": selection["selection_id"], "artifact_sha256": selection["artifact_sha256"],
             "evaluation_attempts": 1, "test_consumed": True}
    with ledger.open("x", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    ids = membership["splits"]["test"]["customer_ids"]
    # The first materialization of real held-out fields occurs only here.
    frame = _read_selected_rows(root/"data/processed/telco_churn_cleaned.csv", ids)
    frame = frame.set_index("customerID", drop=False).loc[ids]
    truth = validate_target(frame)
    if {str(k): int(truth.eq(k).sum()) for k in [0, 1]} != membership["splits"]["test"]["class_counts"]:
        raise ValueError("Test support differs from original administrative metadata; claim retained")
    X = frame.loc[:, selection["features"]]
    normalize_inference_input(X)  # Explicitly validate before the single scoring call.
    probability = predictor.predict_proba(X)[:, 1]
    metrics = probability_metrics(truth, probability, selection["threshold"])
    metrics.update(split_role="test", selected_model=selection["selected_model"], selection_id=selection["selection_id"],
        evaluated_at_utc=utc_now(), artifact_sha256=selection["artifact_sha256"], training_policy=selection["training_policy"])
    pd.DataFrame({"customerID": ids, "y_true": truth.to_numpy(), "churn_probability": probability,
                  "prediction": (probability >= selection["threshold"]).astype(int)}).to_csv(
        destination/"final_test_predictions.csv", index=False, mode="x")
    write_new_json(destination/"final_test_metrics.json", metrics)
    write_new_json(destination/"final_model_metadata.json", {**selection, "final_test_metrics": metrics,
        "test_consumed": True, "consumption_ledger": "reports/phase5/test_consumption.json"})
    claim.update(status="completed", completed_at_utc=utc_now(), test_customers=len(ids),
                 metrics_sha256=sha256(destination/"final_test_metrics.json"))
    # Update only the existing claim, never its identity/attempt count. Even a
    # failed completion write leaves a blocking ledger and cannot permit a retry.
    with ledger.open("w", encoding="utf-8") as handle:
        json.dump(claim, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    return metrics
