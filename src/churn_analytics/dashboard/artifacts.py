"""Allowlisted local evidence, with model/metadata identity checked before loading."""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path

import pandas as pd

from ..data import PROJECT_ROOT
from ..final_evaluation import selection_identity
from ..modeling import FEATURES
from ..phase5_data import sha256

MODEL_PATH = "models/phase5/random_forest_pipeline.joblib"
TABLES = {
    "comparison": "reports/phase5/model_comparison.csv",
    "importance": "reports/phase5/feature_importance.csv",
    "permutation": "reports/phase5/permutation_importance.csv",
    "calibration": "reports/phase5/calibration_bins.csv",
    "thresholds": "reports/phase5/threshold_analysis.csv",
}
FIGURES = {name: f"reports/phase5/figures/{name}.png" for name in (
    "validation_roc", "validation_precision_recall", "validation_calibration",
    "validation_threshold_tradeoffs", "rf_impurity_importance", "rf_permutation_importance")}
METRIC_KEYS = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "brier_score", "log_loss")


class DashboardError(ValueError):
    """An actionable, non-sensitive message suitable for the application UI."""


def read_local_json(root: Path, relative: str) -> dict:
    """Read developer-selected JSON; callers never pass user-supplied paths."""
    try:
        value = json.loads((root/relative).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Not an object")
        return value
    except (OSError, ValueError, TypeError) as exc:
        raise DashboardError("Required project evidence is missing or unreadable. Restore the supplied artifacts; do not regenerate models.") from exc


def verify_final_bundle(root: str | Path = PROJECT_ROOT) -> dict:
    """Check frozen selection, completed consumption, metadata and model identity.

    This never materializes split rows or calls evaluation, fitting or prediction.
    Checksum identity is meaningful only for this trusted local project bundle.
    """
    root = Path(root)
    message = "Model evidence is inconsistent or unavailable. Restore the trusted Phase 5 bundle."
    try:
        selection = read_local_json(root, "reports/phase5/final_selection_manifest.json")
        metadata = read_local_json(root, "reports/phase5/final_model_metadata.json")
        ledger = read_local_json(root, "reports/phase5/test_consumption.json")
        if selection["selection_id"] != selection_identity(selection):
            raise ValueError("Selection identity")
        if any(metadata.get(k) != v for k, v in selection.items()):
            raise ValueError("Metadata differs from frozen selection")
        if selection["artifact_path"] != MODEL_PATH or selection["selected_model"] != "random_forest_tuned":
            raise ValueError("Unexpected selected artifact")
        if selection["threshold"] != 0.50 or selection["protocol"]["threshold"] != 0.50 or tuple(selection["features"]) != FEATURES:
            raise ValueError("Feature/threshold mismatch")
        if ledger["status"] != "completed" or ledger["test_consumed"] is not True or ledger["evaluation_attempts"] != 1:
            raise ValueError("Invalid consumed ledger")
        if ledger["selection_id"] != selection["selection_id"] or ledger["artifact_sha256"] != selection["artifact_sha256"]:
            raise ValueError("Ledger identity mismatch")
        model = (root/MODEL_PATH).resolve()
        if not model.is_relative_to(root.resolve()) or sha256(model) != selection["artifact_sha256"]:
            raise ValueError("Artifact fingerprint mismatch")
        if sha256(root/"reports/phase5/final_test_metrics.json") != ledger["metrics_sha256"]:
            raise ValueError("Saved summary identity mismatch")
        if selection["protocol"]["environment"]["packages"]["scikit-learn"] != version("scikit-learn"):
            raise ValueError("Incompatible sklearn version")
        return selection
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DashboardError(message) from exc


def load_table(name: str, root: str | Path = PROJECT_ROOT) -> pd.DataFrame:
    """Read only a registered aggregate table, never customer-level predictions."""
    if name not in TABLES:
        raise DashboardError("Choose a registered aggregate report.")
    try:
        return pd.read_csv(Path(root)/TABLES[name])
    except (OSError, ValueError) as exc:
        raise DashboardError("This aggregate report is unavailable. Restore the supplied report file.") from exc


def figure_bytes(name: str, root: str | Path = PROJECT_ROOT) -> bytes:
    """Return only a registered local figure, without exposing its filesystem path."""
    if name not in FIGURES:
        raise DashboardError("Choose a registered figure.")
    try:
        return (Path(root)/FIGURES[name]).read_bytes()
    except OSError as exc:
        raise DashboardError("This saved figure is unavailable.") from exc


def saved_performance(root: str | Path = PROJECT_ROOT) -> dict:
    """Return a public aggregate-only view; never expose IDs or individual labels."""
    root = Path(root)
    try:
        selection = verify_final_bundle(root)
        final = read_local_json(root, "reports/phase5/final_test_metrics.json")
        if final["selection_id"] != selection["selection_id"] or final["selected_model"] != selection["selected_model"]:
            raise ValueError("Final summary mismatch")
        return {"model": selection["selected_model"], "threshold": selection["threshold"],
                "version": selection["model_version"], "validation_metrics": {k: selection["validation_metrics"][k] for k in METRIC_KEYS},
                "test_metrics": {k: final[k] for k in METRIC_KEYS}, "confusion_matrix": final["confusion_matrix"],
                "class_support": final["class_support"], "customers": final["customers"], "test_consumed": True}
    except (KeyError, TypeError, ValueError) as exc:
        raise DashboardError("The saved evaluation summary is unavailable or inconsistent.") from exc
