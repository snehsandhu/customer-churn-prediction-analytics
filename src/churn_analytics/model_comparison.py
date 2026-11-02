"""Common ranking/classification/calibration summaries, using development only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .modeling import classification_metrics

THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
CANDIDATE_ORDER = ("logistic_standard", "random_forest_baseline", "random_forest_balanced", "random_forest_tuned")


def probability_metrics(y, probabilities, threshold: float = 0.50) -> dict:
    """Extend the original metric definitions with Brier and log loss."""
    result = classification_metrics(y, probabilities, threshold)
    result["brier_score"] = float(brier_score_loss(y, probabilities))
    result["log_loss"] = float(log_loss(y, np.column_stack([1-np.asarray(probabilities), probabilities]), labels=[0, 1]))
    return result


def threshold_table(y, probabilities) -> pd.DataFrame:
    """Describe a predeclared common grid; never choose thresholds from the test."""
    rows = []
    for threshold in THRESHOLDS:
        metrics = classification_metrics(y, probabilities, threshold)
        tn, fp, fn, tp = np.asarray(metrics["confusion_matrix"]).ravel()
        rows.append({"threshold": threshold, **{k: metrics[k] for k in ["precision", "recall", "f1", "accuracy"]},
                     "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return pd.DataFrame(rows)


def calibration_table(y, probabilities, bins: int = 10) -> pd.DataFrame:
    """Fixed equal-width reliability bins with observed counts; no fitted calibrator."""
    classification_metrics(y, probabilities)  # Shared shape/range checks.
    if bins < 2:
        raise ValueError("At least two calibration bins required")
    frame = pd.DataFrame({"truth": np.asarray(y), "probability": np.asarray(probabilities)})
    frame["bin"] = np.minimum((frame.probability * bins).astype(int), bins-1)
    result = frame.groupby("bin").agg(customers=("truth", "size"), mean_probability=("probability", "mean"),
                                     observed_proportion=("truth", "mean")).reset_index()
    result["lower_bound"] = result.bin / bins
    result["upper_bound"] = (result.bin + 1) / bins
    return result


def select_model(validation_metrics: dict) -> str:
    """Highest unrounded validation AP; exact ties prefer LR then simpler fixed RF.

    No test metrics, threshold scores, or parameter search results are accepted.
    The named mapping is restricted to the predeclared comparison candidates.
    """
    if set(validation_metrics) != set(CANDIDATE_ORDER):
        raise ValueError("Expected the four predeclared validation candidates")
    for metrics in validation_metrics.values():
        if metrics.get("split_role") != "validation" or not np.isfinite(metrics["average_precision"]):
            raise ValueError("Selection requires finite validation-only AP")
    return max(CANDIDATE_ORDER, key=lambda name: validation_metrics[name]["average_precision"])
