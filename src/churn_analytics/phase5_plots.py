"""Validation-only Phase 5 figures; plotting never trains or evaluates test data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve


def create_phase5_figures(predictions: pd.DataFrame, comparison: pd.DataFrame,
                          calibration: pd.DataFrame, thresholds: pd.DataFrame,
                          impurity: pd.DataFrame, permutation: pd.DataFrame, destination: Path) -> list[str]:
    """Save clearly labeled validation ranking/calibration/interpretability charts."""
    if set(predictions.split) != {"validation"}:
        raise ValueError("Development charts accept validation predictions only")
    destination.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = destination / (name + ".png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    for kind in ("roc", "precision_recall"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, frame in predictions.groupby("model_name", sort=False):
            if kind == "roc":
                x, y, _ = roc_curve(frame.y_true, frame.churn_probability)
            else:
                y, x, _ = precision_recall_curve(frame.y_true, frame.churn_probability)
            ax.plot(x, y, label=name.replace("_", " "))
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Random ranking")
        else:
            ax.axhline(predictions.y_true.mean(), ls="--", color="gray", label="Validation prevalence")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate" if kind == "roc" else "Recall",
               ylabel="True positive rate" if kind == "roc" else "Precision", title=f"Validation {kind.replace('_', ' ')} — same 1,409 customers")
        ax.legend(fontsize=8)
        save(fig, "validation_"+kind)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal reliability")
    for name, frame in calibration.groupby("model_name", sort=False):
        ax.plot(frame.mean_probability, frame.observed_proportion, "o-", label=name.replace("_", " "))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed churn proportion",
           title="Validation reliability — 10 fixed equal-width bins\nBin customer counts recorded in calibration_bins.csv")
    ax.legend(fontsize=8)
    save(fig, "validation_calibration")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (name, frame) in zip(axes.flat, thresholds.groupby("model_name", sort=False)):
        ax.plot(frame.threshold, frame.precision, "o-", label="Precision")
        ax.plot(frame.threshold, frame.recall, "o-", label="Recall")
        ax.axvline(.50, ls="--", color="gray", label="Frozen threshold")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Classification threshold", ylabel="Proportion", title=name.replace("_", " "))
        ax.legend(fontsize=8)
    fig.suptitle("Validation threshold trade-offs; no threshold optimized", y=1.02)
    save(fig, "validation_threshold_tradeoffs")

    for frame, x, title, filename in [
        (impurity, "importance", "Tuned RF: aggregated impurity importance (training fitted)", "rf_impurity_importance"),
        (permutation, "ap_decrease_mean", "Tuned RF: validation permutation importance\nMean AP decrease ± repeat SD (5 repeats; can be negative)", "rf_permutation_importance")]:
        fig, ax = plt.subplots(figsize=(10, 8))
        ordered = frame.sort_values(x)
        label = "source_feature" if x == "importance" else "feature"
        ax.barh(ordered[label], ordered[x], xerr=ordered.ap_decrease_std if x != "importance" else None)
        ax.axvline(0, color="black", lw=.8)
        ax.set(xlabel="Normalized impurity decrease" if x == "importance" else "Decrease in Average Precision", title=title)
        save(fig, filename)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(comparison.model_name.str.replace("_", " "), comparison.average_precision)
    ax.set(xlim=(0, 1), xlabel="Validation Average Precision", title="Model selection uses validation ranking performance")
    save(fig, "validation_model_comparison")
    return paths
