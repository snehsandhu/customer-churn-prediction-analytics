"""Validation-only classification figures; no reserved-test input is accepted."""
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def create_model_figures(predictions: pd.DataFrame, coefficients: pd.DataFrame, directory: str | Path) -> list[Path]:
    """Render all candidate validation curves and selected-model coefficients."""
    if predictions.empty or set(predictions["split"]) != {"validation"}:
        raise ValueError("Model figures require validation predictions only; test data is reserved")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = list(predictions.model_name.unique())
    paths = []
    def save(fig, name):
        path = directory / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)
    with sns.axes_style("whitegrid"), plt.rc_context({"font.size": 10}):
        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False, layout="constrained")
        for ax, name in zip(axes.flat, names):
            subset = predictions.loc[predictions.model_name.eq(name)]
            ConfusionMatrixDisplay.from_predictions(subset.y_true, subset.prediction_at_0_50,
                labels=[0, 1], display_labels=["No churn", "Churn"], colorbar=False, cmap="Blues", ax=ax)
            ax.set_title(f'{name}\nValidation n={len(subset):,}; threshold 0.50')
        save(fig, "validation_confusion_matrices")

        fig, ax = plt.subplots(figsize=(8, 6))
        for name in names:
            subset = predictions.loc[predictions.model_name.eq(name)]
            RocCurveDisplay.from_predictions(subset.y_true, subset.churn_probability, name=name, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance ranking")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title("Validation ROC curves — churn is the positive class")
        ax.legend(loc="lower right", fontsize=9)
        save(fig, "validation_roc_curves")

        fig, ax = plt.subplots(figsize=(8, 6))
        for name in names:
            subset = predictions.loc[predictions.model_name.eq(name)]
            PrecisionRecallDisplay.from_predictions(subset.y_true, subset.churn_probability, name=name, ax=ax)
        prevalence = predictions.loc[predictions.model_name.eq(names[0]), "y_true"].mean()
        ax.axhline(prevalence, linestyle="--", color="gray", label=f"Validation prevalence: {prevalence:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title("Validation precision–recall curves (legend reports average precision)")
        ax.legend(fontsize=8)
        save(fig, "validation_precision_recall_curves")

        fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4.5), squeeze=False, layout="constrained")
        for ax, name in zip(axes.flat, names):
            subset = predictions.loc[predictions.model_name.eq(name)]
            for label, color in [(0, "#30708B"), (1, "#C04E38")]:
                ax.hist(subset.loc[subset.y_true.eq(label), "churn_probability"], bins=np.linspace(0, 1, 21),
                        histtype="step", linewidth=2, density=True, color=color, label=f"Actual class {label}")
            ax.axvline(0.5, color="gray", linestyle="--", label="Threshold 0.50")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Predicted churn probability")
            ax.set_ylabel("Within-class density")
            ax.set_title(f"{name}\nValidation probabilities")
            ax.legend(fontsize=8)
        save(fig, "validation_probability_distributions")

        selected = pd.concat([coefficients.nsmallest(10, "coefficient"), coefficients.nlargest(10, "coefficient")]).drop_duplicates("transformed_feature").sort_values("coefficient")
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.barh(range(len(selected)), selected.coefficient,
                color=["#30708B" if n < 0 else "#C04E38" for n in selected.coefficient])
        ax.set_yticks(range(len(selected)), [textwrap.fill(s, 45) for s in selected.transformed_feature])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Regularized coefficient (conditional log-odds scale)")
        ax.set_title("Selected Logistic Regression: largest signed fitted associations\nNumeric effects are per training SD; categorical effects use saved reference levels", loc="left")
        save(fig, "logistic_coefficient_associations")
    return paths
