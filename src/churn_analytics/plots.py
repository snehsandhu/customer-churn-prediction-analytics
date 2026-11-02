"""Consistent Matplotlib/Seaborn figures for the analytical snapshot."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from .analytics import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, churn_by_segment, correlation_tables

COLORS = {"No": "#30708B", "Yes": "#C04E38"}
LABELS = {"tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges (dataset charge units)",
          "TotalCharges": "Total charges (dataset charge units)"}


def _save(fig: plt.Figure, directory: Path, name: str) -> Path:
    """Save and close a figure so batch/notebook execution does not leak figures."""
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _rate_axis(ax: plt.Axes, frame: pd.DataFrame, feature: str) -> None:
    """Draw within-segment rates, Wilson intervals and denominators on 0–100%."""
    summary = churn_by_segment(frame, feature)
    y = np.arange(len(summary))
    rates = summary["churn_pct"].to_numpy(dtype=float)
    error = np.array([rates - summary["ci_low"].to_numpy() * 100,
                      summary["ci_high"].to_numpy() * 100 - rates])
    ax.barh(y, rates, color=COLORS["Yes"], alpha=0.9)
    ax.errorbar(rates, y, xerr=np.maximum(0, error), fmt="none", color="#333333", capsize=3, linewidth=1)
    ax.set_yticks(y, [textwrap.fill(str(value), 25) for value in summary[feature]])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlabel("Within-segment churn percentage (95% Wilson interval)")
    ax.set_title(f"Churn by {feature}", loc="left", weight="bold")
    for i, row in summary.iterrows():
        if row["customer_count"]:
            ax.text(min(float(row["churn_pct"]) + 3, 82), i,
                    f'{row["churn_pct"]:.1f}% | n={int(row["customer_count"]):,}', va="center", fontsize=9)
        else:
            ax.text(2, i, "No observations", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)


def _segment_heatmap(ax: plt.Axes, frame: pd.DataFrame, row: str, column: str) -> None:
    """Display combined segment rates with sample sizes; mark small groups."""
    table = churn_by_segment(frame, [row, column])
    rates = table.pivot(index=row, columns=column, values="churn_pct")
    counts = table.pivot(index=row, columns=column, values="customer_count").fillna(0)
    annotations = rates.copy().astype(object)
    for r in rates.index:
        for c in rates.columns:
            n = int(counts.loc[r, c])
            annotations.loc[r, c] = f'{rates.loc[r, c]:.1f}%\nn={n:,}' + ("*" if n < 30 else "") if n else ""
    sns.heatmap(rates, annot=annotations, fmt="", mask=counts.eq(0), vmin=0, vmax=100,
                cmap="YlOrRd", linewidths=1, ax=ax, cbar_kws={"label": "Churn (%)"})
    ax.set_title(f"{row} × {column}\n* n<30: interpret cautiously; blank = no observations", loc="left")
    ax.tick_params(axis="x", labelrotation=35)
    ax.tick_params(axis="y", labelrotation=0)


def create_figures(frame: pd.DataFrame, directory: str | Path) -> list[Path]:
    """Generate all requested charts from the same validated analytical frame.

    Numerical missing values are excluded only from plots requiring those values;
    sample coverage is labeled. No data is sampled, clipped away, or winsorized.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with sns.axes_style("whitegrid"), plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = frame["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
        ax.bar(["No (non-churned)", "Yes (churned)"], counts, color=[COLORS["No"], COLORS["Yes"]])
        for i, n in enumerate(counts):
            ax.text(i, n + max(counts.max() * 0.025, 1), f"{n:,} ({n / len(frame):.2%})", ha="center")
        ax.set_ylim(0, max(counts.max() * 1.18, 1))
        ax.set_ylabel("Customers")
        ax.set_title(f"IBM Telco snapshot: churn distribution (n={len(frame):,})", loc="left", weight="bold")
        paths.append(_save(fig, directory, "churn_distribution"))

        for feature, name in (("Contract", "churn_rate_by_contract"), ("InternetService", "churn_rate_by_internet_service"),
                              ("PaymentMethod", "churn_rate_by_payment_method"), ("TenureGroup", "churn_rate_by_tenure_group")):
            fig, ax = plt.subplots(figsize=(10, max(3.6, frame[feature].nunique() * 0.7)))
            _rate_axis(ax, frame, feature)
            paths.append(_save(fig, directory, name))

        fig, axes = plt.subplots(1, 3, figsize=(21, 5), layout="constrained")
        for ax, feature in zip(axes, ["TechSupport", "OnlineSecurity", "SeniorCitizen"]):
            _rate_axis(ax, frame, feature)
        paths.append(_save(fig, directory, "churn_by_selected_services"))

        for feature, name in (("tenure", "tenure_distribution_by_churn"), ("MonthlyCharges", "monthly_charges_distribution_by_churn")):
            fig, ax = plt.subplots(figsize=(9, 4.5))
            sns.histplot(data=frame, x=feature, hue="Churn", hue_order=["No", "Yes"], palette=COLORS,
                         stat="density", common_norm=False, bins=24, element="step", fill=False, ax=ax)
            ax.set_xlabel(LABELS[feature])
            ax.set_ylabel("Within-class density")
            ax.set_title(f"{feature} distribution by churn | all {len(frame):,} customers", loc="left", weight="bold")
            ax.set_xlim(left=0)
            paths.append(_save(fig, directory, name))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
        for ax, feature in zip(axes, NUMERICAL_FEATURES):
            sns.histplot(data=frame, x=feature, bins=30, color=COLORS["No"], ax=ax)
            ax.set_xlabel(LABELS[feature])
            ax.set_ylabel("Customers")
            ax.set_title(f'{feature}: n={frame[feature].notna().sum():,}')
            ax.set_xlim(left=0)
        paths.append(_save(fig, directory, "numerical_histograms"))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
        for ax, feature in zip(axes, NUMERICAL_FEATURES):
            sns.boxplot(data=frame, x="Churn", y=feature, order=["No", "Yes"], color="#8DB7C5", ax=ax)
            ax.set_ylabel(LABELS[feature])
            ax.set_ylim(bottom=0)
            ax.set_title(f'{feature} by churn\nobserved n={frame[feature].notna().sum():,}; outliers retained')
        paths.append(_save(fig, directory, "numerical_boxplots_by_churn"))

        fig, axes = plt.subplots(4, 4, figsize=(24, 21), layout="constrained")
        for ax, feature in zip(axes.flat, CATEGORICAL_FEATURES):
            counts = frame[feature].value_counts().sort_index()
            ax.barh(range(len(counts)), counts.to_numpy(), color=COLORS["No"])
            ax.set_yticks(range(len(counts)), [textwrap.fill(str(c), 18) for c in counts.index])
            ax.set_title(feature, loc="left", weight="bold")
            ax.set_xlim(0, max(counts.max() * 1.27, 1))
            ax.set_xlabel("Customers")
            for i, n in enumerate(counts):
                ax.text(n + 15, i, f"{n:,}", va="center", fontsize=9)
        paths.append(_save(fig, directory, "categorical_frequency_counts"))

        corr = correlation_tables(frame)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        sns.heatmap(corr["pearson"], annot=True, fmt=".2f", vmin=-1, vmax=1, center=0,
                    cmap="vlag", square=True, ax=ax, cbar_kws={"label": "Pearson correlation"})
        ax.set_title(f'Numerical correlation (pairwise observations)\nTotalCharges pairs: n={frame.TotalCharges.notna().sum():,}; other pairs: n={len(frame):,}', loc="left")
        paths.append(_save(fig, directory, "numerical_correlation_heatmap"))

        for feature, name in (("tenure", "tenure_vs_total_charges"), ("MonthlyCharges", "monthly_charges_vs_total_charges")):
            observed = frame.dropna(subset=[feature, "TotalCharges"])
            fig, ax = plt.subplots(figsize=(9, 5.5))
            sns.scatterplot(data=observed, x=feature, y="TotalCharges", hue="Churn", hue_order=["No", "Yes"],
                            palette=COLORS, alpha=0.25, s=15, linewidth=0, ax=ax, rasterized=True)
            ax.set_xlabel(LABELS[feature])
            ax.set_ylabel(LABELS["TotalCharges"])
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.set_title(f'{feature} vs TotalCharges | observed n={len(observed):,}', loc="left", weight="bold")
            paths.append(_save(fig, directory, name))

        for row, col, name in (("Contract", "TenureGroup", "churn_contract_tenure"),
                               ("InternetService", "TechSupport", "churn_internet_technical_support"),
                               ("Contract", "MonthlyChargeGroup", "churn_contract_monthly_charges")):
            fig, ax = plt.subplots(figsize=(10, 5.5))
            _segment_heatmap(ax, frame, row, col)
            paths.append(_save(fig, directory, name))
    return paths
