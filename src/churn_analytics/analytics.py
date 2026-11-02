"""Descriptive customer KPIs and transparent exploratory statistical summaries."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .cleaning import CATEGORICAL_DOMAINS

NUMERICAL_FEATURES = ("tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_FEATURES = (*[c for c in CATEGORICAL_DOMAINS if c != "Churn"], "SeniorCitizen")
# A fixed exploratory family; do not search all combinations for small p-values.
ASSOCIATION_FEATURES = (
    "Contract", "TenureGroup", "MonthlyChargeGroup", "InternetService",
    "PaymentMethod", "TechSupport", "OnlineSecurity", "SeniorCitizen",
)


def _check_flags(frame: pd.DataFrame) -> None:
    """Reject missing or invalid binary outcomes before calculating denominators."""
    if "ChurnFlag" not in frame or not frame["ChurnFlag"].isin([0, 1]).all():
        raise ValueError("Analytics requires nonmissing binary ChurnFlag values (0/1)")


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return Wilson binomial bounds; an empty denominator has undefined bounds.

    These model-based intervals assume independent Bernoulli observations. They
    cannot establish that IBM's illustrative snapshot is a random population sample.
    """
    if not 0 < confidence < 1 or total < 0 or successes < 0 or successes > total:
        raise ValueError("Invalid counts or confidence level")
    if int(total) != total or int(successes) != successes:
        raise ValueError("Counts must be whole numbers")
    if total == 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, float(center - margin)), min(1.0, float(center + margin))


def customer_kpis(frame: pd.DataFrame) -> dict:
    """Summarize this snapshot; No means non-churned, not longitudinal retention.

    Proportions are fractions in [0, 1]. Charges are dataset charge units, not
    revenue loss, customer lifetime value or an estimate of future savings.
    """
    _check_flags(frame)
    total = len(frame)
    churned = int(frame["ChurnFlag"].sum())
    low, high = wilson_interval(churned, total)
    return {
        "total_customers": total, "churn_count": churned, "retained_count": total - churned,
        "churn_proportion": churned / total if total else float("nan"),
        "retention_proportion": (total - churned) / total if total else float("nan"),
        "churn_ci_low": low, "churn_ci_high": high,
        "mean_monthly_charges": float(frame["MonthlyCharges"].mean()),
        "median_monthly_charges": float(frame["MonthlyCharges"].median()),
        "mean_tenure_months": float(frame["tenure"].mean()),
        "median_tenure_months": float(frame["tenure"].median()),
        "mean_total_charges_observed": float(frame["TotalCharges"].mean()),
        "median_total_charges_observed": float(frame["TotalCharges"].median()),
        "total_charges_observed_customers": int(frame["TotalCharges"].notna().sum()),
        "total_charges_missing_customers": int(frame["TotalCharges"].isna().sum()),
    }


def churn_by_segment(frame: pd.DataFrame, columns: str | Sequence[str]) -> pd.DataFrame:
    """Use each segment's customer count as the churn-rate denominator.

    Include unused categorical levels with n=0; their rates/intervals are NaN,
    never zero. Multi-column categorical groups include empty combinations.
    Small groups (n<30) are flagged, not removed or interpreted as robust evidence.
    """
    _check_flags(frame)
    keys = [columns] if isinstance(columns, str) else list(columns)
    if not keys or any(c not in frame for c in keys):
        raise ValueError("Supply one or more existing segment columns")
    summary = frame.groupby(keys, observed=False, dropna=False)["ChurnFlag"].agg(
        customer_count="size", churn_count="sum",
    ).reset_index()
    summary["retained_count"] = summary["customer_count"] - summary["churn_count"]
    denominator = summary["customer_count"].replace(0, np.nan)
    summary["churn_rate"] = summary["churn_count"] / denominator
    summary["churn_pct"] = 100 * summary["churn_rate"]
    intervals = [wilson_interval(int(k), int(n)) for k, n in zip(summary["churn_count"], summary["customer_count"])]
    summary["ci_low"] = [ci[0] for ci in intervals]
    summary["ci_high"] = [ci[1] for ci in intervals]
    summary["small_segment"] = summary["customer_count"].lt(30)
    return summary


def customer_segment_counts(frame: pd.DataFrame, columns: str | Sequence[str]) -> pd.DataFrame:
    """Return counts and proportions of customers belonging to each segment."""
    keys = [columns] if isinstance(columns, str) else list(columns)
    result = churn_by_segment(frame, keys)[[*keys, "customer_count"]].copy()
    result["customer_proportion"] = result["customer_count"] / len(frame) if len(frame) else np.nan
    return result


def numerical_summary(frame: pd.DataFrame, by_churn: bool = False) -> pd.DataFrame:
    """Summarize observed values with sample SD, quartiles and IQR diagnostics.

    IQR outliers use 1.5*IQR fences for investigation only. They are not deleted.
    SeniorCitizen is a binary categorical characteristic, not a continuous measure.
    """
    groups = frame.groupby("Churn", observed=True, dropna=False) if by_churn else [("All", frame)]
    rows = []
    for label, group in groups:
        for feature in NUMERICAL_FEATURES:
            series = group[feature].dropna()
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            rows.append({
                "Churn": label, "feature": feature, "observed_count": int(series.size),
                "missing_count": int(group[feature].isna().sum()),
                "mean": series.mean(), "median": series.median(), "std": series.std(ddof=1),
                "minimum": series.min(), "q1": q1, "q3": q3, "iqr": iqr, "maximum": series.max(),
                "iqr_outlier_count": int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()),
            })
    return pd.DataFrame(rows)


def categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Produce frequency counts and proportions for every categorical feature."""
    return pd.concat([
        customer_segment_counts(frame, feature).rename(columns={feature: "category"}).assign(feature=feature)
        for feature in CATEGORICAL_FEATURES
    ], ignore_index=True)[["feature", "category", "customer_count", "customer_proportion"]]


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Apply Holm's step-down family-wise adjustment in original test order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite numbers between 0 and 1")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.minimum(1, np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values)))))
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted


def association_tests(frame: pd.DataFrame, features: Sequence[str] = ASSOCIATION_FEATURES) -> pd.DataFrame:
    """Explore categorical associations; report effect sizes and assumptions.

    Pearson chi-square uses correction=False. All observed AND expected cell
    counts must be >=5 for the asymptotic flag. Sparse/degenerate tests cannot
    support a significance claim here; no alternative test is chosen post hoc.
    Holm adjustment covers the requested family, including untestable members as
    p=1 for adjustment. These tests are exploratory, marginal and non-causal.
    """
    _check_flags(frame)
    if not features or len(set(features)) != len(features):
        raise ValueError("Supply a nonempty family of distinct features")
    rows = []
    for feature in features:
        if frame[feature].isna().any():
            raise ValueError(f"Association feature {feature} contains missing values")
        table = pd.crosstab(frame[feature], frame["ChurnFlag"]).reindex(columns=[0, 1], fill_value=0)
        row = {"feature": feature, "customers": int(table.to_numpy().sum()), "chi2": np.nan,
               "degrees_freedom": np.nan, "p_value": np.nan, "cramers_v": np.nan,
               "min_observed": int(table.to_numpy().min()) if table.size else 0,
               "min_expected": np.nan, "assumptions_ok": False}
        if table.shape[0] >= 2 and table.sum(axis=0).gt(0).all():
            chi2, p_value, dof, expected = chi2_contingency(table, correction=False)
            row.update(chi2=float(chi2), degrees_freedom=int(dof), p_value=float(p_value),
                       cramers_v=float(np.sqrt(chi2 / (table.to_numpy().sum() * min(table.shape[0] - 1, 1)))),
                       min_expected=float(expected.min()),
                       assumptions_ok=bool(expected.min() >= 5 and row["min_observed"] >= 5))
        rows.append(row)
    result = pd.DataFrame(rows)
    adjusted = holm_adjust(result["p_value"].fillna(1).to_numpy())
    result["p_value_holm"] = np.where(result["p_value"].notna(), adjusted, np.nan)
    result["supports_association"] = result["assumptions_ok"] & result["p_value_holm"].lt(0.05)
    return result


def correlation_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Correlate only meaningful numeric measures; report pairwise coverage."""
    values = frame.loc[:, NUMERICAL_FEATURES]
    indicators = values.notna().astype(int)
    return {"pearson": values.corr(method="pearson"), "spearman": values.corr(method="spearman"),
            "pair_counts": indicators.T.dot(indicators)}
