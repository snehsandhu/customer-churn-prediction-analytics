"""Read-only descriptive analytics; historical rows are never sent to a predictor."""
from pathlib import Path

import pandas as pd

from ..analytics import churn_by_segment, customer_kpis
from ..cleaning import load_cleaned_dataset
from ..data import PROJECT_ROOT
from ..phase5_data import sha256
from .artifacts import DashboardError, read_local_json

FILTER_COLUMNS = ("Contract", "InternetService", "PaymentMethod", "TenureGroup", "MonthlyChargeGroup",
                  "SeniorCitizen", "PaperlessBilling", "TechSupport", "OnlineSecurity")
SEGMENTS = ("Contract", "TenureGroup", "MonthlyChargeGroup", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity")


def load_analytics(root: str | Path = PROJECT_ROOT) -> pd.DataFrame:
    """Validate the existing Phase 2 snapshot and immediately discard identifiers."""
    root = Path(root)
    try:
        path = root/"data/processed/telco_churn_cleaned.csv"
        expected = read_local_json(root, "reports/phase4/split_manifest.json")["dataset_sha256"]
        if sha256(path) != expected:
            raise ValueError("Source fingerprint")
        return load_cleaned_dataset(path).drop(columns="customerID")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DashboardError("The analytical snapshot is missing or invalid. Restore the supplied Phase 2 data.") from exc


def filter_customers(frame: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Intersect explicit category selections; an empty selection means no rows."""
    result = frame.copy()
    for column, values in filters.items():
        if column not in FILTER_COLUMNS or not isinstance(values, (list, tuple)):
            raise DashboardError("Use the available categorical filters.")
        if any(value not in frame[column].dropna().unique() for value in values):
            raise DashboardError("A filter category is unsupported.")
        result = result.loc[result[column].isin(values)]
    return result


def segment_table(frame: pd.DataFrame, column: str | list[str]) -> pd.DataFrame:
    """Reuse Phase 2 segment counts/rates, including undefined empty groups."""
    columns = [column] if isinstance(column, str) else column
    if not columns or any(c not in FILTER_COLUMNS for c in columns):
        raise DashboardError("Choose a supported segment.")
    return churn_by_segment(frame, column)


def insights(frame: pd.DataFrame) -> list[str]:
    """Describe exact sample associations and frame follow-ups as hypotheses."""
    statements = []
    for column, label in [("Contract", "contract"), ("TenureGroup", "tenure band"),
                           ("InternetService", "internet service"), ("PaymentMethod", "payment method")]:
        table = segment_table(frame, column).query("customer_count >= 30").sort_values("churn_rate")
        if len(table) < 2:
            continue
        low, high = table.iloc[0], table.iloc[-1]
        statements.append(f"By {label}, {high[column]} shows {high.churn_pct:.1f}% observed churn (n={int(high.customer_count):,}), "
                          f"compared with {low.churn_pct:.1f}% for {low[column]} (n={int(low.customer_count):,}).")
    return statements


def snapshot_kpis(frame: pd.DataFrame) -> dict:
    """Delegate definitions and null/empty handling to established analytics."""
    return customer_kpis(frame)


def numerical_bins(frame: pd.DataFrame, column: str, *, bins: int = 12) -> pd.DataFrame:
    """Aggregate observed values in numeric bin order before formatting labels.

    Converting intervals to strings before grouping sorts '(100, ...]' ahead of
    '(20, ...]'. Retain ordered intervals until grouping, then return browser-safe
    aggregate labels and counts; missing observations never become zero values.
    """
    if column not in ("tenure", "MonthlyCharges", "TotalCharges"):
        raise DashboardError("Choose a supported numerical measure.")
    observed = frame.loc[frame[column].notna(), [column, "ChurnFlag"]].copy()
    if observed.empty:
        return pd.DataFrame(columns=["Band", "Outcome", "Customers"])
    observed["Band"] = pd.cut(observed[column], bins=bins, duplicates="drop")
    observed["Outcome"] = observed.ChurnFlag.map({0: "No churn", 1: "Churn"})
    result = observed.groupby(["Band", "Outcome"], observed=True, sort=True).size().rename("Customers").reset_index()
    result["Band"] = result.Band.astype(str)
    return result
