"""Named, parameterized SQL execution and independent Pandas reconciliation."""

from __future__ import annotations

import math
import sqlite3

import pandas as pd

from .analytics import churn_by_segment, customer_kpis
from .database import SQL_DIR

# Each checked-in SQL file answers a distinct question. Only these names are executable.
QUERY_QUESTIONS = {
    "01_customer_kpis": "What are the snapshot's customer counts, churn percentages and charge/tenure summaries?",
    "02_contract_segments": "How does churn vary by contract type?",
    "03_tenure_segments": "How does churn vary across fixed tenure bands?",
    "04_monthly_charge_segments": "How does churn differ across the Phase 2 charge groups?",
    "05_internet_segments": "How does churn vary by internet service?",
    "06_payment_method_segments": "Which payment-method groups show different churn proportions?",
    "07_technical_support_segments": "How is technical-support subscription associated with churn?",
    "08_online_security_segments": "How is online-security subscription associated with churn?",
    "09_senior_citizen_segments": "How do observed outcomes differ by SeniorCitizen status?",
    "10_paperless_billing_segments": "How does churn differ by paperless-billing status?",
    "11_service_combinations": "Which adequately sized internet/support/security combinations show higher sample churn?",
    "12_contract_churn_contribution": "Which contracts lead by churn rate, churn count and share of all churn?",
    "13_adjacent_tenure_bands": "How do adjacent tenure bands compare, and how many customers accumulate across ordered bands?",
    "14_substantial_above_average_segments": "Which substantial contract/internet groups exceed overall sample churn?",
    "15_charges_by_churn": "How do mean and median charges compare between churned and non-churned customers?",
    "16_customer_charge_positions": "Which customers have the highest charges within each contract, including tied-price rankings?",
    "17_monthly_charge_quartiles": "How do customer-count charge quartiles differ in size, charges and churn?",
    "18_contract_payment_mix": "How does payment-method mix and churn differ within each contract type?",
    "19_missing_charge_coverage": "Where are cumulative charges missing, and what is the observed-value coverage?",
    "20_filtered_contract_tenure": "What are the counts and charges for a selected contract/tenure interval?",
}
DEFAULT_PARAMETERS = {
    "11_service_combinations": {"min_customers": 100},
    "14_substantial_above_average_segments": {"min_customers": 100},
    "16_customer_charge_positions": {"top_n": 3},
    "20_filtered_contract_tenure": {"contract": "Two year", "min_tenure": 0, "max_tenure": 12},
}
SEGMENT_QUERIES = {
    "02_contract_segments": "Contract", "03_tenure_segments": "TenureGroup",
    "04_monthly_charge_segments": "MonthlyChargeGroup", "05_internet_segments": "InternetService",
    "06_payment_method_segments": "PaymentMethod", "07_technical_support_segments": "TechSupport",
    "08_online_security_segments": "OnlineSecurity", "09_senior_citizen_segments": "SeniorCitizen",
    "10_paperless_billing_segments": "PaperlessBilling",
}
FLOAT_TOLERANCE = 1e-10


def read_query(name: str) -> str:
    """Load only a known, checked-in query; never interpolate user paths or SQL."""
    if name not in QUERY_QUESTIONS:
        raise ValueError(f"Unknown SQL query {name!r}; choose a name from QUERY_QUESTIONS")
    return (SQL_DIR / "queries" / f"{name}.sql").read_text(encoding="utf-8")


def query_parameters(name: str, overrides: dict | None = None) -> dict:
    """Validate parameter shapes; values are passed separately to SQLite."""
    if name not in QUERY_QUESTIONS:
        raise ValueError(f"Unknown SQL query {name!r}")
    params = dict(DEFAULT_PARAMETERS.get(name, {}))
    if overrides:
        if set(overrides) - set(params):
            raise ValueError("Unexpected SQL parameters")
        params.update(overrides)
    for key in ("min_customers", "top_n", "min_tenure", "max_tenure"):
        if key in params and (type(params[key]) is not int or params[key] < 0):
            raise ValueError(f"{key} must be a nonnegative integer")
    if "top_n" in params and not 1 <= params["top_n"] <= 1000:
        raise ValueError("top_n must be between 1 and 1000")
    if "contract" in params and not isinstance(params["contract"], str):
        raise ValueError("contract must be text")
    if "min_tenure" in params and params["min_tenure"] > params["max_tenure"]:
        raise ValueError("min_tenure must not exceed max_tenure")
    return params


def execute_query(conn: sqlite3.Connection, name: str, parameters: dict | None = None) -> pd.DataFrame:
    """Execute one read-only analytical SELECT with safely bound data values."""
    return pd.read_sql_query(read_query(name), conn, params=query_parameters(name, parameters))


def execute_all_queries(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Run every business query using explicit documented default parameters."""
    return {name: execute_query(conn, name) for name in QUERY_QUESTIONS}


def validate_snapshot_against_frame(conn: sqlite3.Connection, frame: pd.DataFrame) -> None:
    """Compare every stored customer value after restoring Pandas type metadata."""
    actual = pd.read_sql_query("SELECT * FROM customer_snapshot ORDER BY customerID", conn)
    expected = frame.sort_values("customerID").reset_index(drop=True)
    actual = actual.astype(expected.dtypes.to_dict())
    pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-12, atol=1e-12)


def reconcile_sql_pandas(conn: sqlite3.Connection, frame: pd.DataFrame) -> pd.DataFrame:
    """Compare unrounded SQL results with existing Phase 2 functions.

    Counts must match exactly. Floating metrics use abs_tol=rel_tol=1e-10; matched
    undefined values compare equal. Outer joins expose missing/extra segment keys.
    A mismatch is reported, never hidden by rounding or changing expected values.
    """
    rows = []
    def compare(metric, sql_value, pandas_value, *, exact=False, present=True):
        sql_missing, py_missing = pd.isna(sql_value), pd.isna(pandas_value)
        if sql_missing or py_missing:
            match = bool(sql_missing and py_missing)
            difference = 0.0 if match else float("nan")
        else:
            match = bool(sql_value == pandas_value) if exact else math.isclose(
                float(sql_value), float(pandas_value), rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE,
            )
            difference = abs(float(sql_value) - float(pandas_value))
        rows.append({"metric": metric, "sql_value": sql_value, "pandas_value": pandas_value,
                     "absolute_difference": difference, "comparison": "exact" if exact else "abs/rel 1e-10",
                     "passed": bool(match and present)})
    sql_kpis = execute_query(conn, "01_customer_kpis").to_dict("records")[0]
    python_kpis = customer_kpis(frame)
    mapping = {
        "total_customers": (python_kpis["total_customers"], True),
        "churn_count": (python_kpis["churn_count"], True),
        "retained_count": (python_kpis["retained_count"], True),
        "churn_pct": (100 * python_kpis["churn_proportion"], False),
        "retention_pct": (100 * python_kpis["retention_proportion"], False),
        "mean_monthly_charges": (python_kpis["mean_monthly_charges"], False),
        "median_monthly_charges": (python_kpis["median_monthly_charges"], False),
        "mean_tenure_months": (python_kpis["mean_tenure_months"], False),
        "missing_total_charges": (int(frame.TotalCharges.isna().sum()), True),
        "zero_tenure_customers": (int(frame.tenure.eq(0).sum()), True),
    }
    for metric, (expected, exact) in mapping.items():
        compare(metric, sql_kpis[metric], expected, exact=exact)
    for name, feature in SEGMENT_QUERIES.items():
        sql = execute_query(conn, name)
        python = churn_by_segment(frame, feature)
        merged = sql.merge(python[[feature, "customer_count", "churn_count", "churn_pct"]],
                           on=feature, how="outer", suffixes=("_sql", "_pandas"), indicator=True)
        for record in merged.to_dict("records"):
            for metric in ("customer_count", "churn_count", "churn_pct"):
                compare(f"{feature}={record[feature]}:{metric}", record[f"{metric}_sql"], record[f"{metric}_pandas"],
                        exact=metric != "churn_pct", present=record["_merge"] == "both")
    return pd.DataFrame(rows)
