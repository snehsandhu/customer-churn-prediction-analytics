"""Aggregate-only public exports and explicit prediction-output exports."""
import json
import pandas as pd
from .artifacts import DashboardError, saved_performance
from .data import snapshot_kpis

DENIED_COLUMNS = {"customerid", "customer_id", "churn", "churnflag", "y_true", "churn_probability", "prediction_at_0_50"}


def aggregate_csv(frame: pd.DataFrame) -> bytes:
    """Export trusted aggregate tables, rejecting identifying/outcome-row schemas.

    Escape spreadsheet formula-leading strings. Numerical negatives remain numeric.
    Callers provide only established segment/approved-query tables, never raw rows.
    """
    if not frame.columns.is_unique or any(str(c).lower() in DENIED_COLUMNS for c in frame.columns):
        raise DashboardError("Only approved aggregate tables can be exported.")
    safe = frame.copy()
    for column in safe:
        safe[column] = safe[column].astype(object).map(
            lambda x: "'"+x if isinstance(x,str) and x.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else x)
    return safe.to_csv(index=False).encode("utf-8")


def public_summary(frame: pd.DataFrame, performance: dict) -> bytes:
    """Return only KPI/aggregate evaluation fields already scrubbed by services."""
    payload = {"dataset": "Public IBM Telco Customer Churn example snapshot", "snapshot_kpis": snapshot_kpis(frame),
        "saved_evaluation": performance,
        "limitations": ["Snapshot associations do not establish causation or ROI.",
            "Scores are predictive estimates, not guarantees or personal causal explanations.",
            "The test is consumed; this application does not reevaluate it."]}
    return json.dumps(payload, indent=2, allow_nan=False).encode("utf-8")
