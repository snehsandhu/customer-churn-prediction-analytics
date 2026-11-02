"""Curated aggregate SQL only, with read-only connections and an authorizer."""
from contextlib import closing
from pathlib import Path
import sqlite3

from ..data import PROJECT_ROOT
from ..database import connect_database
from ..sql_analytics import QUERY_QUESTIONS, execute_query, read_query
from .artifacts import DashboardError

SQL_QUESTIONS = {name: QUERY_QUESTIONS[name] for name in (
    "01_customer_kpis", "02_contract_segments", "03_tenure_segments", "06_payment_method_segments",
    "11_service_combinations", "12_contract_churn_contribution", "13_adjacent_tenure_bands",
    "14_substantial_above_average_segments", "17_monthly_charge_quartiles")}
SQL_EXPLANATIONS = {
    "01_customer_kpis": "Counts and averages summarize the whole public snapshot; they do not measure trends or revenue loss.",
    "02_contract_segments": "Divide churned customers by all customers within each contract category.",
    "03_tenure_segments": "Compare customers across fixed tenure bands, with each band's count as its denominator.",
    "06_payment_method_segments": "Compare observed churn within payment-method categories; payment choice is not a proven cause.",
    "11_service_combinations": "Group internet, support and security subscriptions; display only groups meeting the minimum sample size.",
    "12_contract_churn_contribution": "Ranks distinguish within-contract churn rate from each contract's share of all churned customers.",
    "13_adjacent_tenure_bands": "Window functions compare neighboring tenure bands and cumulative counts; these are not calendar trends.",
    "14_substantial_above_average_segments": "Find substantial contract/internet groups above the overall sample rate; investigate before acting.",
    "17_monthly_charge_quartiles": "Split customers into four count-based charge groups. Tied prices may fall in different quartiles.",
}


def open_readonly(root: str | Path = PROJECT_ROOT):
    """Open a fresh connection, forbid writes/ATTACH/DDL even through SQL changes."""
    try:
        conn = connect_database(Path(root)/"data/database/customer_churn.db", readonly=True)
        conn.execute("PRAGMA query_only = ON")
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() == "load_extension":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        conn.set_authorizer(authorize)
        return conn
    except (OSError, sqlite3.Error) as exc:
        raise DashboardError("The read-only analytics database is unavailable. Restore the supplied Phase 3 database.") from exc


def query_business(name: str, *, min_customers: int = 100, root: str | Path = PROJECT_ROOT):
    """Execute a registered aggregate SELECT with bound validated parameters."""
    if name not in SQL_QUESTIONS:
        raise DashboardError("Choose an approved aggregate business question.")
    if type(min_customers) is not int or not 1 <= min_customers <= 1000:
        raise DashboardError("Minimum group size must be a whole number from 1 to 1,000.")
    params = {"min_customers": min_customers} if name in {"11_service_combinations", "14_substantial_above_average_segments"} else None
    try:
        with closing(open_readonly(root)) as conn:
            result = execute_query(conn, name, params)
        return result
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise DashboardError("This approved query is unavailable. Check the supplied SQL and database files.") from exc


def query_text(name: str) -> str:
    """Allowlist before obtaining the original checked-in SQL definition."""
    if name not in SQL_QUESTIONS:
        raise DashboardError("Choose an approved aggregate business question.")
    try:
        return read_query(name)
    except OSError as exc:
        raise DashboardError("The saved query definition is unavailable.") from exc
