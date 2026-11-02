"""Validated, transactional SQLite ingestion of the existing Phase 2 artifact."""

from __future__ import annotations

from contextlib import closing
from functools import lru_cache
from pathlib import Path
import sqlite3

import pandas as pd

from .cleaning import CLEAN_COLUMNS, CHARGE_LABELS, TENURE_LABELS, validate_cleaned_dataset
from .data import PROJECT_ROOT

DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data/database/customer_churn.db"
SQL_DIR = PROJECT_ROOT / "sql"
APPLICATION_ID = int.from_bytes(b"CHRN", "big")
SCHEMA_VERSION = 1
TABLES = ("contract_lookup", "payment_method_lookup", "tenure_group_lookup", "monthly_charge_group_lookup", "customer_snapshot")
CONTRACT_ORDER = {"Month-to-month": 1, "One year": 2, "Two year": 3}


def connect_database(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced; callers must close it.

    STRICT tables require SQLite >=3.37. Boolean flags are stored as constrained
    INTEGER 0/1; category labels as TEXT; ordered category metadata lives in lookups.
    """
    if sqlite3.sqlite_version_info < (3, 37, 0):
        raise RuntimeError(f"SQLite >=3.37 is required; found {sqlite3.sqlite_version}")
    path = Path(path).resolve()
    if readonly:
        if not path.is_file():
            raise FileNotFoundError(f"Database missing: {path}. Run python -m churn_analytics.phase3 first.")
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, isolation_level=None)
    else:
        if path.suffix.lower() not in {".db", ".sqlite", ".sqlite3"}:
            raise ValueError("Database path must have a .db, .sqlite or .sqlite3 extension")
        if any(path.is_relative_to(PROJECT_ROOT / "data" / name) for name in ("raw", "processed")):
            raise ValueError("Database must not be written into data/raw/ or data/processed/")
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _schema_signature(conn: sqlite3.Connection) -> dict:
    """Compare actual SQLite-normalized table DDL, including CHECK/FK constraints."""
    placeholders = ",".join("?" for _ in TABLES)
    return {row[0]: row[1] for row in conn.execute(
        f"SELECT name, sql FROM sqlite_schema WHERE type='table' AND name IN ({placeholders})", TABLES,
    )}


@lru_cache(maxsize=1)
def _expected_signature() -> dict:
    """Let SQLite normalize the checked-in schema rather than parsing SQL text."""
    with closing(sqlite3.connect(":memory:")) as template:
        template.executescript((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
        return _schema_signature(template)


def validate_database_schema(conn: sqlite3.Connection) -> None:
    """Reject schema drift before data replacement, including removed constraints."""
    if _schema_signature(conn) != _expected_signature():
        raise ValueError("Database schema differs from sql/schema.sql; refusing to replace snapshot data")
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise ValueError("Foreign key enforcement must be enabled on this connection")


def initialize_database(conn: sqlite3.Connection) -> None:
    """Initialize an empty database or verify ownership/schema of an existing one."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    if tables:
        if conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
            raise ValueError("Existing database is not a managed churn database; no data was changed")
        if tables != set(TABLES) or conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise ValueError("Unexpected database tables/version; no data was changed")
        validate_database_schema(conn)
    else:
        conn.executescript((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
        conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        validate_database_schema(conn)
    conn.executescript((SQL_DIR / "indexes.sql").read_text(encoding="utf-8"))


def database_checks(conn: sqlite3.Connection) -> dict:
    """Verify integrity, relationships, snapshot identity and lookup row counts."""
    validate_database_schema(conn)
    integrity = [r[0] for r in conn.execute("PRAGMA integrity_check")]
    foreign_keys = [tuple(r) for r in conn.execute("PRAGMA foreign_key_check")]
    row = conn.execute("SELECT COUNT(*) AS customers, COUNT(DISTINCT customerID) AS unique_ids, COUNT(*) - COUNT(TotalCharges) AS null_total_charges FROM customer_snapshot").fetchone()
    if integrity != ["ok"] or foreign_keys or row["customers"] != row["unique_ids"]:
        raise ValueError(f"Database integrity check failed: {integrity}; foreign keys={foreign_keys}")
    return {"integrity_check": integrity[0], "foreign_key_violations": len(foreign_keys),
            "total_customers": row["customers"], "unique_customer_ids": row["unique_ids"],
            "null_total_charges": row["null_total_charges"],
            "table_rows": {name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] for name in TABLES}}


def _sql_value(value):
    """Bind Python primitives, mapping analytical missingness to SQL NULL."""
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def load_snapshot(frame: pd.DataFrame, path: str | Path = DEFAULT_DATABASE_PATH) -> dict:
    """Replace only a validated, owned snapshot in one transaction.

    No cleaning is performed here. The caller supplies the Phase 2 dataset. Input
    errors are detected before opening the database; insertion failures roll back
    the prior snapshot, rather than leaving a partially refreshed database.
    """
    report = validate_cleaned_dataset(frame)
    if report["errors"]:
        raise ValueError("Invalid Phase 2 input: " + "; ".join(report["errors"]))
    rows = [tuple(_sql_value(value) for value in row) for row in frame.loc[:, CLEAN_COLUMNS].itertuples(index=False, name=None)]
    columns = ", ".join(f'"{name}"' for name in CLEAN_COLUMNS)
    placeholders = ", ".join("?" for _ in CLEAN_COLUMNS)
    try:
        with closing(connect_database(path)) as conn:
            initialize_database(conn)
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM customer_snapshot")
                for name in TABLES[:-1]:
                    conn.execute(f"DELETE FROM {name}")
                conn.executemany("INSERT INTO contract_lookup VALUES (?, ?)",
                                 [(name, CONTRACT_ORDER[name]) for name in sorted(frame.Contract.unique())])
                conn.executemany("INSERT INTO payment_method_lookup VALUES (?)",
                                 [(name,) for name in sorted(frame.PaymentMethod.unique())])
                conn.executemany("INSERT INTO tenure_group_lookup VALUES (?, ?)",
                                 [(name, i) for i, name in enumerate(TENURE_LABELS, 1)])
                conn.executemany("INSERT INTO monthly_charge_group_lookup VALUES (?, ?)",
                                 [(name, i) for i, name in enumerate(CHARGE_LABELS, 1)])
                conn.executemany(f"INSERT INTO customer_snapshot ({columns}) VALUES ({placeholders})", rows)
                checks = database_checks(conn)
                if checks["total_customers"] != len(frame) or checks["null_total_charges"] != int(frame.TotalCharges.isna().sum()):
                    raise ValueError("Inserted row/missingness counts do not match the Phase 2 artifact")
                conn.execute("ANALYZE")
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(f"SQLite ingestion failed for {Path(path).resolve()}: {exc}") from exc
    return {"inserted_records": len(rows), "database_path": str(Path(path).resolve()), **checks}
