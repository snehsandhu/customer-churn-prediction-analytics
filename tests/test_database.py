"""Temporary-database ingestion, constraints, NULL semantics and rollback tests."""
from contextlib import closing
import hashlib
import sqlite3

import pandas as pd
import pytest

from churn_analytics.database import (
    TABLES, connect_database, database_checks, initialize_database, load_snapshot,
    validate_database_schema,
)


@pytest.fixture
def database_path(tmp_path, cleaned_customers):
    """Create a local database with the eight-row synthetic fixture."""
    path = tmp_path / "customers.db"
    load_snapshot(cleaned_customers, path)
    return path


def test_creation_schema_types_and_relationships(database_path):
    with closing(connect_database(database_path)) as conn:
        validate_database_schema(conn)
        columns = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(customer_snapshot)")}
        assert len(columns) == 25
        assert columns["customerID"] == "TEXT"
        assert columns["TotalCharges"] == "REAL"
        assert columns["ChurnFlag"] == "INTEGER"
        assert len(conn.execute("PRAGMA foreign_key_list(customer_snapshot)").fetchall()) == 4
        strict = {r[1]: r[5] for r in conn.execute("PRAGMA table_list")}
        assert all(strict[name] == 1 for name in TABLES)
        assert database_checks(conn)["integrity_check"] == "ok"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_load_repeatability_and_no_frame_mutation(database_path, cleaned_customers):
    before = cleaned_customers.copy(deep=True)
    first = load_snapshot(cleaned_customers, database_path)
    second = load_snapshot(cleaned_customers, database_path)
    assert first == second
    assert second["inserted_records"] == second["unique_customer_ids"] == 8
    assert second["table_rows"] == dict(zip(TABLES, [3, 4, 6, 3, 8]))
    pd.testing.assert_frame_equal(cleaned_customers, before)


def test_sql_null_and_primitive_storage_types(database_path):
    with closing(connect_database(database_path)) as conn:
        row = conn.execute("SELECT TotalCharges, TotalChargesMissing, typeof(TotalChargesMissing), typeof(MonthlyCharges) FROM customer_snapshot WHERE customerID=?", ("C0",)).fetchone()
        assert tuple(row) == (None, 1, "integer", "real")
        assert conn.execute("SELECT COUNT(TotalCharges) FROM customer_snapshot").fetchone()[0] == 7
        assert conn.execute("SELECT AVG(TotalCharges) FROM customer_snapshot WHERE tenure=0").fetchone()[0] is None


@pytest.mark.parametrize("assignment,parameters", [
    ("customerID=?", (None,)), ("customerID=?", ("",)), ("customerID=?", (" \t ",)),
    ("customerID=?", ("C1",)), ("Contract=?", ("Unknown",)),
    ("Churn=?", ("Maybe",)), ("ChurnFlag=?", (1,)),
    ("TotalChargesMissing=?", (0,)), ("MonthlyCharges=?", (-1,)),
    ("MonthlyCharges=?", (float("inf"),)), ("tenure=?", (1.5,)),
    ("TechSupport=?", ("No internet service",)),
])
def test_constraints_reject_invalid_direct_writes(database_path, assignment, parameters):
    # Assignment fragments are fixed test literals, never user-provided identifiers.
    with closing(connect_database(database_path)) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"UPDATE customer_snapshot SET {assignment} WHERE customerID='C0'", parameters)
        assert conn.execute("SELECT COUNT(*) FROM customer_snapshot").fetchone()[0] == 8


def test_invalid_frame_rejected_before_database_creation(tmp_path, cleaned_customers):
    cleaned_customers.loc[0, "ChurnFlag"] = 1
    path = tmp_path / "invalid.db"
    with pytest.raises(ValueError, match="Invalid Phase 2"):
        load_snapshot(cleaned_customers, path)
    assert not path.exists()


def test_unowned_database_is_preserved(tmp_path, cleaned_customers):
    path = tmp_path / "other.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE important(value TEXT)")
        conn.execute("INSERT INTO important VALUES ('preserve')")
        conn.commit()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not a managed"):
        load_snapshot(cleaned_customers, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_insertion_failure_rolls_back_previous_snapshot(database_path, cleaned_customers):
    with closing(connect_database(database_path)) as conn:
        conn.execute("""CREATE TRIGGER reject_import BEFORE INSERT ON customer_snapshot
                        WHEN NEW.customerID='C3' BEGIN SELECT RAISE(ABORT, 'intentional test failure'); END""")
    with pytest.raises(RuntimeError, match="intentional test failure"):
        load_snapshot(cleaned_customers.iloc[:-1], database_path)
    with closing(connect_database(database_path)) as conn:
        assert database_checks(conn)["total_customers"] == 8
        assert conn.execute("SELECT customerID FROM customer_snapshot WHERE customerID='C7'").fetchone()[0] == "C7"


def test_schema_drift_is_rejected_before_refresh(database_path, cleaned_customers):
    with closing(connect_database(database_path)) as conn:
        conn.execute("ALTER TABLE customer_snapshot ADD COLUMN unintended TEXT")
    with pytest.raises(ValueError, match="schema differs"):
        load_snapshot(cleaned_customers, database_path)
    with closing(connect_database(database_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM customer_snapshot").fetchone()[0] == 8


def test_readonly_connection_and_missing_database(database_path, tmp_path):
    with closing(connect_database(database_path, readonly=True)) as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM customer_snapshot")
    with pytest.raises(FileNotFoundError, match="Database missing"):
        connect_database(tmp_path / "missing.db", readonly=True)


def test_sql_looking_customer_id_is_stored_as_data(tmp_path, cleaned_customers):
    identifier = "C0'); DROP TABLE customer_snapshot; --"
    cleaned_customers.loc[0, "customerID"] = identifier
    path = tmp_path / "safe.db"
    load_snapshot(cleaned_customers, path)
    with closing(connect_database(path)) as conn:
        assert conn.execute("SELECT customerID FROM customer_snapshot WHERE customerID=?", (identifier,)).fetchone()[0] == identifier
        assert conn.execute("SELECT COUNT(*) FROM customer_snapshot").fetchone()[0] == 8
