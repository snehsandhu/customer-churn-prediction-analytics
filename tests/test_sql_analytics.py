"""Known-answer SQL calculations, empty segments, windows and reconciliation."""
from contextlib import closing
import sqlite3

import pytest

from churn_analytics.cleaning import clean_dataset
from churn_analytics.database import connect_database, initialize_database, load_snapshot
from churn_analytics.sql_analytics import (
    QUERY_QUESTIONS, execute_all_queries, execute_query, read_query,
    reconcile_sql_pandas, validate_snapshot_against_frame,
)
from churn_analytics.sql_performance import benchmark_index


@pytest.fixture
def sql_connection(tmp_path, cleaned_customers):
    path = tmp_path / "analytics.db"
    load_snapshot(cleaned_customers, path)
    with closing(connect_database(path)) as conn:
        yield conn


def test_every_business_query_executes(sql_connection):
    results = execute_all_queries(sql_connection)
    assert set(results) == set(QUERY_QUESTIONS)
    assert len(results) == 20
    assert all(table is not None for table in results.values())


def test_even_median_and_known_kpis(sql_connection):
    result = execute_query(sql_connection, "01_customer_kpis").to_dict("records")[0]
    assert (result["total_customers"], result["churn_count"], result["retained_count"]) == (8, 3, 5)
    assert result["churn_pct"] == 37.5
    assert result["retention_pct"] == 62.5
    assert result["mean_monthly_charges"] == pytest.approx(64.24875)
    assert result["median_monthly_charges"] == pytest.approx(64.995)
    assert result["mean_tenure_months"] == 27.875
    assert result["missing_total_charges"] == result["zero_tenure_customers"] == 1


def test_odd_median(tmp_path, cleaned_customers):
    path = tmp_path / "odd.db"
    load_snapshot(cleaned_customers.iloc[:5], path)
    with closing(connect_database(path)) as conn:
        result = execute_query(conn, "01_customer_kpis").iloc[0]
        assert result.median_monthly_charges == 50


def test_segment_denominators_and_lookup_joins(sql_connection):
    contract = execute_query(sql_connection, "02_contract_segments").set_index("Contract")
    assert contract.loc["Month-to-month", "customer_count"] == 3
    assert contract.loc["Month-to-month", "churn_pct"] == pytest.approx(100 / 3)
    assert contract.loc["One year", "churn_pct"] == 50
    joined = sql_connection.execute("""SELECT COUNT(*) FROM customer_snapshot AS s
        INNER JOIN contract_lookup AS c ON s.Contract=c.contract_name
        INNER JOIN payment_method_lookup AS p ON s.PaymentMethod=p.payment_method""").fetchone()[0]
    assert joined == 8
    mix = execute_query(sql_connection, "18_contract_payment_mix")
    assert mix.customer_count.sum() == 8
    assert mix.groupby("Contract").within_contract_customer_pct.sum().round(8).eq(100).all()


def test_left_join_empty_categories_are_zero_count_undefined_rate(sql_connection):
    bands = execute_query(sql_connection, "03_tenure_segments").set_index("TenureGroup")
    empty = bands.loc["37-48 months"]
    assert empty.customer_count == empty.churn_count == 0
    assert empty.isna()["churn_pct"]
    sql_connection.execute("DELETE FROM customer_snapshot WHERE Contract=?", ("Two year",))
    row = execute_query(sql_connection, "02_contract_segments").set_index("Contract").loc["Two year"]
    assert row.customer_count == 0
    assert row.churn_count == 0
    assert row.isna()["churn_pct"]


def test_empty_snapshot_kpis_are_safe(tmp_path):
    with closing(connect_database(tmp_path / "empty.db")) as conn:
        initialize_database(conn)
        result = execute_query(conn, "01_customer_kpis").to_dict("records")[0]
        assert result["total_customers"] == result["churn_count"] == 0
        assert result["churn_pct"] is None
        assert result["median_monthly_charges"] is None
        assert execute_query(conn, "17_monthly_charge_quartiles").empty


def test_sql_pandas_full_roundtrip_and_reconciliation(sql_connection, cleaned_customers):
    validate_snapshot_against_frame(sql_connection, cleaned_customers)
    result = reconcile_sql_pandas(sql_connection, cleaned_customers)
    assert result.passed.all()
    # The fixture has only PaperlessBilling=Yes: one fewer segment than IBM data.
    assert len(result) == 94


def test_reconciliation_exposes_disagreement(sql_connection, cleaned_customers):
    sql_connection.execute("UPDATE customer_snapshot SET Churn=?, ChurnFlag=? WHERE customerID=?", ("Yes", 1, "C0"))
    result = reconcile_sql_pandas(sql_connection, cleaned_customers)
    assert not result.passed.all()
    assert not result.loc[result.metric.eq("churn_count"), "passed"].iloc[0]


def test_values_are_parameterized_and_query_names_allowlisted(sql_connection):
    attack = "Two year' OR 1=1; DROP TABLE customer_snapshot; --"
    result = execute_query(sql_connection, "20_filtered_contract_tenure", {"contract": attack})
    assert result.iloc[0].customer_count == 0
    assert sql_connection.execute("SELECT COUNT(*) FROM customer_snapshot").fetchone()[0] == 8
    with pytest.raises(ValueError, match="Unknown SQL"):
        read_query("../schema")
    with pytest.raises(ValueError, match="Unexpected"):
        execute_query(sql_connection, "20_filtered_contract_tenure", {"column": "Contract"})
    with pytest.raises(ValueError, match="nonnegative integer"):
        execute_query(sql_connection, "11_service_combinations", {"min_customers": "0 OR 1=1"})


def test_tied_charge_rankings_are_distinct(tmp_path, raw_customers):
    raw_customers.loc[[0, 1, 6], "MonthlyCharges"] = [100, 100, 50]
    path = tmp_path / "ties.db"
    load_snapshot(clean_dataset(raw_customers), path)
    with closing(connect_database(path)) as conn:
        rows = execute_query(conn, "16_customer_charge_positions", {"top_n": 3})
    month = rows.loc[rows.Contract.eq("Month-to-month")]
    assert month.customerID.tolist() == ["C0", "C1", "C6"]
    assert month.selection_position.tolist() == [1, 2, 3]
    assert month.charge_rank.tolist() == [1, 1, 3]
    assert month.distinct_price_rank.tolist() == [1, 1, 2]
    assert month.contract_mean_monthly_charges.tolist() == pytest.approx([250 / 3] * 3)


def test_lag_lead_and_cumulative_counts(sql_connection):
    bands = execute_query(sql_connection, "13_adjacent_tenure_bands")
    assert bands.iloc[0].customer_count == 3
    assert bands.iloc[0].cumulative_customers == 3
    assert bands.iloc[-1].cumulative_customers == 8
    assert bands.iloc[0].next_band_churn_pct == 50
    assert bands.iloc[1].difference_from_previous_pp == pytest.approx(50 - 200 / 3)


def test_quartiles_and_threshold_filters(sql_connection):
    quartiles = execute_query(sql_connection, "17_monthly_charge_quartiles")
    assert quartiles.customer_count.tolist() == [2, 2, 2, 2]
    assert quartiles.churn_count.sum() == 3
    assert execute_query(sql_connection, "14_substantial_above_average_segments", {"min_customers": 100}).empty


def test_measured_benchmark_preserves_live_indexes_and_results(sql_connection):
    before = [tuple(r) for r in sql_connection.execute("PRAGMA index_list(customer_snapshot)")]
    result = benchmark_index(sql_connection, repeats=3)
    assert result["results_match"]
    assert result["before_index"]["repeats"] == result["after_index"]["repeats"] == 3
    assert result["before_index"]["median_ms"] > 0
    assert result["after_index"]["median_ms"] > 0
    assert result["before_index"]["plan"] and result["after_index"]["plan"]
    # Do not assert a speedup or an exact planner string: both are environment-dependent.
    assert [tuple(r) for r in sql_connection.execute("PRAGMA index_list(customer_snapshot)")] == before
