"""Offline CSV-to-SQL/report integration and command-line parameter handling."""
import hashlib
import json

import pytest

from churn_analytics.cleaning import save_cleaned_dataset
from churn_analytics.database import connect_database
from churn_analytics.phase3 import main, run_phase3


def test_full_phase3_workflow_preserves_input_and_is_repeatable(tmp_path, cleaned_customers):
    processed = save_cleaned_dataset(cleaned_customers, tmp_path / "processed.csv")
    fingerprint = hashlib.sha256(processed.read_bytes()).hexdigest()
    database = tmp_path / "analytics.db"
    reports = tmp_path / "reports"
    first = run_phase3(processed, database, reports, benchmark_repeats=3)
    assert first["queries_executed"] == 20
    assert first["reconciliation_checks"] == first["reconciliation_passed"] == 94
    assert first["sql_kpis"]["churn_pct"] == 37.5
    assert first["full_snapshot_roundtrip"] == "passed"
    assert first["benchmark_results_match"]
    assert len(list((reports / "phase3/query_results").glob("*.csv"))) == 20
    business = (reports / "PHASE3_SQL_BUSINESS_INSIGHTS.md").read_text()
    assert business.count("```sql") == 20
    assert "37.50%" in business
    performance = json.loads((reports / "phase3/query_performance.json").read_text())
    assert performance["before_index"]["repeats"] == 3
    second = run_phase3(processed, database, reports, benchmark_repeats=3)
    assert first == second  # Timings are deliberately excluded from this logical manifest.
    assert hashlib.sha256(processed.read_bytes()).hexdigest() == fingerprint


def test_missing_phase2_artifact_does_not_reclean_or_create_database(tmp_path):
    db = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError, match="Phase 2 CSV missing"):
        run_phase3(tmp_path / "absent.csv", db, tmp_path / "reports")
    assert not db.exists()


def test_query_cli_uses_json_parameters_readonly(tmp_path, cleaned_customers, monkeypatch, capsys):
    processed = save_cleaned_dataset(cleaned_customers, tmp_path / "processed.csv")
    database = tmp_path / "analytics.db"
    run_phase3(processed, database, tmp_path / "reports", benchmark_repeats=1)
    monkeypatch.setattr("sys.argv", ["phase3", "--database-path", str(database),
        "--query", "20_filtered_contract_tenure", "--params", json.dumps({"contract": "Month-to-month", "min_tenure": 0, "max_tenure": 12})])
    main()
    output = capsys.readouterr().out
    assert "customer_count" in output
    assert "50.0" in output  # The fixture's two short-tenure monthly customers: 1/2 churned.
    monkeypatch.setattr("sys.argv", ["phase3", "--database-path", str(database),
        "--query", "20_filtered_contract_tenure", "--params", "[1,2]"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "JSON object" in capsys.readouterr().err
