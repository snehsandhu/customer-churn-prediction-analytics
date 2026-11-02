"""Offline integration of raw CSV -> validated processed CSV -> calculated reports."""
import hashlib
import json

import pytest

from churn_analytics.cleaning import load_cleaned_dataset
from churn_analytics.phase2 import run_phase2


def test_workflow_reports_from_saved_data_and_preserves_raw(tmp_path, raw_customers, monkeypatch):
    # Figure content is checked in actual-dataset verification; keep unit tests fast.
    monkeypatch.setattr("churn_analytics.plots.create_figures", lambda frame, directory: [])
    raw_path = tmp_path / "raw.csv"
    raw_customers.to_csv(raw_path, index=False)
    fingerprint = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    output = tmp_path / "processed.csv"
    reports = tmp_path / "reports"
    manifest = run_phase2(raw_path, output, reports)
    assert manifest["raw_sha256"] == fingerprint
    assert manifest["cleaned_rows"] == 8
    assert len(load_cleaned_dataset(output)) == 8
    kpis = json.loads((reports / "kpis.json").read_text())
    assert kpis["churn_count"] == 3
    assert kpis["churn_proportion"] == 3 / 8
    assert (reports / "PHASE2_BUSINESS_INSIGHTS.md").is_file()
    assert (reports / "data_quality_report.md").is_file()
    assert (reports / "tables/segment_contract_tenure.csv").is_file()
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == fingerprint
    original_output = output.read_bytes()
    second = run_phase2(raw_path, output, reports)
    assert output.read_bytes() == original_output
    assert second == manifest


def test_workflow_refuses_to_overwrite_source(tmp_path, raw_customers):
    raw = tmp_path / "raw.csv"
    raw_customers.to_csv(raw, index=False)
    with pytest.raises(ValueError, match="must differ"):
        run_phase2(raw, raw, tmp_path / "reports")
