"""Phase 2 cleaning contracts, malformed input and immutable raw-file checks."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from churn_analytics.cleaning import (
    CLEAN_COLUMNS, DEFAULT_RAW_PATH, build_quality_report, clean_dataset,
    load_cleaned_dataset, save_cleaned_dataset, validate_cleaned_dataset,
)
from churn_analytics.data import load_dataset


def test_preserves_rows_schema_and_input(raw_customers):
    original = raw_customers.copy(deep=True)
    result = clean_dataset(raw_customers)
    assert tuple(result.columns) == CLEAN_COLUMNS
    assert len(result) == 8
    assert result.customerID.tolist() == original.customerID.tolist()
    assert result.ChurnFlag.tolist() == [0, 1, 1, 0, 1, 0, 0, 0]
    assert result.TotalChargesMissing.tolist() == [True] + [False] * 7
    assert np.isnan(result.loc[0, "TotalCharges"])
    assert result.loc[1, "TotalCharges"] == 294
    assert not validate_cleaned_dataset(result)["errors"]
    pd.testing.assert_frame_equal(raw_customers, original)


def test_whitespace_and_actual_nulls_are_retained_as_missing(raw_customers):
    raw_customers.loc[:2, "tenure"] = 0
    raw_customers.loc[:2, "TotalCharges"] = ["\t \n", "", None]
    result = clean_dataset(raw_customers)
    assert result.TotalCharges.isna().sum() == 3
    assert result.TotalChargesMissing.sum() == 3


def test_repeatable_idempotent_and_strips_whitespace(raw_customers):
    raw_customers.loc[1, "Contract"] = "  Month-to-month\t"
    first = clean_dataset(raw_customers)
    pd.testing.assert_frame_equal(first, clean_dataset(raw_customers))
    pd.testing.assert_frame_equal(first, clean_dataset(first))
    assert first.loc[1, "Contract"] == "Month-to-month"


@pytest.mark.parametrize("column,value,message", [
    ("TotalCharges", "bad", "Invalid numeric"),
    ("MonthlyCharges", "bad", "Invalid numeric"),
    ("MonthlyCharges", np.inf, "infinite"),
    ("tenure", -1, "negative"),
    ("tenure", 1.5, "whole numbers"),
    ("SeniorCitizen", 2, "0 or 1"),
    ("MonthlyCharges", None, "missing"),
    ("TotalCharges", None, "only when tenure is zero"),
    ("TotalCharges", -5, "negative"),
    ("Churn", "Maybe", "Churn"),
    ("Churn", None, "Churn"),
    ("Contract", "Unknown", "categories"),
    ("customerID", "C0", "Duplicate customer IDs"),
    ("customerID", " C0 ", "Duplicate customer IDs"),
    ("customerID", " ", "Missing or blank customer IDs"),
    ("TechSupport", "No internet service", "Inconsistent"),
    ("MultipleLines", "No phone service", "Inconsistent"),
])
def test_rejects_invalid_or_inconsistent_values(raw_customers, column, value, message):
    raw_customers[column] = raw_customers[column].astype(object)
    raw_customers.loc[1, column] = value
    with pytest.raises(ValueError, match=message):
        clean_dataset(raw_customers)


def test_rejects_duplicate_records_and_schema_changes(raw_customers):
    with pytest.raises(ValueError, match="Duplicate customer IDs"):
        clean_dataset(pd.concat([raw_customers, raw_customers.iloc[[0]]], ignore_index=True))
    for invalid in [raw_customers.drop(columns="gender"), raw_customers.assign(extra=1)]:
        with pytest.raises(ValueError, match="schema"):
            clean_dataset(invalid)
    with pytest.raises(ValueError, match="empty"):
        clean_dataset(raw_customers.iloc[:0])


def test_fixed_group_boundaries(raw_customers):
    frame = clean_dataset(raw_customers)
    assert frame.TenureGroup.astype(str).tolist() == ["0-12 months", "0-12 months", "0-12 months", "13-24 months", "13-24 months", "25-36 months", "49-60 months", "61+ months"]
    assert frame.MonthlyChargeGroup.astype(str).tolist() == ["<50", "<50", "50 to <80", "50 to <80", "80+", "80+", "80+", "<50"]


@pytest.mark.parametrize("column,value", [("ChurnFlag", 1), ("TotalChargesMissing", False), ("tenure", -1)])
def test_clean_validator_detects_tampering(cleaned_customers, column, value):
    cleaned_customers.loc[0, column] = value
    assert validate_cleaned_dataset(cleaned_customers)["errors"]


def test_roundtrip_and_raw_file_unchanged(tmp_path, raw_customers):
    raw_path = tmp_path / "raw.csv"
    raw_customers.to_csv(raw_path, index=False)
    before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    cleaned = clean_dataset(load_dataset(raw_path))
    target = save_cleaned_dataset(cleaned, tmp_path / "processed/clean.csv")
    saved = target.read_bytes()
    save_cleaned_dataset(cleaned, target)
    assert target.read_bytes() == saved
    pd.testing.assert_frame_equal(cleaned, load_cleaned_dataset(target), check_exact=False)
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before


def test_refuses_raw_destination(cleaned_customers):
    with pytest.raises(ValueError, match="data/raw"):
        save_cleaned_dataset(cleaned_customers, DEFAULT_RAW_PATH)


def test_reload_rejects_invalid_saved_groups(tmp_path, cleaned_customers):
    path = save_cleaned_dataset(cleaned_customers, tmp_path / "clean.csv")
    frame = pd.read_csv(path)
    frame.loc[1, "TenureGroup"] = "unknown"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="TenureGroup"):
        load_cleaned_dataset(path)


def test_quality_report_exposes_policy_and_effects(raw_customers, cleaned_customers):
    report = build_quality_report(raw_customers, cleaned_customers)
    assert report["original"]["missing_values"]["TotalCharges"] == 0
    assert report["original"]["blank_values"]["TotalCharges"] == 1
    assert report["cleaned"]["missing_values"]["TotalCharges"] == 1
    assert report["effects"]["rows_removed"] == 0
    assert report["alternatives"][2]["rows"] == 7
