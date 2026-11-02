"""Offline tests: synthetic records test behavior, never model performance."""
from io import BytesIO

import pandas as pd
import pytest

from churn_analytics import data


@pytest.fixture
def frame():
    """A minimal complete schema with IBM-like whitespace in TotalCharges."""
    values = {column: ["No", "Yes"] for column in data.EXPECTED_COLUMNS}
    values.update(customerID=["A", "B"], SeniorCitizen=[0, 1], tenure=[0, 12],
                  MonthlyCharges=[20.0, 50.0], TotalCharges=[" ", "600"], Churn=["No", "Yes"])
    return pd.DataFrame(values)


@pytest.fixture
def canonical_frame(raw_customers):
    """Canonical-size synthetic CSV, not fabricated IBM analytical evidence."""
    count = data.CANONICAL_ROWS
    fixture = pd.concat([raw_customers] * (count // len(raw_customers) + 1), ignore_index=True).iloc[:count].copy()
    fixture["customerID"] = [f"SYNTH-{i:05}" for i in range(count)]
    return fixture


def test_load_preserves_raw_values(tmp_path, frame):
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    original = path.read_bytes()
    loaded = data.load_dataset(path)
    assert loaded.shape == (2, 21)
    assert loaded.loc[0, "TotalCharges"] == " "
    assert path.read_bytes() == original


def test_missing_file_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="--download"):
        data.load_dataset(tmp_path / "absent.csv")


def test_validation_reports_without_mutating(frame):
    before = frame.copy(deep=True)
    report = data.validate_dataset(frame, expected_rows=2)
    assert report["errors"] == []
    assert report["dimensions"] == {"rows": 2, "columns": 21}
    assert report["duplicate_customer_ids"] == 0
    assert report["target_class_distribution"] == {"No": 1, "Yes": 1}
    assert report["missing_values"]["TotalCharges"] == 0
    assert report["blank_values"]["TotalCharges"] == 1
    assert report["total_charges_investigation"]["blank_with_zero_tenure"] == 1
    assert report["numeric_conversion_failures"]["TotalCharges"] == 1
    pd.testing.assert_frame_equal(frame, before)


def test_bad_schema(frame):
    report = data.validate_dataset(frame.drop(columns="gender").assign(unexpected=1))
    assert any("Schema mismatch" in error for error in report["errors"])
    assert report["warnings"]


@pytest.mark.parametrize("column,value,message", [
    ("customerID", "A", "Duplicate customer IDs"),
    ("customerID", " ", "Missing or blank customer IDs"),
    ("Churn", "Maybe", "Churn must"),
    ("Churn", None, "Churn must"),
    ("tenure", "bad", "cannot be parsed"),
    ("TotalCharges", "bad", "cannot be parsed"),
])
def test_invalid_values(frame, column, value, message):
    frame[column] = frame[column].astype(object)
    frame.loc[1, column] = value
    report = data.validate_dataset(frame, expected_rows=None)
    assert any(message in error for error in report["errors"])


def test_nulls_and_nonzero_tenure(frame):
    frame.loc[0, "tenure"] = 2
    frame.loc[1, "MonthlyCharges"] = None
    report = data.validate_dataset(frame)
    assert report["missing_values"]["MonthlyCharges"] == 1
    assert report["total_charges_investigation"]["blank_with_nonzero_tenure"] == 1


def test_download_and_cache(tmp_path, canonical_frame, monkeypatch):
    calls = []
    def fake_urlopen(url, timeout):
        calls.append(url)
        return BytesIO(canonical_frame.to_csv(index=False).encode())
    monkeypatch.setattr(data, "urlopen", fake_urlopen)
    path = tmp_path / "raw.csv"
    assert data.download_dataset(path) == path
    assert data.download_dataset(path) == path
    assert len(calls) == 1
    assert data.load_dataset(path).shape == (data.CANONICAL_ROWS, 21)


def test_network_failure_has_manual_fallback(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("offline")
    monkeypatch.setattr(data, "urlopen", fail)
    path = tmp_path / "raw.csv"
    with pytest.raises(RuntimeError, match="Manual fallback"):
        data.download_dataset(path)
    assert not path.exists()


def test_invalid_download_leaves_no_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: BytesIO(b"wrong\nvalue\n"))
    path = tmp_path / "raw.csv"
    with pytest.raises(RuntimeError, match="failed validation"):
        data.download_dataset(path)
    assert list(tmp_path.iterdir()) == []


def test_duplicate_column_names(frame):
    duplicate = pd.concat([frame, frame[["tenure"]]], axis=1)
    assert "Duplicate column names" in data.validate_dataset(duplicate)["errors"]


def test_existing_raw_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "raw.csv"
    path.write_bytes(b"keep original bytes")
    def fail(*args, **kwargs):
        pytest.fail("Existing raw data must not trigger a download")
    monkeypatch.setattr(data, "urlopen", fail)
    # Invalid cached content is now rejected, but never replaced or downloaded over.
    with pytest.raises(RuntimeError, match="Existing CSV failed canonical"):
        data.download_dataset(path)
    assert path.read_bytes() == b"keep original bytes"


def test_cli_success_and_error_status(tmp_path, frame, monkeypatch, capsys):
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr("sys.argv", ["data", "--path", str(path)])
    data.main()
    assert '"rows": 2' in capsys.readouterr().out
    frame.loc[1, "Churn"] = "Unknown"
    frame.to_csv(path, index=False)
    with pytest.raises(SystemExit) as exc:
        data.main()
    assert exc.value.code == 1
