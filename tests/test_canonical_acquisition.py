"""F02 regression: strict acquisition, general fixtures and no clobbered raw files."""
from http.client import IncompleteRead
from io import BytesIO

import pandas as pd
import pytest

from churn_analytics import data


@pytest.fixture
def canonical(raw_customers):
    count = data.CANONICAL_ROWS
    frame = pd.concat([raw_customers] * (count // len(raw_customers) + 1), ignore_index=True).iloc[:count].copy()
    frame.customerID = [f"CANONICAL-SYNTH-{i:05}" for i in range(count)]
    return frame


def test_canonical_shape_is_accepted_without_mutation(canonical):
    original = canonical.copy(deep=True)
    report = data.validate_canonical_dataset(canonical)
    assert not report["errors"]
    assert report["validation_policy"] == "canonical_ibm_telco_structure"
    pd.testing.assert_frame_equal(canonical, original)


def test_generic_validation_still_accepts_small_intentional_data(raw_customers):
    assert not data.validate_dataset(raw_customers, expected_rows=None)["errors"]
    assert not data.validate_dataset(raw_customers, expected_rows=len(raw_customers))["errors"]
    assert any("requires 7043 rows" in e for e in data.validate_canonical_dataset(raw_customers)["errors"])


@pytest.mark.parametrize("problem", ["truncated", "missing_column", "extra_column", "duplicate_id"])
def test_bad_canonical_download_is_rejected_without_partial_destination(tmp_path, canonical, monkeypatch, problem):
    if problem == "truncated":
        canonical = canonical.iloc[:-1]
    elif problem == "missing_column":
        canonical = canonical.drop(columns="gender")
    elif problem == "extra_column":
        canonical = canonical.assign(unexpected="value")
    else:
        canonical.loc[1, "customerID"] = canonical.loc[0, "customerID"]
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: BytesIO(canonical.to_csv(index=False).encode()))
    destination = tmp_path / "raw.csv"
    with pytest.raises(RuntimeError, match="failed validation"):
        data.download_dataset(destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_incomplete_transfer_has_fallback_and_cleans_up(tmp_path, monkeypatch):
    class BrokenResponse(BytesIO):
        def read(self, *args, **kwargs):
            raise IncompleteRead(b"partial", 100)
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: BrokenResponse())
    with pytest.raises(RuntimeError, match="Manual fallback"):
        data.download_dataset(tmp_path / "raw.csv")
    assert list(tmp_path.iterdir()) == []


def test_invalid_cached_file_is_not_replaced_or_downloaded_over(tmp_path, raw_customers, monkeypatch):
    destination = tmp_path / "cached.csv"
    raw_customers.to_csv(destination, index=False)
    original = destination.read_bytes()
    def fail(*a, **k):
        pytest.fail("Do not automatically download over an invalid cached file")
    monkeypatch.setattr(data, "urlopen", fail)
    with pytest.raises(RuntimeError, match="left unchanged"):
        data.download_dataset(destination)
    assert destination.read_bytes() == original


def test_destination_created_during_transfer_cannot_be_overwritten(tmp_path, canonical, monkeypatch):
    destination = tmp_path / "raw.csv"
    existing = b"another process created this file"
    def response(*a, **k):
        destination.write_bytes(existing)
        return BytesIO(canonical.to_csv(index=False).encode())
    monkeypatch.setattr(data, "urlopen", response)
    with pytest.raises(RuntimeError):
        data.download_dataset(destination)
    assert destination.read_bytes() == existing
    assert list(tmp_path.iterdir()) == [destination]
