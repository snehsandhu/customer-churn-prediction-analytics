"""Acquire the original IBM CSV and report quality issues without cleaning it."""

from __future__ import annotations

import argparse
from http.client import HTTPException
import json
import os
from pathlib import Path
import tempfile
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data/raw/Telco-Customer-Churn.csv"
EXPECTED_COLUMNS = (
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
)
NUMERIC_COLUMNS = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
CANONICAL_ROWS = 7043


def download_dataset(path: str | Path = DEFAULT_RAW_PATH, timeout: float = 30) -> Path:
    """Acquire/reuse a strictly validated canonical CSV without replacing files.

    General-purpose loading/validation remains available for small datasets.
    Failed/short transfers leave no destination; invalid cached files are reported
    without deleting, overwriting or automatically downloading over them.
    """
    path = Path(path)
    if path.exists():
        try:
            report = validate_canonical_dataset(load_dataset(path))
            if report["errors"]:
                raise ValueError("; ".join(report["errors"]))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise RuntimeError(
                f"Existing CSV failed canonical IBM validation: {exc}. "
                f"File left unchanged: {path.resolve()}. Inspect it before any manual replacement."
            ) from exc
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with urlopen(DATA_URL, timeout=timeout) as response:
            content = response.read()
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
        # Check the schema and required values before saving a response as raw data.
        candidate = pd.read_csv(temporary_path, dtype={"customerID": "string"})
        report = validate_canonical_dataset(candidate)
        if report["errors"]:
            raise ValueError("Downloaded CSV failed validation: " + "; ".join(report["errors"]))
        # Same-directory hard-link promotion is atomic and fails if a destination
        # appeared during transfer. Unlike replace(), it cannot overwrite it.
        os.link(temporary_path, path)
    except (OSError, URLError, HTTPException, ValueError, pd.errors.ParserError) as exc:
        raise RuntimeError(
            f"Could not download the IBM dataset: {exc}\n"
            f"Manual fallback: download {DATA_URL} in a browser and save the original "
            f"CSV to {path.resolve()}. Then rerun with --download to validate "
            "the cached CSV without replacing it."
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def load_dataset(path: str | Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Read a local CSV without modifying it; retain whitespace for investigation."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset missing: {path.resolve()}. Run python -m churn_analytics.data "
            f"--download, or download {DATA_URL} manually to that path."
        )
    return pd.read_csv(path, dtype={"customerID": "string"})


def validate_dataset(df: pd.DataFrame, expected_rows: int | None = 7043) -> dict:
    """Return JSON-safe diagnostics without mutating or cleaning the frame.

    Errors identify schema, ID, target, or numeric-conversion problems. Warnings
    flag row-count differences, missing values, and whitespace for investigation.
    Numeric parsing is diagnostic only; no imputation policy is chosen in Phase 1.
    """
    errors, warnings = [], []
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_columns = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing_columns or extra_columns:
        errors.append(f"Schema mismatch: missing={missing_columns}, extra={extra_columns}")
    if df.columns.duplicated().any():
        errors.append("Duplicate column names")
        return {"errors": errors, "warnings": warnings}
    if expected_rows is not None and len(df) != expected_rows:
        warnings.append(f"Expected {expected_rows} rows; found {len(df)}")
    missing = {str(c): int(v) for c, v in df.isna().sum().items()}
    blanks = {
        str(c): int(df[c].astype("string").str.fullmatch(r"\s*").fillna(False).sum())
        for c in df.columns
    }
    if any(missing.values()):
        warnings.append("Null values detected; inspect missing_values")
    if any(blanks.values()):
        warnings.append("Whitespace-only values detected; inspect blank_values")
    duplicates = None
    if "customerID" in df:
        ids = df["customerID"]
        duplicates = int(ids.duplicated(keep="first").sum())
        if duplicates:
            errors.append(f"Duplicate customer IDs beyond first occurrence: {duplicates}")
        if ids.isna().any() or blanks["customerID"]:
            errors.append("Missing or blank customer IDs")
    target = {}
    if "Churn" in df:
        target = {str(k): int(v) for k, v in df["Churn"].value_counts(dropna=False).items()}
        if df["Churn"].isna().any() or set(df["Churn"].dropna().unique()) != {"Yes", "No"}:
            errors.append("Churn must contain both Yes and No, with no missing or other values")
    numeric_issues = {}
    for column in NUMERIC_COLUMNS:
        if column in df:
            values = df[column]
            parsed = pd.to_numeric(values, errors="coerce")
            invalid = values.notna() & parsed.isna()
            numeric_issues[column] = int(invalid.sum())
            # IBM's blank TotalCharges are reported separately, not silently fixed.
            unexpected = int(invalid.sum()) - blanks[column]
            if unexpected:
                errors.append(f"{column}: {unexpected} nonblank values cannot be parsed as numbers")
    total_charges = {}
    if {"TotalCharges", "tenure"}.issubset(df.columns):
        blank_mask = df["TotalCharges"].astype("string").str.fullmatch(r"\s*").fillna(False)
        tenure = pd.to_numeric(df["tenure"], errors="coerce")
        total_charges = {
            "blank_count": int(blank_mask.sum()),
            "blank_with_zero_tenure": int((blank_mask & tenure.eq(0)).sum()),
            "blank_with_nonzero_tenure": int((blank_mask & tenure.ne(0) & tenure.notna()).sum()),
            "blank_with_missing_or_invalid_tenure": int((blank_mask & tenure.isna()).sum()),
            "zero_tenure_count": int(tenure.eq(0).sum()),
        }
    return {
        "dimensions": {"rows": len(df), "columns": len(df.columns)},
        "column_names": list(df.columns),
        "data_types": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing_values": missing, "blank_values": blanks,
        "duplicate_customer_ids": duplicates,
        "target_class_distribution": target,
        "numeric_conversion_failures": numeric_issues,
        "total_charges_investigation": total_charges,
        "errors": errors, "warnings": warnings,
    }


def validate_canonical_dataset(df: pd.DataFrame) -> dict:
    """Require the documented IBM dimensions plus general schema/ID/value checks.

    This is a structural acquisition policy, not cryptographic authentication of
    every source value. It deliberately does not impose canonical row counts on
    validate_dataset(), which remains reusable for synthetic/filtered datasets.
    """
    report = validate_dataset(df, expected_rows=None)
    if len(df) != CANONICAL_ROWS:
        report["errors"].append(f"Canonical IBM dataset requires {CANONICAL_ROWS} rows; found {len(df)}")
    if len(df.columns) != len(EXPECTED_COLUMNS):
        report["errors"].append(f"Canonical IBM dataset requires {len(EXPECTED_COLUMNS)} columns; found {len(df.columns)}")
    report["validation_policy"] = "canonical_ibm_telco_structure"
    return report


def main() -> None:
    """Print a validation report; exit unsuccessfully for invalid input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    try:
        if args.download:
            download_dataset(args.path)
        report = validate_dataset(load_dataset(args.path))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
