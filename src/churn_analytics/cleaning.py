"""Deterministic analytical cleaning; no learned imputation or model features."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .data import DEFAULT_RAW_PATH, EXPECTED_COLUMNS, NUMERIC_COLUMNS, PROJECT_ROOT, validate_dataset

DEFAULT_CLEAN_PATH = PROJECT_ROOT / "data/processed/telco_churn_cleaned.csv"
INTERNET_ADDONS = ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")
BINARY_COLUMNS = ("Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn")
CATEGORICAL_DOMAINS = {
    "gender": {"Female", "Male"},
    **{c: {"Yes", "No"} for c in BINARY_COLUMNS},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    **{c: {"Yes", "No", "No internet service"} for c in INTERNET_ADDONS},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
}
STRING_COLUMNS = ("customerID", *CATEGORICAL_DOMAINS)
TENURE_LABELS = ("0-12 months", "13-24 months", "25-36 months", "37-48 months", "49-60 months", "61+ months")
CHARGE_LABELS = ("<50", "50 to <80", "80+")
DERIVED_COLUMNS = ("ChurnFlag", "TotalChargesMissing", "TenureGroup", "MonthlyChargeGroup")
CLEAN_COLUMNS = (*EXPECTED_COLUMNS, *DERIVED_COLUMNS)


def _source_errors(frame: pd.DataFrame) -> list[str]:
    """Check canonical source values, including domain and service consistency."""
    errors = list(validate_dataset(frame, expected_rows=None)["errors"])
    for column in NUMERIC_COLUMNS:
        values = frame[column]
        if not pd.api.types.is_numeric_dtype(values):
            errors.append(f"{column} must have a numeric dtype")
            continue
        if not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
            errors.append(f"{column} contains infinite values")
        if values.lt(0).any():
            errors.append(f"{column} contains negative values")
        if column != "TotalCharges" and values.isna().any():
            errors.append(f"{column} contains missing values")
    for column in ("tenure", "SeniorCitizen"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.dropna().mod(1).ne(0).any():
            errors.append(f"{column} must contain whole numbers")
    if not frame["SeniorCitizen"].isin([0, 1]).all():
        errors.append("SeniorCitizen must contain 0 or 1")
    for column, allowed in CATEGORICAL_DOMAINS.items():
        if frame[column].isna().any() or not frame[column].isin(allowed).all():
            errors.append(f"{column} contains missing or unexpected categories")
    if (frame["TotalCharges"].isna() & frame["tenure"].ne(0)).any():
        errors.append("Missing TotalCharges is allowed only when tenure is zero; investigate other records")
    if not frame["PhoneService"].eq("No").equals(frame["MultipleLines"].eq("No phone service")):
        errors.append("Inconsistent PhoneService and MultipleLines")
    for column in INTERNET_ADDONS:
        if not frame["InternetService"].eq("No").equals(frame[column].eq("No internet service")):
            errors.append(f"Inconsistent InternetService and {column}")
    return errors


def _derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add fixed analytical labels without estimating parameters from the data."""
    result = frame.copy(deep=True)
    result["ChurnFlag"] = result["Churn"].map({"No": 0, "Yes": 1}).astype("int8")
    result["TotalChargesMissing"] = result["TotalCharges"].isna()
    result["TenureGroup"] = pd.cut(
        result["tenure"], [-1, 12, 24, 36, 48, 60, np.inf], labels=TENURE_LABELS, ordered=True,
    )
    result["MonthlyChargeGroup"] = pd.cut(
        result["MonthlyCharges"], [0, 50, 80, np.inf], right=False, labels=CHARGE_LABELS, ordered=True,
    )
    return result.loc[:, CLEAN_COLUMNS]


def clean_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Return an independent, validated 25-column analytical DataFrame.

    Preserve every row and every original column. Strip outer whitespace, parse
    numeric fields, retain missing TotalCharges at zero tenure, and add four
    documented analytical columns. Invalid values raise rather than disappear.
    Already-clean frames are accepted and derived columns are recomputed, making
    repeated calls deterministic and idempotent. IDs remain labels, not predictors.
    """
    columns = set(raw.columns)
    if not raw.columns.is_unique or columns not in (set(EXPECTED_COLUMNS), set(CLEAN_COLUMNS)):
        raise ValueError("Expected the original 21-column schema or the complete cleaned schema")
    if raw.empty:
        raise ValueError("Cannot clean an empty dataset")
    frame = raw.loc[:, EXPECTED_COLUMNS].copy(deep=True)
    for column in STRING_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip()
    for column in NUMERIC_COLUMNS:
        # Whitespace is missing information; malformed nonblank numbers are errors.
        text = frame[column].astype("string").str.strip().replace("", pd.NA)
        try:
            frame[column] = pd.to_numeric(text, errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid numeric values in {column}") from exc
    errors = _source_errors(frame)
    if errors:
        raise ValueError("Cleaning validation failed: " + "; ".join(errors))
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype("int64")
    frame["tenure"] = frame["tenure"].astype("int64")
    result = _derived_columns(frame)
    report = validate_cleaned_dataset(result)
    if report["errors"]:
        raise ValueError("Cleaned dataset failed validation: " + "; ".join(report["errors"]))
    return result


def validate_cleaned_dataset(frame: pd.DataFrame) -> dict:
    """Validate the analytical schema, retained missingness, and derived values."""
    errors = []
    if tuple(frame.columns) != CLEAN_COLUMNS or not frame.columns.is_unique:
        return {"errors": ["Cleaned schema/order must match CLEAN_COLUMNS"], "warnings": []}
    base = frame.loc[:, EXPECTED_COLUMNS]
    errors.extend(_source_errors(base))
    for column in STRING_COLUMNS:
        values = base[column].astype("string")
        if (values != values.str.strip()).fillna(False).any():
            errors.append(f"{column} still has outer whitespace")
    for column in ("tenure", "SeniorCitizen", "ChurnFlag"):
        if not pd.api.types.is_integer_dtype(frame[column]):
            errors.append(f"{column} must have an integer dtype")
    if not pd.api.types.is_bool_dtype(frame["TotalChargesMissing"]):
        errors.append("TotalChargesMissing must have a boolean dtype")
    if not errors:
        expected = _derived_columns(base)
        for column in DERIVED_COLUMNS:
            if not frame[column].astype("string").equals(expected[column].astype("string")):
                errors.append(f"{column} does not match source values")
    warnings = []
    if frame["TotalCharges"].isna().any():
        warnings.append("TotalCharges remains missing at zero tenure; report observed-value coverage")
    return {
        "dimensions": {"rows": len(frame), "columns": len(frame.columns)},
        "data_types": {c: str(t) for c, t in frame.dtypes.items()},
        "missing_values": {c: int(n) for c, n in frame.isna().sum().items()},
        "duplicate_customer_ids": int(frame["customerID"].duplicated().sum()),
        "duplicate_records": int(frame.duplicated().sum()),
        "errors": errors, "warnings": warnings,
    }


def assess_data_quality(frame: pd.DataFrame) -> dict:
    """Extend Phase 1's diagnostics with ranges, categories and row duplication."""
    base = frame.loc[:, EXPECTED_COLUMNS]
    report = validate_dataset(base)
    report["duplicate_records"] = int(frame.duplicated().sum())
    report["outer_whitespace_values"] = {
        c: int((frame[c].astype("string") != frame[c].astype("string").str.strip()).fillna(False).sum())
        for c in EXPECTED_COLUMNS
    }
    report["categorical_values"] = {
        c: {str(k): int(v) for k, v in base[c].value_counts(dropna=False).items()}
        for c in CATEGORICAL_DOMAINS
    }
    report["numeric_ranges"] = {}
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(base[column], errors="coerce")
        report["numeric_ranges"][column] = {
            "observed": int(values.notna().sum()),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "negative_count": int(values.lt(0).sum()),
            "infinite_count": int(np.isinf(values.to_numpy(dtype=float)).sum()),
        }
    return report


def build_quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Make a before/after audit, including alternative missing-charge policies."""
    observed = cleaned["TotalCharges"].notna()
    original_rate = float(cleaned["ChurnFlag"].mean())
    return {
        "original": assess_data_quality(raw),
        "cleaned": validate_cleaned_dataset(cleaned),
        "effects": {
            "rows_removed": len(raw) - len(cleaned),
            "columns_added": list(DERIVED_COLUMNS),
            "missing_total_charges_retained": int((~observed).sum()),
            "nonzero_tenure_with_missing_total_charges": int((~observed & cleaned["tenure"].ne(0)).sum()),
            "missing_total_charges_target_counts": {str(k): int(v) for k, v in cleaned.loc[~observed, "Churn"].value_counts().items()},
            "source_value_errors_after_cleaning": _source_errors(cleaned.loc[:, EXPECTED_COLUMNS]),
        },
        "policy": "Retain all rows, convert blank TotalCharges to NaN, flag missingness; do not impute or drop.",
        "alternatives": [
            {"strategy": "retain_missing (chosen)", "rows": len(cleaned), "churn_rate": original_rate,
             "mean_total_charges": float(cleaned["TotalCharges"].mean()), "charge_observations": int(observed.sum())},
            {"strategy": "impute_zero (hypothetical only)", "rows": len(cleaned), "churn_rate": original_rate,
             "mean_total_charges": float(cleaned["TotalCharges"].fillna(0).mean()), "charge_observations": len(cleaned)},
            {"strategy": "exclude_missing (hypothetical only)", "rows": int(observed.sum()),
             "churn_rate": float(cleaned.loc[observed, "ChurnFlag"].mean()),
             "mean_total_charges": float(cleaned.loc[observed, "TotalCharges"].mean()), "charge_observations": int(observed.sum())},
        ],
    }


def save_cleaned_dataset(frame: pd.DataFrame, path: str | Path = DEFAULT_CLEAN_PATH) -> Path:
    """Atomically save validated analytical data; refuse writes into data/raw/."""
    path = Path(path)
    if path.resolve().is_relative_to(DEFAULT_RAW_PATH.parent.resolve()):
        raise ValueError("Refusing to write cleaned data into data/raw/")
    report = validate_cleaned_dataset(frame)
    if report["errors"]:
        raise ValueError("Cannot save invalid cleaned data: " + "; ".join(report["errors"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, newline="", encoding="utf-8") as handle:
            temporary_path = Path(handle.name)
            frame.to_csv(handle, index=False, na_rep="")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def load_cleaned_dataset(path: str | Path = DEFAULT_CLEAN_PATH) -> pd.DataFrame:
    """Reload the CSV with documented types and revalidate the saved artifact."""
    frame = pd.read_csv(path, dtype={c: "string" for c in STRING_COLUMNS})
    for column, labels in (("TenureGroup", TENURE_LABELS), ("MonthlyChargeGroup", CHARGE_LABELS)):
        if column not in frame or not frame[column].isin(labels).all():
            raise ValueError(f"Missing or invalid {column}")
        frame[column] = pd.Categorical(frame[column], categories=labels, ordered=True)
    report = validate_cleaned_dataset(frame)
    if report["errors"]:
        raise ValueError("Invalid processed CSV: " + "; ".join(report["errors"]))
    frame["ChurnFlag"] = frame["ChurnFlag"].astype("int8")
    return frame
