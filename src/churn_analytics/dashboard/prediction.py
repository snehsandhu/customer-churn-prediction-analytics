"""In-memory scoring through the preserved F01 adapter, without training hooks."""
import csv
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import PROJECT_ROOT
from ..inference import load_churn_predictor, normalize_inference_input
from ..modeling import FEATURES
from .artifacts import DashboardError, MODEL_PATH, verify_final_bundle

MAX_UPLOAD_BYTES = 1_048_576
MAX_UPLOAD_ROWS = 500
DEFAULT_RECORD = {"tenure": 12, "MonthlyCharges": 70.0, "gender": "Female", "SeniorCitizen": 0,
    "Partner": "No", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"}


def load_predictor(root: str | Path = PROJECT_ROOT):
    """Validate metadata/fingerprint before loading this project's trusted model."""
    selection = verify_final_bundle(root)
    try:
        return load_churn_predictor(Path(root)/MODEL_PATH, expected_sha256=selection["artifact_sha256"], threshold=selection["threshold"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError) as exc:
        raise DashboardError("The trusted selected model cannot be loaded. Restore its compatible Phase 5 artifact.") from exc


def validate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Require complete inputs within explicit application bounds; never fix typos."""
    try:
        if not isinstance(frame, pd.DataFrame) or frame.empty or len(frame) > MAX_UPLOAD_ROWS:
            raise ValueError("Rows")
        if not frame.columns.is_unique or set(frame.columns) != set(FEATURES):
            raise ValueError("Feature schema")
        if frame.isna().any().any() or any(frame[c].astype(str).str.strip().eq("").any() for c in frame):
            raise ValueError("Missing fields")
        for column in ("tenure", "MonthlyCharges"):
            if column in frame and frame[column].map(lambda x: isinstance(x, (bool, np.bool_))).any():
                raise ValueError("Boolean numeric field")
        X = normalize_inference_input(frame)
        if not X.tenure.between(0, 72).all() or not X.MonthlyCharges.between(0, 200).all():
            raise ValueError("Application bounds")
        return X
    except (ValueError, TypeError, KeyError) as exc:
        raise DashboardError("Check all 18 required fields: exact categories, whole tenure 0–72 months, charges 0–200, SeniorCitizen 0/1, and consistent phone/internet services.") from exc


def score_inputs(frame: pd.DataFrame, predictor) -> pd.DataFrame:
    """Score only caller-supplied records once; apply the frozen threshold."""
    X = validate_inputs(frame)
    if predictor.threshold != 0.50:
        raise DashboardError("Prediction is blocked: the model threshold must remain 50%.")
    try:
        p = predictor.predict_proba(X)[:, 1]
    except (ValueError, TypeError, RuntimeError) as exc:
        raise DashboardError("The model could not score these inputs. Check the input contract and trusted artifact.") from exc
    return pd.DataFrame({"row_index": np.arange(1, len(X)+1), "predicted_label": np.where(p >= .50, "Churn", "No Churn"),
                         "churn_probability": p, "decision_threshold": .50})


def parse_upload(content: bytes) -> pd.DataFrame:
    """Parse bounded UTF-8 CSV, checking duplicate headers before Pandas mangles them.

    Do not save or globally cache uploaded bytes. No identifier/target columns are
    accepted. A generated 1-based row index preserves source order in outputs.
    """
    if not isinstance(content, bytes) or not 0 < len(content) <= MAX_UPLOAD_BYTES:
        raise DashboardError("Upload a nonempty UTF-8 CSV up to 1 MiB and 500 rows.")
    try:
        text = content.decode("utf-8-sig")
        reader = csv.reader(StringIO(text), strict=True)
        header = next(reader)
        if len(header) != len(set(header)) or set(header) != set(FEATURES):
            raise ValueError("Header")
        rows = []
        for row in reader:
            if len(row) != len(header) or len(rows) >= MAX_UPLOAD_ROWS:
                raise ValueError("Rows/fields")
            rows.append(row)
        if not rows:
            raise ValueError("Empty")
        frame = pd.DataFrame(rows, columns=header)
        validate_inputs(frame)
        return frame
    except (UnicodeError, csv.Error, ValueError, StopIteration) as exc:
        raise DashboardError("Invalid CSV. Use exactly 18 unique feature headers, complete valid values and at most 500 rows; identifiers and target columns are not accepted.") from exc


def template_csv() -> bytes:
    """One invented, explicitly illustrative feature-only record, never test data."""
    return pd.DataFrame([DEFAULT_RECORD], columns=FEATURES).to_csv(index=False).encode("utf-8")
