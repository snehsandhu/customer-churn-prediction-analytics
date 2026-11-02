"""Supported, inference-only interface for trusted fitted LR/RF artifacts.

Always use this adapter for future application scoring. The legacy joblib contains
learned preprocessing/classification but not the original string normalization.
No fitting, model replacement, threshold tuning or artifact writing occurs here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .cleaning import CATEGORICAL_DOMAINS, INTERNET_ADDONS
from .modeling import CATEGORICAL_FEATURES, FEATURES, THRESHOLD, select_features


def normalize_inference_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an exact feature-only schema, then use canonical normalization.

    Accept columns in any order and numeric/valid-string SeniorCitizen 0/1. Reject
    extra/missing/duplicate fields, invalid levels, invalid numerics and known
    service contradictions. No trimming, case repair or unknown-level guessing
    is performed. Actual missing values retain the fitted imputation behavior.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Inference requires a pandas DataFrame")
    if not frame.columns.is_unique or set(frame.columns) != set(FEATURES):
        raise ValueError("Inference requires exactly the 18 eligible feature columns; no IDs or target fields")
    if frame.empty:
        raise ValueError("Inference requires at least one customer row")
    for column in CATEGORICAL_FEATURES:
        if column != "SeniorCitizen":
            values = frame[column].dropna()
            if not values.map(lambda value: isinstance(value, str)).all():
                raise ValueError(f"{column} requires supported string categories or missing values")
            if not values.isin(CATEGORICAL_DOMAINS[column]).all():
                raise ValueError(f"Unsupported category in {column}; no automatic category correction")
    X = select_features(frame)
    if X["tenure"].dropna().mod(1).ne(0).any():
        raise ValueError("tenure must contain whole nonnegative months or missing values")
    known_phone = X.PhoneService.notna() & X.MultipleLines.notna()
    if (X.PhoneService.eq("No") != X.MultipleLines.eq("No phone service"))[known_phone].any():
        raise ValueError("Inconsistent PhoneService and MultipleLines")
    for column in INTERNET_ADDONS:
        known = X.InternetService.notna() & X[column].notna()
        if (X.InternetService.eq("No") != X[column].eq("No internet service"))[known].any():
            raise ValueError(f"Inconsistent InternetService and {column}")
    return X


class ChurnPredictor:
    """Validate and score with an already-fitted Pipeline; intentionally no fit API.

    This generic scoring interface has no dataset-role context. Development code
    must retain the existing split/ID guards and must not pass reserved test rows.
    The future app should load this adapter once and submit feature-only records.
    """

    def __init__(self, fitted_pipeline: Pipeline, *, threshold: float = THRESHOLD):
        if not isinstance(fitted_pipeline, Pipeline) or list(fitted_pipeline.named_steps) != ["preprocessor", "classifier"]:
            raise ValueError("Expected the fitted preprocessing/classifier Pipeline")
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        classifier = fitted_pipeline.named_steps["classifier"]
        if not isinstance(preprocessor, ColumnTransformer) or not isinstance(classifier, (LogisticRegression, RandomForestClassifier)):
            raise ValueError("Expected a fitted ColumnTransformer and Logistic Regression or Random Forest")
        check_is_fitted(preprocessor, attributes=["transformers_"])
        attributes = ["coef_", "intercept_", "classes_"] if isinstance(classifier, LogisticRegression) else ["estimators_", "classes_"]
        check_is_fitted(classifier, attributes=attributes)
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Threshold must be finite and between zero and one")
        if tuple(preprocessor.feature_names_in_) != FEATURES or list(classifier.classes_) != [0, 1]:
            raise ValueError("Fitted model feature/class contract does not match this interface")
        encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        if len(encoder.categories_) != len(CATEGORICAL_FEATURES) or not all(
            isinstance(value, str) for levels in encoder.categories_ for value in levels
        ):
            raise ValueError("Fitted encoder does not use the canonical string-category contract")
        self._pipeline = fitted_pipeline
        self.threshold = float(threshold)

    @property
    def classes_(self) -> np.ndarray:
        """Return probability-column labels without exposing the stored array."""
        return self._pipeline.classes_.copy()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return P(No), P(Yes) after strict input validation and normalization."""
        X = normalize_inference_input(frame)
        probabilities = np.asarray(self._pipeline.predict_proba(X), dtype=float)
        if probabilities.shape != (len(X), 2) or not np.isfinite(probabilities).all():
            raise ValueError("Fitted model returned invalid probability dimensions/values")
        if ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-12, atol=1e-12):
            raise ValueError("Fitted model probabilities must lie in [0,1] and sum to one")
        return probabilities

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the supplied frozen inclusive threshold (default remains 0.50)."""
        return (self.predict_proba(frame)[:, 1] >= self.threshold).astype("int8")


def load_churn_predictor(path: str | Path, *, expected_sha256: str, threshold: float = THRESHOLD) -> ChurnPredictor:
    """Load ONLY a trusted local artifact after checking its expected identity.

    Pickle/joblib can execute code. An expected checksum detects a changed file;
    it does not make an untrusted file safe. No artifact is fitted or rewritten.
    """
    path = Path(path)
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected_sha256):
        raise ValueError("A valid expected SHA-256 is required before loading")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256.lower():
        raise ValueError("Model artifact SHA-256 mismatch; refusing to load")
    return ChurnPredictor(joblib.load(path), threshold=threshold)
