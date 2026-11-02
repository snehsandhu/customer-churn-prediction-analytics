"""Leakage-aware feature policy, guarded sklearn training and validation metrics."""

from __future__ import annotations

import warnings
from numbers import Real

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cleaning import CLEAN_COLUMNS
from .model_splits import DEVELOPMENT_ROLES, DevelopmentSplit, RANDOM_STATE, split_identity

NUMERIC_FEATURES = ("tenure", "MonthlyCharges")
CATEGORICAL_FEATURES = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
)
FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
EXCLUDED_FEATURES = {
    "customerID": "Identifier; retain only as partition/index metadata, never a predictor.",
    "Churn": "Outcome label; used only to validate ChurnFlag mapping.",
    "ChurnFlag": "Prediction target, kept separate from features.",
    "TotalCharges": "Cumulative amount has no documented pre-outcome availability; excluded from the primary baseline.",
    "TotalChargesMissing": "Derived from an amount with uncertain prediction-time availability; excluded.",
    "TenureGroup": "Redundant fixed grouping of included numeric tenure; excluded.",
    "MonthlyChargeGroup": "Redundant fixed grouping of included MonthlyCharges; excluded.",
}
MODEL_NAMES = ("dummy_most_frequent", "logistic_standard", "logistic_balanced")
THRESHOLD = 0.50


def feature_policy() -> pd.DataFrame:
    """Explain every Phase 2 column; plausible availability is an assumption."""
    rows = []
    for column in CLEAN_COLUMNS:
        included = column in FEATURES
        reason = EXCLUDED_FEATURES.get(column, "Customer/service/billing attribute plausibly known before a churn decision; timestamp availability remains unverified.")
        if column in NUMERIC_FEATURES:
            reason = "Recorded duration/current recurring charge plausibly known before churn; snapshot timing is unverified. Use train-only imputation/scaling."
        if column in ("gender", "SeniorCitizen"):
            reason += " Demographic inclusion requires suitability/fairness assessment before any operational use."
        rows.append({"column": column, "included": included,
                     "model_type": "numeric" if column in NUMERIC_FEATURES else ("categorical" if included else "excluded"),
                     "reason": reason})
    return pd.DataFrame(rows)


def _senior_citizen_label(value):
    """Normalize binary numeric/string forms without guessing other categories."""
    if not pd.api.types.is_scalar(value):
        raise ValueError("SeniorCitizen must be numeric 0/1, string '0'/'1', or missing")
    if pd.isna(value):
        return np.nan
    if isinstance(value, str) and value in {"0", "1"}:
        return value
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and value in (0, 1):
        return str(int(value))
    raise ValueError("SeniorCitizen must be numeric 0/1, string '0'/'1', or missing")


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical feature selection/type normalization; never estimate parameters.

    Used by development data preparation, guarded evaluation and the supported
    inference adapter. Complete cleaned records may contain excluded fields;
    inference.py separately enforces an exact feature-only scoring schema.
    """
    if not frame.columns.is_unique or not set(FEATURES).issubset(frame.columns):
        raise ValueError("Missing/duplicate eligible feature columns")
    X = frame.loc[:, FEATURES].copy(deep=True)
    for column in NUMERIC_FEATURES:
        X[column] = pd.to_numeric(X[column], errors="raise").astype("float64")
        if not np.isfinite(X[column].dropna()).all() or X[column].lt(0).any():
            raise ValueError(f"Invalid numeric feature {column}")
    for column in CATEGORICAL_FEATURES:
        # Uniform string levels, including SeniorCitizen 0/1; np.nan is compatible
        # with SimpleImputer. This conversion does not learn category vocabularies.
        values = X[column].map(_senior_citizen_label) if column == "SeniorCitizen" else X[column]
        text = values.astype("string")
        X[column] = text.astype(object).where(text.notna(), np.nan)
    return X


def build_preprocessor() -> ColumnTransformer:
    """Estimate medians, scales and category vocabulary only inside Pipeline.fit."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ], remainder="drop", verbose_feature_names_out=True)


def build_pipeline(name: str) -> Pipeline:
    """Build fresh, unfitted baselines without hyperparameter search.

    C=1 and lbfgs use L2 regularization. The obsolete penalty/n_jobs arguments are
    omitted; this supports installed sklearn 1.9 and the older supported releases.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown Phase 4 model: {name}")
    if name == "dummy_most_frequent":
        classifier = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    else:
        classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, tol=1e-6,
                                        class_weight="balanced" if name == "logistic_balanced" else None,
                                        random_state=RANDOM_STATE)
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])


def guard_development_data(data: DevelopmentSplit, manifest: dict, *, fitting: bool = False) -> None:
    """Reject reserved/overlapping/spoofed IDs before fit, transform or prediction."""
    if data.role not in DEVELOPMENT_ROLES or (fitting and data.role != "train"):
        raise ValueError("Test data is reserved; fit only training, evaluate only train/validation")
    expected = manifest["splits"][data.role]["customer_ids"]
    if list(data.X.index.astype(str)) != expected or not data.X.index.is_unique:
        raise ValueError("Customer IDs do not match the declared development split")
    if set(data.X.index.astype(str)) & set(manifest["splits"]["test"]["customer_ids"]):
        raise ValueError("Reserved test customer found in development data")
    if tuple(data.X.columns) != FEATURES or not data.y.index.equals(data.X.index):
        raise ValueError("Feature allowlist or target alignment is invalid")
    if not data.y.isin([0, 1]).all() or data.y.nunique() != 2:
        raise ValueError("Development target must contain both binary classes")
    if manifest["test_evaluated"] is not False or manifest["manifest_id"] != split_identity(manifest):
        raise ValueError("Invalid/reserved split manifest")


def fit_pipeline(pipeline: Pipeline, train: DevelopmentSplit, manifest: dict) -> dict:
    """Fit preprocessing and classifier on train only; fail on convergence issues."""
    guard_development_data(train, manifest, fitting=True)
    normalized_train = select_features(train.X)
    if normalized_train.loc[:, NUMERIC_FEATURES].isna().all().any():
        raise ValueError("Cannot learn a median from an entirely missing numeric training feature")
    if hasattr(pipeline, "_phase4_manifest_id"):
        raise ValueError("This pipeline is already fitted; build a new pipeline instead of accidental refitting")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pipeline.fit(normalized_train, train.y)
    convergence = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
    other = [str(w.message) for w in captured if not issubclass(w.category, ConvergenceWarning)]
    classifier = pipeline.named_steps["classifier"]
    iterations = classifier.n_iter_.tolist() if hasattr(classifier, "n_iter_") else None
    if convergence or (iterations and max(iterations) >= classifier.max_iter):
        raise RuntimeError(f"Logistic Regression did not meet the convergence check: {convergence}, iterations={iterations}")
    pipeline._phase4_manifest_id = manifest["manifest_id"]
    pipeline._phase4_training_ids = tuple(manifest["splits"]["train"]["customer_ids"])
    return {"training_customers": len(train.y), "n_iter": iterations,
            "convergence_warnings": convergence, "other_fit_warnings": other,
            "converged": True if iterations is not None else "not applicable (dummy)"}


def classification_metrics(y_true, probabilities, threshold: float = THRESHOLD) -> dict:
    """Evaluate binary probabilities; AP is non-interpolated, not trapezoid PR-AUC."""
    truth = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if truth.ndim != 1 or p.ndim != 1 or len(truth) == 0 or len(truth) != len(p):
        raise ValueError("Nonempty aligned one-dimensional targets/probabilities required")
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary targets or probabilities outside [0,1]")
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one")
    predictions = (p >= threshold).astype(int)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    return {
        "threshold": threshold, "customers": len(truth),
        "class_support": {str(k): int((truth == k).sum()) for k in [0, 1]},
        "churn_proportion": float(truth.mean()), "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, p)) if np.unique(truth).size == 2 else None,
        "average_precision": float(average_precision_score(truth, p)) if truth.sum() else None,
        "confusion_matrix": matrix.tolist(),
        "probability_summary": {"minimum": float(p.min()), "maximum": float(p.max()), "mean": float(p.mean()),
                                "std": float(p.std()), **{f"q{q:02}": float(np.percentile(p, q)) for q in [5, 25, 50, 75, 95]}},
    }


def evaluate_pipeline(pipeline: Pipeline, data: DevelopmentSplit, manifest: dict) -> tuple[dict, pd.DataFrame]:
    """Guard IDs first, then make predictions only for an allowed development set."""
    guard_development_data(data, manifest)
    if getattr(pipeline, "_phase4_manifest_id", None) != manifest["manifest_id"]:
        raise ValueError("Model was not fitted on this manifest's training partition")
    if tuple(pipeline._phase4_training_ids) != tuple(manifest["splits"]["train"]["customer_ids"]):
        raise ValueError("Model training-ID provenance mismatch")
    if list(pipeline.classes_) != [0, 1]:
        raise ValueError("Expected probability columns for classes [0,1]")
    # Idempotent normalization also protects callers constructing a valid-role
    # DevelopmentSplit directly from Phase 2 dtypes instead of using the helper.
    probabilities = pipeline.predict_proba(select_features(data.X))[:, 1]
    metrics = classification_metrics(data.y, probabilities)
    metrics["split_role"] = data.role
    predictions = pd.DataFrame({"customerID": data.X.index.astype(str), "split": data.role,
                                "y_true": data.y.to_numpy(), "churn_probability": probabilities,
                                "prediction_at_0_50": (probabilities >= THRESHOLD).astype(int)})
    return metrics, predictions


def coefficient_report(pipeline: Pipeline) -> tuple[pd.DataFrame, dict]:
    """Align names, source columns, reference levels and conditional odds ratios."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression) or not hasattr(classifier, "coef_"):
        raise ValueError("A fitted Logistic Regression pipeline is required")
    names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    if len(names) != len(coefficients):
        raise ValueError("Coefficient/feature-name alignment failed")
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    metadata = [{"source_feature": column, "category": None, "reference_category": None,
                 "scale": "per training standard deviation", "training_scale": float(scaler.scale_[i])}
                for i, column in enumerate(NUMERIC_FEATURES)]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    references = {}
    for column, categories, dropped in zip(CATEGORICAL_FEATURES, encoder.categories_, encoder.drop_idx_):
        reference = str(categories[dropped])
        references[column] = reference
        for index, level in enumerate(categories):
            if index != dropped:
                metadata.append({"source_feature": column, "category": str(level), "reference_category": reference,
                                 "scale": "category vs training-fitted reference", "training_scale": None})
    if len(metadata) != len(names):
        raise ValueError("Categorical metadata alignment failed")
    report = pd.DataFrame(metadata)
    report.insert(0, "transformed_feature", names)
    report["coefficient"] = coefficients
    report["conditional_odds_ratio"] = np.exp(coefficients)
    if not np.isfinite(report.conditional_odds_ratio).all():
        raise ValueError("Nonfinite coefficient odds ratio")
    info = {"intercept": float(classifier.intercept_[0]), "reference_categories": references,
            "numeric_training_scales": {column: float(scaler.scale_[i]) for i, column in enumerate(NUMERIC_FEATURES)},
            "interpretation": "Regularized fitted associations, not causal importance; numeric effects are per training SD and categories compare with saved reference levels."}
    return report, info
