"""F01 regression: the supported adapter preserves an already-fitted estimator.

Only small synthetic models are fitted in these tests, never the protected IBM
artifact. All model files created here live in pytest temporary directories.
"""
import hashlib
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from churn_analytics.inference import ChurnPredictor, load_churn_predictor, normalize_inference_input
from churn_analytics.model_splits import DevelopmentSplit, create_split_manifest, development_split
from churn_analytics.modeling import FEATURES, build_pipeline, evaluate_pipeline, fit_pipeline, select_features


@pytest.fixture
def fitted_synthetic(cleaned_customers):
    pipeline = build_pipeline("logistic_standard")
    pipeline.fit(select_features(cleaned_customers), cleaned_customers.ChurnFlag)
    return pipeline


@pytest.mark.parametrize("representation", ["integers", "strings", "binary_floats"])
def test_numeric_and_string_senior_forms_match_normalized_predictions(cleaned_customers, fitted_synthetic, representation):
    X = cleaned_customers.loc[:, FEATURES].copy()
    expected = fitted_synthetic.predict_proba(select_features(X))
    if representation == "strings":
        X.SeniorCitizen = X.SeniorCitizen.astype(str)
    elif representation == "binary_floats":
        X.SeniorCitizen = X.SeniorCitizen.astype(float)
    original = X.copy(deep=True)
    predictor = ChurnPredictor(fitted_synthetic)
    np.testing.assert_allclose(predictor.predict_proba(X), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(predictor.predict(X), (expected[:, 1] >= 0.5).astype(int))
    pd.testing.assert_frame_equal(X, original)
    assert not hasattr(predictor, "fit")
    assert predictor.classes_.tolist() == [0, 1]


@pytest.mark.parametrize("value", [2, -1, 0.5, "1.0", "01", "yes", " 1", True, False, [1]])
def test_invalid_senior_values_are_not_silently_coerced(cleaned_customers, value):
    X = cleaned_customers.loc[:, FEATURES].copy()
    X.SeniorCitizen = X.SeniorCitizen.astype(object)
    X.at[0, "SeniorCitizen"] = value
    with pytest.raises(ValueError, match="SeniorCitizen"):
        normalize_inference_input(X)


@pytest.mark.parametrize("column,value", [("gender", "female"), ("Contract", " Month-to-month"),
    ("InternetService", "Fiber Optic"), ("PaymentMethod", "unknown"), ("PhoneService", 1),
    ("TechSupport", "__MISSING__")])
def test_unsupported_categories_fail_before_model_prediction(cleaned_customers, fitted_synthetic, column, value):
    X = cleaned_customers.loc[:, FEATURES].copy()
    X[column] = X[column].astype(object)
    X.at[0, column] = value
    predictor = ChurnPredictor(fitted_synthetic)
    fitted_synthetic.predict_proba = Mock(side_effect=AssertionError("Must fail before scoring"))
    with pytest.raises(ValueError):
        predictor.predict_proba(X)
    fitted_synthetic.predict_proba.assert_not_called()


def test_strict_schema_and_reordering(cleaned_customers, fitted_synthetic):
    X = cleaned_customers.loc[:, FEATURES].copy()
    predictor = ChurnPredictor(fitted_synthetic)
    np.testing.assert_allclose(predictor.predict_proba(X.iloc[:, ::-1]), predictor.predict_proba(X))
    invalid = [X.drop(columns="tenure"), X.assign(ChurnFlag=0),
               pd.concat([X, X[["tenure"]]], axis=1), X.iloc[:0]]
    for frame in invalid:
        with pytest.raises(ValueError):
            predictor.predict_proba(frame)
    with pytest.raises(ValueError, match="DataFrame"):
        predictor.predict_proba(X.to_numpy())


@pytest.mark.parametrize("column,value", [("tenure", -1), ("tenure", 1.5),
    ("MonthlyCharges", -1), ("MonthlyCharges", np.inf), ("MonthlyCharges", "invalid")])
def test_invalid_numeric_inputs_are_rejected(cleaned_customers, column, value):
    X = cleaned_customers.loc[:, FEATURES].copy()
    X[column] = X[column].astype(object)
    X.at[0, column] = value
    with pytest.raises(ValueError):
        normalize_inference_input(X)


def test_service_contradictions_are_rejected(cleaned_customers):
    X = cleaned_customers.loc[:, FEATURES].copy()
    X.loc[0, "TechSupport"] = "No internet service"  # DSL is still present.
    with pytest.raises(ValueError, match="Inconsistent InternetService"):
        normalize_inference_input(X)


def test_missing_values_keep_trained_imputation_contract(cleaned_customers, fitted_synthetic):
    X = cleaned_customers.loc[:, FEATURES].copy()
    X.loc[0, "MonthlyCharges"] = np.nan
    X.loc[0, "gender"] = None
    X.SeniorCitizen = X.SeniorCitizen.astype(object)
    X.loc[0, "SeniorCitizen"] = None
    with pytest.warns(UserWarning, match="unknown categories"):
        probabilities = ChurnPredictor(fitted_synthetic).predict_proba(X)
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_adapter_does_not_modify_learned_state(cleaned_customers, fitted_synthetic):
    before = joblib.hash(fitted_synthetic)
    predictor = ChurnPredictor(fitted_synthetic)
    predictor.predict_proba(cleaned_customers.loc[:, FEATURES])
    assert joblib.hash(fitted_synthetic) == before
    labels = predictor.classes_
    labels[0] = 7  # In range for the fitted int8 labels, but outside the class set.
    assert predictor.classes_.tolist() == [0, 1]


def test_trusted_loader_integrity_and_unchanged_file(tmp_path, cleaned_customers, fitted_synthetic):
    path = tmp_path / "synthetic_owned_model.joblib"
    joblib.dump(fitted_synthetic, path)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    predictor = load_churn_predictor(path, expected_sha256=fingerprint)
    np.testing.assert_allclose(predictor.predict_proba(cleaned_customers.loc[:, FEATURES]),
                               fitted_synthetic.predict_proba(select_features(cleaned_customers)))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == fingerprint
    with pytest.raises(ValueError, match="mismatch"):
        load_churn_predictor(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="valid expected"):
        load_churn_predictor(path, expected_sha256="not-a-fingerprint")


def test_unfitted_and_wrong_pipeline_rejected():
    with pytest.raises(NotFittedError):
        ChurnPredictor(build_pipeline("logistic_standard"))
    with pytest.raises(ValueError):
        ChurnPredictor(build_pipeline("dummy_most_frequent"))


def test_guarded_fitting_and_evaluation_normalize_direct_phase2_dtypes(cleaned_customers):
    frame = pd.concat([cleaned_customers] * 16, ignore_index=True)
    frame.customerID = [f"CORRECTION-{i:04}" for i in range(len(frame))]
    manifest = create_split_manifest(frame, "synthetic")
    train = development_split(frame, manifest, "train")
    validation = development_split(frame, manifest, "validation")
    model = build_pipeline("logistic_standard")
    raw_train_X = frame.set_index("customerID").loc[manifest["splits"]["train"]["customer_ids"], list(FEATURES)]
    fit_pipeline(model, DevelopmentSplit("train", raw_train_X, train.y), manifest)
    _, expected = evaluate_pipeline(model, validation, manifest)
    raw_X = frame.set_index("customerID").loc[manifest["splits"]["validation"]["customer_ids"], list(FEATURES)]
    direct = DevelopmentSplit("validation", raw_X, validation.y)
    _, actual = evaluate_pipeline(model, direct, manifest)
    np.testing.assert_allclose(actual.churn_probability, expected.churn_probability, rtol=1e-12, atol=1e-12)
    assert actual.prediction_at_0_50.tolist() == expected.prediction_at_0_50.tolist()
