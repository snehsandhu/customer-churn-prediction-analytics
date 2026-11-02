"""Phase 4 synthetic-data checks; no network and no reserved-test predictions."""
from dataclasses import replace
import hashlib
import json
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn_analytics.cleaning import save_cleaned_dataset
from churn_analytics.model_splits import (
    DevelopmentSplit, create_split_manifest, development_split, split_identity,
    validate_split_manifest, validate_target,
)
from churn_analytics.modeling import (
    CATEGORICAL_FEATURES, EXCLUDED_FEATURES, FEATURES, NUMERIC_FEATURES,
    build_pipeline, classification_metrics, coefficient_report,
    evaluate_pipeline, feature_policy, fit_pipeline, select_features,
)
from churn_analytics.phase4 import run_phase4


@pytest.fixture
def modeling_frame(cleaned_customers):
    """128 valid synthetic customers; not a source of portfolio performance claims."""
    frame = pd.concat([cleaned_customers] * 16, ignore_index=True)
    frame["customerID"] = pd.Series([f"S{i:04}" for i in range(len(frame))], dtype="string")
    return frame


@pytest.fixture
def development(modeling_frame):
    manifest = create_split_manifest(modeling_frame, "synthetic-fingerprint")
    return manifest, development_split(modeling_frame, manifest, "train"), development_split(modeling_frame, manifest, "validation")


def test_target_mapping_is_existing_validated_binary_target(modeling_frame):
    target = validate_target(modeling_frame)
    assert target.tolist() == [0, 1, 1, 0, 1, 0, 0, 0] * 16
    assert target.dtype == np.dtype("int8")


@pytest.mark.parametrize("column,value", [("Churn", "Maybe"), ("ChurnFlag", 2), ("ChurnFlag", 1)])
def test_invalid_target_mapping_is_rejected(modeling_frame, column, value):
    modeling_frame[column] = modeling_frame[column].astype(object)
    modeling_frame.loc[0, column] = value
    with pytest.raises(ValueError, match="Target mapping"):
        validate_target(modeling_frame)


def test_duplicate_customer_ids_are_rejected(modeling_frame):
    modeling_frame.loc[1, "customerID"] = modeling_frame.loc[0, "customerID"]
    with pytest.raises(ValueError, match="unique"):
        create_split_manifest(modeling_frame, "synthetic")


def test_every_feature_has_a_policy_and_outcomes_cannot_leak(modeling_frame):
    policy = feature_policy()
    assert len(policy) == 25 and policy.included.sum() == 18
    assert len(NUMERIC_FEATURES) == 2 and len(CATEGORICAL_FEATURES) == 16
    X = select_features(modeling_frame)
    assert tuple(X.columns) == FEATURES
    assert not set(X.columns) & set(EXCLUDED_FEATURES)
    altered = modeling_frame.copy(deep=True)
    for column in EXCLUDED_FEATURES:
        altered[column] = "must not reach the classifier"
    pd.testing.assert_frame_equal(X, select_features(altered))


def test_splits_are_disjoint_complete_reproducible_and_stratified(modeling_frame):
    first = create_split_manifest(modeling_frame, "synthetic")
    assert first == create_split_manifest(modeling_frame, "synthetic")
    assert first == create_split_manifest(modeling_frame.sample(frac=1, random_state=7), "synthetic")
    groups = [set(first["splits"][r]["customer_ids"]) for r in ["train", "validation", "test"]]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert set.union(*groups) == set(modeling_frame.customerID)
    assert sum(len(g) for g in groups) == len(modeling_frame)
    for info in first["splits"].values():
        assert abs(info["churn_proportion"] - modeling_frame.ChurnFlag.mean()) <= 2 / info["size"]
    assert first["test_evaluated"] is False


def test_split_fingerprint_and_overlap_checks(modeling_frame):
    manifest = create_split_manifest(modeling_frame, "original")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_split_manifest(modeling_frame, manifest, "changed")
    manifest["splits"]["validation"]["customer_ids"][0] = manifest["splits"]["train"]["customer_ids"][0]
    manifest["manifest_id"] = split_identity(manifest)
    with pytest.raises(ValueError):
        validate_split_manifest(modeling_frame, manifest, "original")


def test_test_partition_cannot_be_materialized_for_development(modeling_frame, development):
    manifest, _, _ = development
    with pytest.raises(ValueError, match="Test split is reserved"):
        development_split(modeling_frame, manifest, "test")


def test_validation_and_test_fitting_are_blocked_before_fit(development):
    manifest, _, validation = development
    pipeline = build_pipeline("logistic_standard")
    pipeline.fit = Mock(side_effect=AssertionError("Must not fit"))
    for invalid in [validation, replace(validation, role="test")]:
        with pytest.raises(ValueError, match="Test data is reserved"):
            fit_pipeline(pipeline, invalid, manifest)
    pipeline.fit.assert_not_called()


def test_test_and_spoofed_test_ids_are_blocked_before_prediction(development):
    manifest, _, validation = development
    predictor = Mock()
    with pytest.raises(ValueError, match="Test data is reserved"):
        evaluate_pipeline(predictor, replace(validation, role="test"), manifest)
    test_ids = manifest["splits"]["test"]["customer_ids"]
    spoofed = DevelopmentSplit("validation", validation.X.set_axis(test_ids), validation.y.set_axis(test_ids))
    with pytest.raises(ValueError, match="Customer IDs"):
        evaluate_pipeline(predictor, spoofed, manifest)
    predictor.predict_proba.assert_not_called()


def test_preprocessing_fits_only_training_statistics_and_categories(development):
    manifest, train, validation = development
    train.X.loc[:, "tenure"] = 2.0
    train.X.iloc[0, train.X.columns.get_loc("tenure")] = np.nan
    train.X.iloc[1, train.X.columns.get_loc("tenure")] = 4.0
    validation.X.loc[:, "tenure"] = 9999.0
    validation.X.loc[:, "InternetService"] = "VALIDATION_ONLY_PROVIDER"
    pipeline = build_pipeline("logistic_standard")
    fit_pipeline(pipeline, train, manifest)
    prep = pipeline.named_steps["preprocessor"]
    numeric = prep.named_transformers_["numeric"]
    assert numeric.named_steps["imputer"].statistics_[0] == 2
    assert numeric.named_steps["scaler"].mean_[0] == pytest.approx(2 + 2 / len(train.y))
    assert numeric.named_steps["scaler"].n_samples_seen_ == len(train.y)
    before = numeric.named_steps["scaler"].mean_.copy()
    with pytest.warns(UserWarning, match="unknown categories"):
        metrics, predictions = evaluate_pipeline(pipeline, validation, manifest)
    np.testing.assert_array_equal(before, numeric.named_steps["scaler"].mean_)
    encoder = prep.named_transformers_["categorical"].named_steps["onehot"]
    index = list(CATEGORICAL_FEATURES).index("InternetService")
    assert "VALIDATION_ONLY_PROVIDER" not in encoder.categories_[index]
    assert predictions.churn_probability.between(0, 1).all()


def test_missing_numeric_and_categorical_values_are_handled(development):
    manifest, train, validation = development
    train.X.iloc[0, train.X.columns.get_loc("MonthlyCharges")] = np.nan
    train.X.iloc[0, train.X.columns.get_loc("TechSupport")] = np.nan
    validation.X.iloc[0, validation.X.columns.get_loc("MonthlyCharges")] = np.nan
    validation.X.iloc[0, validation.X.columns.get_loc("TechSupport")] = np.nan
    pipeline = build_pipeline("logistic_standard")
    diagnostics = fit_pipeline(pipeline, train, manifest)
    metrics, predictions = evaluate_pipeline(pipeline, validation, manifest)
    assert diagnostics["converged"]
    assert np.isfinite(predictions.churn_probability).all()
    assert predictions.churn_probability.between(0, 1).all()


def test_entirely_missing_numeric_training_field_is_rejected(development):
    manifest, train, _ = development
    train.X.loc[:, "tenure"] = np.nan
    with pytest.raises(ValueError, match="entirely missing"):
        fit_pipeline(build_pipeline("logistic_standard"), train, manifest)


def test_feature_conversions_handle_missing_without_learning(modeling_frame):
    modeling_frame.loc[0, "TechSupport"] = None
    X = select_features(modeling_frame)
    assert pd.isna(X.loc[0, "TechSupport"])
    assert set(X.SeniorCitizen) == {"0", "1"}


def test_known_classification_metrics_and_zero_division():
    evidence = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert evidence["confusion_matrix"] == [[1, 1], [1, 1]]
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert evidence[key] == 0.5
    assert evidence["roc_auc"] == 0.75
    assert evidence["average_precision"] == pytest.approx(5 / 6)
    no_positives = classification_metrics([0, 0], [0.1, 0.2])
    assert no_positives["precision"] == no_positives["recall"] == no_positives["f1"] == 0
    assert no_positives["roc_auc"] is None and no_positives["average_precision"] is None


@pytest.mark.parametrize("probabilities", [[-0.1, 0.5], [0.1, 1.1], [np.nan, 0.5], [np.inf, 0.5]])
def test_invalid_probabilities_fail(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        classification_metrics([0, 1], probabilities)


def test_dummy_exposes_accuracy_without_churn_recall(development):
    manifest, train, validation = development
    pipeline = build_pipeline("dummy_most_frequent")
    fit_pipeline(pipeline, train, manifest)
    evidence, _ = evaluate_pipeline(pipeline, validation, manifest)
    assert evidence["accuracy"] == pytest.approx(float(validation.y.eq(0).mean()))
    assert evidence["recall"] == evidence["precision"] == evidence["f1"] == 0
    assert evidence["roc_auc"] == 0.5
    assert evidence["average_precision"] == pytest.approx(float(validation.y.mean()))


def test_logistic_coefficients_align_with_transformed_names(development):
    manifest, train, validation = development
    pipeline = build_pipeline("logistic_standard")
    fit_pipeline(pipeline, train, manifest)
    coefficients, metadata = coefficient_report(pipeline)
    assert coefficients.transformed_feature.tolist() == pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    np.testing.assert_allclose(coefficients.coefficient, pipeline.named_steps["classifier"].coef_[0])
    np.testing.assert_allclose(coefficients.conditional_odds_ratio, np.exp(coefficients.coefficient))
    assert metadata["intercept"] == pipeline.named_steps["classifier"].intercept_[0]
    assert set(metadata["reference_categories"]) == set(CATEGORICAL_FEATURES)
    assert coefficients.iloc[:2].source_feature.tolist() == list(NUMERIC_FEATURES)
    assert not set(pipeline.named_steps["preprocessor"].feature_names_in_) & set(EXCLUDED_FEATURES)


def test_trusted_serialization_reload_consistency_and_no_refitting(tmp_path, development):
    manifest, train, validation = development
    pipeline = build_pipeline("logistic_standard")
    fit_pipeline(pipeline, train, manifest)
    _, expected = evaluate_pipeline(pipeline, validation, manifest)
    artifact = tmp_path / "owned_pipeline.joblib"
    joblib.dump(pipeline, artifact)
    loaded = joblib.load(artifact)  # This test's own newly created trusted file.
    _, actual = evaluate_pipeline(loaded, validation, manifest)
    np.testing.assert_allclose(actual.churn_probability, expected.churn_probability, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="already fitted"):
        fit_pipeline(loaded, train, manifest)


def test_phase4_workflow_never_sends_test_ids_to_sklearn(tmp_path, modeling_frame, monkeypatch):
    source = save_cleaned_dataset(modeling_frame, tmp_path / "processed.csv")
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = create_split_manifest(modeling_frame, fingerprint)
    train_ids = set(plan["splits"]["train"]["customer_ids"])
    allowed_ids = train_ids | set(plan["splits"]["validation"]["customer_ids"])
    reserved_ids = set(plan["splits"]["test"]["customer_ids"])
    reports, models = tmp_path / "reports", tmp_path / "models"
    original_fit, original_predict = Pipeline.fit, Pipeline.predict_proba
    fit_calls, predict_calls = [], []
    def spy_fit(self, X, y=None, **kwargs):
        if isinstance(X, pd.DataFrame):
            assert set(X.index).issubset(train_ids)
            assert not set(X.index) & reserved_ids
            assert (reports / "phase4/run_config.json").is_file()
            if tuple(X.columns) == FEATURES:
                fit_calls.append(len(X))
        return original_fit(self, X, y, **kwargs)
    def spy_predict(self, X, **kwargs):
        if isinstance(X, pd.DataFrame):
            assert set(X.index).issubset(allowed_ids)
            assert not set(X.index) & reserved_ids
            predict_calls.append(len(X))
        return original_predict(self, X, **kwargs)
    monkeypatch.setattr(Pipeline, "fit", spy_fit)
    monkeypatch.setattr(Pipeline, "predict_proba", spy_predict)
    monkeypatch.setattr("churn_analytics.model_plots.create_model_figures", lambda *args: [])
    result = run_phase4(source, reports, models)
    assert len(fit_calls) == 3 and len(predict_calls) == 7
    assert result["test_evaluated"] is False
    assert hashlib.sha256(source.read_bytes()).hexdigest() == fingerprint
    metrics = json.loads((reports / "phase4/metrics.json").read_text())
    assert all("test" not in candidate for candidate in metrics["models"].values())
    predictions = pd.read_csv(reports / "phase4/validation_predictions.csv")
    assert set(predictions.customerID) == set(plan["splits"]["validation"]["customer_ids"])
    assert set(predictions.split) == {"validation"}
    audit = json.loads((reports / "phase4/model_audit.json").read_text())
    assert audit["test_predictions"] == audit["test_feature_rows_transformed"] == 0
    assert audit["reload_validation_consistency"] == "passed"
    assert (models / "logistic_regression_pipeline.joblib").is_file()
    source.write_bytes(source.read_bytes() + b"\n")  # Same records, different byte identity.
    fit_calls.clear()
    with pytest.raises(ValueError, match="fingerprint"):
        run_phase4(source, reports, models)
    assert not fit_calls


def test_model_plots_reject_test_predictions(tmp_path):
    from churn_analytics.model_plots import create_model_figures
    with pytest.raises(ValueError, match="validation predictions only"):
        create_model_figures(pd.DataFrame({"split": ["test"]}), pd.DataFrame(), tmp_path)
