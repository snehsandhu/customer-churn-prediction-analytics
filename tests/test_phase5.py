"""Phase 5 regression and leakage tests: synthetic customers only, offline."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from churn_analytics.cleaning import save_cleaned_dataset
from churn_analytics.data import PROJECT_ROOT
from churn_analytics.final_evaluation import CODE_FILES, evaluate_final_once, freeze_selection, selection_identity
from churn_analytics.inference import ChurnPredictor, load_churn_predictor
from churn_analytics.model_comparison import CANDIDATE_ORDER, calibration_table, probability_metrics, select_model, threshold_table
from churn_analytics.model_splits import DevelopmentSplit, create_split_manifest, development_split, split_identity
from churn_analytics.modeling import FEATURES, build_pipeline, evaluate_pipeline, fit_pipeline
from churn_analytics.phase5 import run_phase5
from churn_analytics.phase5_data import load_development, load_membership, read_json, sha256, write_new_json
from churn_analytics.random_forest import build_random_forest, fit_random_forest, impurity_importance, tune_random_forest, validation_permutation_importance

SMALL_SPACE = {"classifier__n_estimators": [6, 8], "classifier__max_depth": [3, 4],
               "classifier__class_weight": [None, "balanced_subsample"]}


@pytest.fixture
def phase5_case(tmp_path, cleaned_customers):
    """A complete persisted Phase 4 fixture; it never reads real reserved labels."""
    root = tmp_path/"synthetic_project"
    frame = pd.concat([cleaned_customers]*16, ignore_index=True)
    frame.customerID = [f"P5-{i:04}" for i in range(len(frame))]
    source = root/"data/processed/telco_churn_cleaned.csv"
    save_cleaned_dataset(frame, source)
    raw = root/"data/raw/Telco-Customer-Churn.csv"
    raw.parent.mkdir(parents=True)
    frame.iloc[:, :21].to_csv(raw, index=False)
    manifest = create_split_manifest(frame, sha256(source))
    p4 = root/"reports/phase4"
    p4.mkdir(parents=True)
    write_new_json(p4/"split_manifest.json", manifest)
    pd.DataFrame([{"customerID": identifier, "split": role} for role, info in manifest['splits'].items()
        for identifier in info['customer_ids']]).to_csv(p4/"split_assignments.csv", index=False)
    train, validation = load_development(root, manifest)
    model = build_pipeline("logistic_standard")
    fit_pipeline(model, train, manifest)
    metrics, predictions = evaluate_pipeline(model, validation, manifest)
    predictions.to_csv(p4/"validation_predictions.csv", index=False)
    write_new_json(p4/"metrics.json", {"models": {"logistic_standard": {"validation": metrics}}})
    (root/"models").mkdir()
    joblib.dump(model, root/"models/logistic_regression_pipeline.joblib")
    write_new_json(p4/"model_audit.json", {"model_sha256": sha256(root/"models/logistic_regression_pipeline.joblib")})
    for rel in CODE_FILES:
        target = root/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT/rel, target)
    return root, manifest, train, validation


def freeze_fixture(case):
    root, manifest, train, validation = case
    (root/"reports/phase5").mkdir()
    metrics = {**probability_metrics(validation.y, np.full(len(validation.y), .25)), "split_role": "validation"}
    return freeze_selection(root, "logistic_standard", "models/logistic_regression_pipeline.joblib", metrics,
        {"threshold": .5}, [])


def test_reuses_original_membership_without_resplitting(phase5_case, monkeypatch):
    root, expected, train, validation = phase5_case
    monkeypatch.setattr('churn_analytics.model_splits.create_split_manifest', Mock(side_effect=AssertionError("No re-splitting")))
    assert load_membership(root) == expected
    original = {p: sha256(root/p) for p in ['data/processed/telco_churn_cleaned.csv', 'data/raw/Telco-Customer-Churn.csv']}
    observed = load_development(root, expected)
    assert observed[0].X.index.tolist() == train.X.index.tolist()
    assert observed[1].X.index.tolist() == validation.X.index.tolist()
    assert not (set(observed[0].X.index)|set(observed[1].X.index)) & set(expected['splits']['test']['customer_ids'])
    assert original == {p: sha256(root/p) for p in original}


@pytest.mark.parametrize('problem', ['bytes', 'assignment', 'duplicate', 'overlap', 'seed'])
def test_integrity_failure_stops_before_development(phase5_case, problem):
    root, manifest, *_ = phase5_case
    if problem == 'bytes':
        with (root/'data/processed/telco_churn_cleaned.csv').open('a') as f: f.write('\n')
    elif problem == 'assignment':
        (root/'reports/phase4/split_assignments.csv').write_text('customerID,split\nbad,train\n')
    else:
        if problem == 'duplicate': manifest['splits']['train']['customer_ids'][1] = manifest['splits']['train']['customer_ids'][0]
        if problem == 'overlap': manifest['splits']['train']['customer_ids'][0] = manifest['splits']['test']['customer_ids'][0]
        if problem == 'seed': manifest['random_state'] = 1
        manifest['manifest_id'] = split_identity(manifest)
        (root/'reports/phase4/split_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): load_membership(root)


def test_rf_configuration_class_weights_and_determinism(phase5_case):
    _, manifest, train, validation = phase5_case
    first, second = build_random_forest(n_estimators=8), build_random_forest(n_estimators=8)
    assert first.named_steps['classifier'].random_state == 42
    assert first.named_steps['classifier'].n_jobs == 1
    assert build_random_forest(class_weight='balanced_subsample').named_steps['classifier'].class_weight == 'balanced_subsample'
    for model in [first, second]: fit_random_forest(model, train, manifest)
    np.testing.assert_array_equal(ChurnPredictor(first).predict_proba(validation.X), ChurnPredictor(second).predict_proba(validation.X))
    assert 'scaler' not in first.named_steps['preprocessor'].named_transformers_['numeric'].named_steps
    with pytest.raises(ValueError, match='Refitting'): fit_random_forest(first, train, manifest)


def test_training_and_search_refuse_validation_test_and_spoofed_ids(phase5_case):
    _, manifest, train, validation = phase5_case
    model = build_random_forest(n_estimators=6)
    model.fit = Mock(side_effect=AssertionError('Invalid rows reached fit'))
    bad_index = train.X.copy()
    bad_index.index = [manifest['splits']['test']['customer_ids'][0], *bad_index.index[1:]]
    for invalid in [validation, replace(train, role='test'), replace(train, X=bad_index)]:
        with pytest.raises(ValueError): fit_random_forest(model, invalid, manifest)
        with pytest.raises(ValueError): tune_random_forest(invalid, manifest, n_iter=1, folds=2, search_space=SMALL_SPACE)
    model.fit.assert_not_called()


def test_cv_preprocessors_only_fit_original_training_folds_and_search_repeats(phase5_case, monkeypatch):
    _, manifest, train, validation = phase5_case
    observed = []
    original = ColumnTransformer.fit_transform
    def record(self, X, *args, **kwargs):
        observed.append(set(X.index))
        assert set(X.index).issubset(set(train.X.index))
        assert not set(X.index) & (set(validation.X.index)|set(manifest['splits']['test']['customer_ids']))
        return original(self, X, *args, **kwargs)
    monkeypatch.setattr(ColumnTransformer, 'fit_transform', record)
    first, evidence = tune_random_forest(train, manifest, n_iter=2, folds=3, search_space=SMALL_SPACE)
    expected = [set(f['fit_ids']) for f in evidence['cv_membership']]
    assert observed == expected*2 + [set(train.X.index)]
    for fold in evidence['cv_membership']:
        assert not set(fold['fit_ids']) & set(fold['score_ids'])
        assert set(fold['fit_ids']) | set(fold['score_ids']) == set(train.X.index)
    second, repeat = tune_random_forest(train, manifest, n_iter=2, folds=3, search_space=SMALL_SPACE)
    assert first.best_params_ == second.best_params_
    np.testing.assert_array_equal(first.cv_results_['mean_test_score'], second.cv_results_['mean_test_score'])
    imputer = first.best_estimator_.named_steps['preprocessor'].named_transformers_['numeric'].named_steps['imputer']
    np.testing.assert_allclose(imputer.statistics_, train.X[['tenure','MonthlyCharges']].median())
    assert evidence['cv_membership'] == repeat['cv_membership']


def test_rf_serialization_senior_representations_missing_and_strict_categories(phase5_case, tmp_path):
    root, manifest, train, validation = phase5_case
    model = build_random_forest(n_estimators=8)
    fit_random_forest(model, train, manifest)
    path = tmp_path/'synthetic_rf.joblib'
    joblib.dump(model, path)
    adapter = load_churn_predictor(path, expected_sha256=sha256(path))
    numeric = validation.X.copy()
    numeric.SeniorCitizen = numeric.SeniorCitizen.astype(int)
    p = adapter.predict_proba(numeric)
    np.testing.assert_array_equal(p, adapter.predict_proba(validation.X))
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    bad = numeric.copy()
    bad.Contract = 'typo'
    with pytest.raises(ValueError): adapter.predict_proba(bad)
    missing = numeric.copy()
    missing.loc[missing.index[0], ['MonthlyCharges','gender']] = [np.nan, None]
    assert np.isfinite(adapter.predict_proba(missing)).all()
    assert sha256(root/'models/logistic_regression_pipeline.joblib') == read_json(root/'reports/phase4/model_audit.json')['model_sha256']


def test_metric_threshold_and_calibration_known_values():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .4, .35, .8])
    metrics = probability_metrics(y, p)
    assert metrics['confusion_matrix'] == [[2,0],[1,1]]
    assert metrics['average_precision'] == average_precision_score(y, p)
    assert metrics['brier_score'] == pytest.approx(np.mean((p-y)**2))
    table = threshold_table(y, p).set_index('threshold')
    assert table.loc[.5, 'recall'] == .5
    assert table.loc[.3, 'recall'] == 1
    bins = calibration_table(y, p)
    assert bins.customers.sum() == 4
    assert np.average(bins.observed_proportion, weights=bins.customers) == .5
    with pytest.raises(ValueError): probability_metrics(y, [2]*4)


def test_selection_uses_only_unrounded_validation_ap_and_fixed_ties():
    values = {name: {'average_precision': .6, 'split_role': 'validation'} for name in CANDIDATE_ORDER}
    assert select_model(values) == 'logistic_standard'
    values['random_forest_tuned']['average_precision'] += 1e-9
    assert select_model(values) == 'random_forest_tuned'
    values['logistic_standard']['split_role'] = 'test'
    with pytest.raises(ValueError): select_model(values)


def test_importance_alignment_and_validation_permutation_reproducibility(phase5_case):
    _, manifest, train, validation = phase5_case
    model = build_random_forest(n_estimators=8)
    fit_random_forest(model, train, manifest)
    detail, grouped = impurity_importance(model)
    assert len(detail) == len(model.named_steps['preprocessor'].get_feature_names_out())
    assert set(grouped.source_feature) == set(FEATURES)
    assert grouped.importance.sum() == pytest.approx(1)
    first = validation_permutation_importance(model, validation, manifest, repeats=2)
    pd.testing.assert_frame_equal(first, validation_permutation_importance(model, validation, manifest, repeats=2))
    assert set(first.feature) == set(FEATURES)
    with pytest.raises(ValueError): validation_permutation_importance(model, train, manifest)


def test_no_final_access_without_frozen_selection(phase5_case, monkeypatch):
    root, *_ = phase5_case
    forbidden = Mock(side_effect=AssertionError('Unfrozen test accessed'))
    monkeypatch.setattr('churn_analytics.final_evaluation._read_selected_rows', forbidden)
    with pytest.raises(FileNotFoundError): evaluate_final_once(root)
    forbidden.assert_not_called()


@pytest.mark.parametrize('change', ['model', 'threshold', 'code', 'dataset'])
def test_freeze_rejects_drift_before_any_test_access(phase5_case, monkeypatch, change):
    root, *_ = phase5_case
    selection = freeze_fixture(phase5_case)
    if change == 'threshold':
        selection['threshold'] = .3
        (root/'reports/phase5/final_selection_manifest.json').write_text(json.dumps(selection))
    else:
        file = {'model':'models/logistic_regression_pipeline.joblib','code':CODE_FILES[0],
                'dataset':'data/processed/telco_churn_cleaned.csv'}[change]
        with (root/file).open('ab') as f: f.write(b'\n')
    forbidden = Mock(side_effect=AssertionError('Drift must block test access'))
    monkeypatch.setattr('churn_analytics.final_evaluation._read_selected_rows', forbidden)
    with pytest.raises(ValueError): evaluate_final_once(root)
    forbidden.assert_not_called()
    assert not (root/'reports/phase5/test_consumption.json').exists()


def test_final_claim_precedes_access_and_evaluates_once_without_fitting(phase5_case, monkeypatch):
    root, manifest, *_ = phase5_case
    selection = freeze_fixture(phase5_case)
    from churn_analytics import final_evaluation as final
    read = final._read_selected_rows
    calls = []
    def guarded(source, ids):
        ledger = read_json(root/'reports/phase5/test_consumption.json')
        assert ledger['status'] == 'claimed_before_test_access'
        assert ledger['selection_id'] == selection['selection_id']
        assert ids == manifest['splits']['test']['customer_ids']
        calls.append(ids)
        return read(source, ids)
    monkeypatch.setattr(final, '_read_selected_rows', guarded)
    monkeypatch.setattr('sklearn.pipeline.Pipeline.fit', Mock(side_effect=AssertionError('No final refit')))
    result = evaluate_final_once(root)
    assert result['customers'] == len(manifest['splits']['test']['customer_ids'])
    assert read_json(root/'reports/phase5/test_consumption.json')['status'] == 'completed'
    assert selection == read_json(root/'reports/phase5/final_selection_manifest.json')
    with pytest.raises(ValueError, match='repeated'): evaluate_final_once(root)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='already'): freeze_fixture_after_claim(root, result)


def freeze_fixture_after_claim(root, metrics):
    return freeze_selection(root, 'logistic_standard', 'models/logistic_regression_pipeline.joblib',
                            metrics, {'threshold':.5}, [])


def test_failed_claim_is_retained_and_blocks_retry(phase5_case, monkeypatch):
    root, *_ = phase5_case
    freeze_fixture(phase5_case)
    monkeypatch.setattr('churn_analytics.final_evaluation._read_selected_rows', Mock(side_effect=RuntimeError('Interrupted')))
    with pytest.raises(RuntimeError): evaluate_final_once(root)
    assert read_json(root/'reports/phase5/test_consumption.json')['test_consumed']
    with pytest.raises(ValueError, match='repeated'): evaluate_final_once(root)


def test_complete_synthetic_development_never_materializes_test_and_preserves_inputs(phase5_case, monkeypatch):
    root, manifest, train, validation = phase5_case
    protected = [p for p in root.rglob('*') if p.is_file()]
    before = {str(p): sha256(p) for p in protected}
    from churn_analytics import phase5_data
    original = phase5_data._read_selected_rows
    def only_development(source, ids):
        assert not set(ids) & set(manifest['splits']['test']['customer_ids'])
        return original(source, ids)
    monkeypatch.setattr(phase5_data, '_read_selected_rows', only_development)
    monkeypatch.setattr('churn_analytics.phase5_plots.create_phase5_figures', lambda *args: [])
    result = run_phase5(root, n_iter=2, folds=2, search_space=SMALL_SPACE,
                        rf_parameters={'n_estimators':8}, permutation_repeats=2)
    assert not result['test_evaluated']
    assert read_json(root/'reports/phase5/final_selection_manifest.json')['selection_id'] == result['selection_id']
    assert before == {str(p): sha256(p) for p in protected}
    predictions = pd.read_csv(root/'reports/phase5/validation_predictions.csv')
    assert set(predictions.customerID) == set(validation.X.index)
    assert not (root/'reports/phase5/test_consumption.json').exists()
    with pytest.raises(ValueError, match='already'): run_phase5(root)


def test_cv_average_precision_scorer_uses_probabilities(phase5_case):
    """Probability-ranking scores must not silently dispatch to hard predictions."""
    from sklearn.metrics import get_scorer
    _, manifest, train, validation = phase5_case
    noisy_y = train.y.copy()
    noisy_y.iloc[::3] = 1 - noisy_y.iloc[::3]
    model = build_random_forest(n_estimators=9, max_depth=1, min_samples_leaf=20)
    fit_random_forest(model, replace(train, y=noisy_y), manifest)
    expected = average_precision_score(validation.y, model.predict_proba(validation.X)[:, 1])
    observed = get_scorer('average_precision')(model, validation.X, validation.y)
    assert observed == pytest.approx(expected, abs=1e-15)
