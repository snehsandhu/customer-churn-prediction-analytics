"""Train-only Random Forest development, bounded CV and descriptive importance."""
from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .modeling import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, guard_development_data, select_features

SEARCH_SPACE = {
    "classifier__n_estimators": [150, 250],
    "classifier__max_depth": [6, 10, None],
    "classifier__min_samples_split": [2, 10],
    "classifier__min_samples_leaf": [2, 5, 10],
    "classifier__max_features": ["sqrt", 0.5],
    "classifier__class_weight": [None, "balanced_subsample"],
}


def build_random_forest(**parameters) -> Pipeline:
    """Create an unfitted RF with median/constant imputation and full one-hot levels.

    Trees need no standardization. Keep every categorical indicator, rather than
    dropping a reference as in LR. Feature eligibility and normalization are shared.
    Single-thread execution bounds resources and avoids nested parallelism.
    """
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ], remainder="drop")
    options = dict(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=1)
    options.update(parameters)
    return Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(**options))])


def stamp_training(model, manifest: dict) -> None:
    """Attach the same training provenance used by guarded Phase 4 evaluation."""
    model._phase4_manifest_id = manifest["manifest_id"]
    model._phase4_training_ids = tuple(manifest["splits"]["train"]["customer_ids"])


def fit_random_forest(model, train, manifest: dict) -> float:
    """Guard membership before train-only fitting; return measured elapsed seconds."""
    guard_development_data(train, manifest, fitting=True)
    X = select_features(train.X)
    if X.loc[:, NUMERIC_FEATURES].isna().all().any():
        raise ValueError("Entirely missing numeric training field")
    if hasattr(model, "_phase4_manifest_id"):
        raise ValueError("Refitting an existing candidate is prohibited")
    start = perf_counter()
    model.fit(X, train.y)
    duration = perf_counter() - start
    stamp_training(model, manifest)
    return duration


def tune_random_forest(train, manifest: dict, *, n_iter: int = 8, folds: int = 3, search_space=None):
    """Fit every CV preprocessor inside an original-training-only fold.

    RandomizedSearchCV's refit uses the original training set, never train+val.
    Fold customer IDs are recorded to make the boundary independently auditable.
    """
    guard_development_data(train, manifest, fitting=True)
    X = select_features(train.X)
    if X.loc[:, NUMERIC_FEATURES].isna().all().any():
        raise ValueError("Entirely missing numeric training field")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    splits = list(cv.split(X, train.y))
    audit = [{"fold": i, "fit_ids": X.index[a].tolist(), "score_ids": X.index[b].tolist()}
             for i, (a, b) in enumerate(splits)]
    search = RandomizedSearchCV(build_random_forest(), SEARCH_SPACE if search_space is None else search_space,
        n_iter=n_iter, scoring="average_precision", cv=splits, random_state=42, n_jobs=1,
        refit=True, error_score="raise", return_train_score=True)
    start = perf_counter()
    search.fit(X, train.y)
    duration = perf_counter() - start
    stamp_training(search.best_estimator_, manifest)
    return search, {"seconds": duration, "n_iter": n_iter, "folds": folds,
                    "random_state": 42, "scoring": "average_precision", "cv_membership": audit,
                    "best_params": search.best_params_, "best_cv_mean": float(search.best_score_),
                    "best_cv_std": float(search.cv_results_["std_test_score"][search.best_index_])}


def impurity_importance(model) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align transformed importances and aggregate one-hot levels to source fields."""
    pre = model.named_steps["preprocessor"]
    names = pre.get_feature_names_out()
    values = model.named_steps["classifier"].feature_importances_
    sources = list(NUMERIC_FEATURES)
    encoder = pre.named_transformers_["categorical"].named_steps["onehot"]
    for column, levels in zip(CATEGORICAL_FEATURES, encoder.categories_):
        sources.extend([column] * len(levels))
    if not len(names) == len(values) == len(sources):
        raise ValueError("Feature importance names do not align")
    detail = pd.DataFrame({"transformed_feature": names, "source_feature": sources, "importance": values})
    grouped = detail.groupby("source_feature", as_index=False).importance.sum().sort_values("importance", ascending=False)
    return detail, grouped


def validation_permutation_importance(model, validation, manifest: dict, *, repeats: int = 5) -> pd.DataFrame:
    """Permute original columns through the full trained pipeline, validation only.

    Permutation deliberately breaks associations, including service consistency.
    Use already normalized features, not the strict operational input validator,
    which correctly rejects such synthetic combinations for actual scoring.
    """
    guard_development_data(validation, manifest)
    if validation.role != "validation" or getattr(model, "_phase4_manifest_id", None) != manifest["manifest_id"]:
        raise ValueError("Permutation requires this manifest's validation candidate")
    result = permutation_importance(model, select_features(validation.X), validation.y,
        scoring="average_precision", n_repeats=repeats, random_state=42, n_jobs=1)
    output = pd.DataFrame({"feature": FEATURES, "ap_decrease_mean": result.importances_mean,
                           "ap_decrease_std": result.importances_std})
    for i in range(repeats):
        output[f"repeat_{i+1}"] = result.importances[:, i]
    return output.sort_values("ap_decrease_mean", ascending=False)
