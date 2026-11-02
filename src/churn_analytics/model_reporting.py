"""Calculated Phase 4 report; all model evidence is training/validation only."""
from __future__ import annotations

import pandas as pd
from .sql_reporting import result_markdown


def metric_table(metrics: dict, role: str = "validation") -> pd.DataFrame:
    """Present candidate metrics on one explicitly named development partition."""
    if role not in {"train", "validation"}:
        raise ValueError("Test metrics are prohibited in Phase 4")
    return pd.DataFrame([{"model": name, **{key: evidence[role][key] for key in
        ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]}}
        for name, evidence in metrics["models"].items()])


def logistic_report(config: dict, manifest: dict, metrics: dict, policy: pd.DataFrame,
                    coefficients: pd.DataFrame, coefficient_info: dict) -> str:
    """Explain actual validation results without any final-test performance claim."""
    selected = metrics["selected_model"]
    evidence = metrics["models"][selected]["validation"]
    tn, fp = evidence["confusion_matrix"][0]
    fn, tp = evidence["confusion_matrix"][1]
    split_table = pd.DataFrame([{"split": role, "customers": info["size"],
        "class_0": info["class_counts"]["0"], "class_1": info["class_counts"]["1"],
        "churn_pct": 100 * info["churn_proportion"],
        "use": "Reserved: metadata only" if role == "test" else "Fitting" if role == "train" else "Development evaluation"}
        for role, info in manifest["splits"].items()])
    rows = [
        "# Phase 4 — Logistic Regression: exploratory validation report",
        "## Executive summary",
        f'The training-only fitted candidate **{selected}** was selected by the predeclared rule: highest **validation average precision**, ties favor standard Logistic Regression. At threshold **0.50**, validation accuracy is **{evidence["accuracy"]:.4f}**, precision **{evidence["precision"]:.4f}**, recall **{evidence["recall"]:.4f}**, F1 **{evidence["f1"]:.4f}**, ROC-AUC **{evidence["roc_auc"]:.4f}**, and average precision **{evidence["average_precision"]:.4f}**. These are development results, not final-test or production performance. The test partition remains reserved for Phase 5.',
        "## Prediction objective and dataset",
        f'Predict the existing ChurnFlag (No=0, Yes=1) from plausible customer/service attributes in the unchanged Phase 2 dataset ({manifest["total_customers"]:,} records). Churn validates the mapping only. This snapshot has no documented feature cutoff or future prediction horizon, so the exercise is exploratory classification of recorded labels rather than a validated prospective churn-risk service.',
        "## Feature eligibility policy", result_markdown(policy),
        "The primary baseline excludes TotalCharges and its missingness flag because pre-outcome availability is unverified. No charge sensitivity model is fitted. Tenure/charge groups are omitted when underlying numeric values are included. Demographics remain recorded attributes, not grounds for causal or exclusionary action; suitability and subgroup behavior require further assessment. Even included feature timing is a plausible assumption, not established by the source.",
        "## Reproducible splits", result_markdown(split_table),
        f'Dataset SHA-256: `{manifest["dataset_sha256"]}`. Customer IDs are sorted before two stratified splits (80/20, then 75/25 of development), random_state=42 each. Every customer is assigned once; splits are disjoint. Test class counts are administrative split metadata only. No test features are transformed or used for fitting, prediction, scoring or threshold choice. Prior Phase 2 descriptive EDA covered the full dataset, so this is not an entirely unseen holdout.',
        "## Preprocessing and model configuration",
        "A single sklearn Pipeline contains a ColumnTransformer and classifier. Numeric tenure/MonthlyCharges use training-fitted median imputation then StandardScaler. Categorical fields (including SeniorCitizen as a label) use a constant __MISSING__ token and OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False). The first training-observed category is the reference. No vocabulary, median or scale is fitted on validation/test rows.",
        "Unseen categories encode as all zeros, sharing a representation with the dropped reference. A missing token absent from training is likewise unseen. This permits inference but cannot estimate a new category's independent effect. Reference dropping also makes penalized estimates reference-dependent. Structural service correlations remain. Eligible numeric fields have no missing values in the verified source, but the pipeline's train-only missing-value handling is tested synthetically.",
        "Logistic Regression uses lbfgs, L2 regularization, C=1, max_iter=2000, tol=1e-6 and seed 42. Installed sklearn 1.9 deprecates penalty/n_jobs arguments, which are omitted. One standard and one class_weight='balanced' model are compared; no hyperparameter or threshold search is performed. Class weights use training class counts only. Convergence warnings fail the workflow, and fitted n_iter_ is recorded.",
        "## Dummy baseline and validation comparison", result_markdown(metric_table(metrics)),
        "DummyClassifier(strategy='most_frequent') always predicts the training majority class. Its accuracy can look respectable while precision/recall/F1 for churn are zero. Both Logistic Regression candidates are evaluated on exactly the same validation IDs. Class weighting changes the training loss emphasis; threshold-0.50 recall/precision and probability calibration can change. It does not guarantee better average precision or calibrated real-world probabilities.",
        "Training diagnostics (not final evaluation):", result_markdown(metric_table(metrics, "train")),
        "## Confusion matrix and metric definitions",
        f'Selected-model validation counts: **TN={tn}, FP={fp}, FN={fn}, TP={tp}**. Rows in the saved confusion matrix are actual [No,Yes]; columns are predicted [No,Yes]. True positives identify churn-labeled customers; true negatives identify No labels. False positives flag No-labeled customers; false negatives miss churn labels. Threshold rule is probability >=0.50.',
        "Accuracy counts all correct classifications. Precision asks what share of flagged customers churned; recall asks what share of churned customers were flagged. F1 balances precision/recall. False negatives can miss potential outreach opportunities, while false positives can create unnecessary contact or cost. No monetary cost matrix, savings or operationally optimal threshold is invented.",
        "## Ranking metrics and probabilities",
        "ROC-AUC assesses positive-versus-negative ranking across thresholds; it is not accuracy at 0.50. Precision–recall curves focus on the positive class. Reported average precision is the non-interpolated recall-weighted precision summary, not trapezoidal PR area. The validation prevalence is the reference level for AP. Probability distributions, min/max/quantiles and class supports are saved for every candidate. Finite probabilities in [0,1] are verified; their calibration is not established.",
        "## Coefficients and conditional odds ratios",
        f'Intercept: **{coefficient_info["intercept"]:.6f}**. Transformed feature names are aligned directly with fitted coefficients; numeric training scales and categorical reference levels are saved in `phase4/coefficient_metadata.json`.',
        "Largest positive coefficients:", result_markdown(coefficients.nlargest(10, "coefficient")),
        "Largest negative coefficients:", result_markdown(coefficients.nsmallest(10, "coefficient")),
        "The numeric coefficient is the change in conditional log-odds for one training standard deviation, holding represented features fixed. exp(coefficient) is the corresponding fitted odds multiplier; it is not a probability ratio. Categorical multipliers compare with the saved reference category. Regularization shrinks estimates, and correlated/structurally dependent attributes make independent changes difficult to interpret. The intercept's all-reference/mean-numeric profile may not describe a feasible customer. These are fitted associations, not causal feature importance or intervention effects.",
        "## Artifacts and reproducibility",
        f'The selected training-fitted Pipeline is saved to `{config["model_filename"]}` with preprocessing and classifier together. It is not refitted on train+validation or the full dataset. Configuration, exact versions, all split IDs, input fingerprint, training provenance, metrics and validation-only predictions are recorded. Reload predictions are checked on the same validation IDs. Only load joblib files you created and trust: pickle-based loading can execute code, and cross-version portability is not guaranteed.',
        "## Business implications, limitations and Phase 5 recommendations",
        "The model can support discussion of ranking and outreach trade-offs, but does not demonstrate that outreach prevents churn or saves money. There is no verified prediction timestamp/horizon, causal identification, deployment validation or retention ROI. One validation split supports limited development comparison; candidate selection can overfit that validation split. Prior full-data EDA and snapshot selection limit claims about independence/generalization. Unseen categories, missingness, subgroup behavior and population drift remain concerns.",
        "Phase 5 may compare its explicitly authorized model candidates using training/validation data and should keep final-test evaluation separate from model/threshold choices. Final test access requires new authorization. Obtain genuinely unseen, timestamped data for prospective confirmation. Phase 4 makes no test predictions, trains no Random Forest and builds no Streamlit app.",
        "Methods: [sklearn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [encoding behavior](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).",
    ]
    return "\n\n".join(rows) + "\n"
