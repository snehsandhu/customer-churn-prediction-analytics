"""Render Phase 5 business documentation from saved evidence, without scoring."""
from pathlib import Path

import pandas as pd

from .phase5_data import read_json


def markdown_table(frame: pd.DataFrame) -> str:
    """Render small readable tables without an optional tabulate dependency."""
    table = frame.copy().fillna("not remeasured")
    return "| " + " | ".join(map(str, table.columns)) + " |\n| " + " | ".join(["---"]*len(table.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(f"{v:.6f}" if isinstance(v,float) else str(v) for v in row) + " |" for row in table.itertuples(index=False,name=None))


def write_business_report(root: str | Path) -> Path:
    """Summarize frozen selection and completed evidence; never evaluate a model."""
    root = Path(root)
    directory = root/"reports/phase5"
    selection = read_json(directory/"final_selection_manifest.json")
    search = read_json(directory/"search_audit.json")
    comparison = pd.read_csv(directory/"model_comparison.csv")
    metric_keys = ['model_name','average_precision','roc_auc','precision','recall','f1','accuracy','brier_score','log_loss']
    table = markdown_table(comparison[metric_keys])
    thresholds = pd.read_csv(directory/'threshold_analysis.csv')
    threshold_table = markdown_table(thresholds.loc[thresholds.model_name.eq(selection['selected_model']),
        ['threshold','precision','recall','f1','fp','fn']])
    timing = markdown_table(comparison[['model_name','training_seconds','validation_inference_median_seconds','inference_customers']])
    importance = pd.read_csv(directory/'permutation_importance.csv')
    impurity = pd.read_csv(directory/'feature_importance.csv')
    lr_ap = float(comparison.loc[comparison.model_name.eq('logistic_standard'),'average_precision'].iloc[0])
    winning_ap = selection['validation_metrics']['average_precision']
    brier = comparison.set_index('model_name').brier_score
    final_path = directory/'final_test_metrics.json'
    if final_path.exists():
        final = read_json(final_path)
        test = markdown_table(pd.DataFrame([{'metric':key,'value':final[key]} for key in
            ['customers','accuracy','precision','recall','f1','roc_auc','average_precision','brier_score','log_loss']]))
        test += f"\n\nConfusion matrix [TN, FP; FN, TP]: `{final['confusion_matrix']}`. Class support: `{final['class_support']}`.\n"
        test += "The test set is consumed. This is one final evaluation of the frozen train-fitted candidate; no post-test selection, threshold changes or refit occurred."
    else:
        test = "Not evaluated yet. Only the explicitly guarded final-evaluate command may consume the held-out test set after selection is frozen."
    report = f'''# Phase 5 — Random Forest and model comparison

## Executive summary

Selected **{selection['selected_model']}**, using unrounded validation Average
Precision. Threshold is frozen at **0.50**. The model remains fitted on the original
4,225 training customers. No train+validation refit or probability calibrator was
used. Validation and original test partitions each contain 1,409 customers.
The raw/cleaned source, persisted split, SQL database and original Logistic
Regression artifact remain unchanged. No dashboard or deployment was implemented.

## Data, feature eligibility and design

The IBM snapshot has 7,043 customers; prior evidence records 1,869 churned and
5,174 non-churned customers. The 11 missing TotalCharges remain missing in the
preserved processed file. Modeling retains exactly the existing 18 eligible
features. ID, Churn/ChurnFlag, TotalCharges, its missingness flag and redundant
analytical groups stay excluded. Snapshot timing and feature availability remain
assumptions; earlier full-population EDA means the test is not wholly unseen.

RF averages probability estimates from bootstrapped decision trees, using a random
subset of features at splits. Our fixed baselines use 200 trees, min_samples_leaf=2,
seed 42 and one worker. Numerical fields receive training median imputation without
scaling; categorical fields receive constant imputation and all one-hot levels.
LR retains its existing training-only scaling and dropped-reference encoding.
Both use identical source features, target, memberships and canonical F01 normalization.

## Training-only hyperparameter optimization

RandomizedSearchCV scored Average Precision over {search['n_iter']} sampled
configurations × {search['folds']} stratified shuffled folds, seed 42. Each fold fits
its own complete preprocessing/model Pipeline. Its final refit uses the original
train only. Search uses one worker to bound CPU use and avoid nested parallelism.
The space varies tree count, depth, split/leaf minimums, feature sampling and class
weight. Exact space and fold memberships are saved in development_protocol.json and
search_audit.json. No validation/test fields enter CV fitting or CV scoring.

Best parameters: `{search['best_params']}`.
Best training-CV AP mean: **{search['best_cv_mean']:.9f}**;
fold SD: **{search['best_cv_std']:.9f}**.
Measured search/refit duration: **{search['seconds']:.3f} seconds**.
Fold SD is not a confidence interval for future model quality.
[Scikit-learn search/refit reference](https://sklearn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).

## Validation comparison and selection

{table}

Confusion matrices and full precision values are saved in
random_forest_validation_metrics.json and model_comparison.csv. The historical
standard LR metrics were reproduced from the preserved artifact and checked at
rtol=atol=1e-12 against Phase 4 evidence. Balanced LR is a historical Phase 4
reference, not a new candidate artifact refitted in this phase.

The predeclared rule chooses the largest unrounded validation AP; exact ties prefer
LR, then fixed baseline RF, balanced RF and tuned RF. The selected AP is
{winning_ap:.9f}, compared with LR {lr_ap:.9f}; difference
{winning_ap-lr_ap:+.9f}. This is a point difference on one validation sample,
not statistical proof of general superiority or a guaranteed business improvement.
The same validation set was reused for several development diagnostics, so selection
can overfit it. Simplicity favors LR when tied; this rule was fixed before RF scores.

![Validation AP](phase5/figures/validation_model_comparison.png)
![Validation ROC](phase5/figures/validation_roc.png)
![Validation precision-recall](phase5/figures/validation_precision_recall.png)

## Imbalance and threshold trade-offs

Unweighted and balanced_subsample forests were explicitly compared. Class weights
change training emphasis rather than create customer records; no SMOTE was used.
AP and ROC-AUC assess rankings, while precision, recall, F1 and confusion matrices
depend on a threshold. Lowering the threshold often finds more churners but also
increases false positives and potential outreach burden. False negatives represent
missed churn labels, not proven lost intervention opportunities.

Selected candidate on the predeclared validation grid:

{threshold_table}

We retain 0.50 for every candidate. Without intervention costs, treatment efficacy,
capacity constraints or customer-level benefit evidence, there is no defensible
profit-maximizing threshold or ROI claim. The grid is descriptive and never optimized
against test labels.

![Threshold trade-offs](phase5/figures/validation_threshold_tradeoffs.png)

## Probability calibration

Brier score and log loss are reported above (lower is better). Calibration bins use
10 fixed equal-width probability intervals; per-bin counts and observed proportions
are saved in calibration_bins.csv; empty bins are omitted. Sparse bins are noisy.
Tuned RF validation Brier was {brier['random_forest_tuned']:.6f}, compared with LR
{brier['logistic_standard']:.6f}, unweighted fixed RF {brier['random_forest_baseline']:.6f}
and balanced fixed RF {brier['random_forest_balanced']:.6f}. The balanced forest's
curve frequently places predicted probabilities above observed proportions. These
are descriptive validation diagnostics; Brier reflects both reliability and
discrimination, so a lower value alone does not prove perfect calibration.
Ranking quality does not
imply reliable probability estimates, and neither measure guarantees real-world risk.
No calibrator was fitted to validation labels; no calibration method was selected
using test labels. A later calibration experiment would require training-only CV
and a new independent evaluation plan after this test has been consumed.

![Calibration](phase5/figures/validation_calibration.png)

## Interpretability

Tuned RF impurity importance aggregated to original features:

{markdown_table(impurity.head(8))}

Validation permutation importance (AP decrease, five repeats, seed 42):

{markdown_table(importance[['feature','ap_decrease_mean','ap_decrease_std']].head(8))}

Impurity importance favors variables with many possible splits and distributes
credit among correlated features. Permutation measures this fitted model's reliance
on a column; it can understate correlated substitutes, be negative, and create
nonphysical service combinations. Error bars are repeat SD, not population confidence
intervals. Original columns are shuffled through the full trained Pipeline, after
canonical normalization; the operational adapter still rejects inconsistent live
inputs. [Scikit-learn permutation reference](https://sklearn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html).

LR coefficients are signed conditional log-odds contrasts per training SD or category
reference; RF importances are nonnegative impurity decreases, while permutation
importance is a score decrease. These different quantities are not directly comparable
and none establishes causal influence or retention effects.

![Impurity importance](phase5/figures/rf_impurity_importance.png)
![Permutation importance](phase5/figures/rf_permutation_importance.png)

## Frozen selection and final test evaluation

Selection ID: `{selection['selection_id']}`.
Selected artifact: `{selection['artifact_path']}`.
Artifact SHA-256: `{selection['artifact_sha256']}`.
The manifest binds code, preprocessing, threshold, training provenance, dataset,
splits, comparison evidence and dependency versions before any test label access.
The claim is fsynced before test materialization. Repeated/failed claims block retries;
normal notebook reruns only read saved results. Phase 4's test_evaluated=false is
historical; Phase 5's test_consumption.json is the authoritative consumption ledger.

{test}

## Measured runtime and reproducibility

{timing}

Training timings include preprocessing; tuned duration includes the search/refit.
Inference is the median of three full validation-batch calls including normalization
and validation, excluding artifact load, on one local machine. LR training was not
remeasured because its original estimator was preserved. Cache/CPU scheduling and
hardware affect timings; they are not production throughput benchmarks.

## Limitations, business implications and Phase 6 handoff

The snapshot lacks a prospective horizon, randomized retention treatments, monetary
outcomes or demonstrated temporal/generalization stability. Demographic inputs need
suitability/fairness review. Class proportions, service plans and customer behavior
may shift, affecting calibration and ranking. Earlier full-data EDA and repeated
validation use limit clean separation of development and evaluation.

Use the results to propose a prospective, consent-appropriate outreach experiment,
with measured costs and treatment response; do not claim retained customers,
revenue savings or ROI from classification metrics alone. Future Phase 6 should
load the trusted artifact and frozen threshold through ChurnPredictor, display the
recorded limitations and never use consumed test data to tune. No Phase 6 code is
included. Final metadata is in reports/phase5/final_model_metadata.json when the
controlled test evaluation has completed.
'''
    path = root/'reports/PHASE5_MODEL_COMPARISON_REPORT.md'
    path.write_text(report, encoding='utf-8')
    return path
