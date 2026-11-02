> Publication-copy note: this guide documents the historical complete local
> project. Trusted datasets/models/provenance and private audit logs are withheld
> from this source release. See [asset requirements](PUBLIC_RELEASE_ASSETS.md),
> [current verification](PUBLICATION_VERIFICATION.md) and
> [new authentic captures](screenshots/README.md). Historical development commands
> are for review; do not retrain frozen models or reopen the consumed test.

# Phase 5 interview guide

Read the actual validation/test results in
the model comparison report (`../reports/PHASE5_MODEL_COMPARISON_REPORT.md`; owner-held historical artifact) and
verification (`../PHASE5_VERIFICATION.md`; owner-held historical artifact). The explanations below describe the
implementation; example interpretations are not promises of retention benefits.

## Implementation tour and safe commands

From the existing project root, activate the existing environment. No new packages
are needed for Phase 5: scikit-learn, Pandas, Matplotlib/Seaborn, joblib and Jupyter
are already dependencies.

```bash
source .venv/bin/activate
python -m pytest -q
python -m churn_analytics.phase5 status
python -m jupyterlab notebooks/05_random_forest_model_comparison.ipynb
```

For an automated **read-only artifact notebook** run, save to a new output name:

```bash
python -m nbconvert --to notebook --execute notebooks/05_random_forest_model_comparison.ipynb --output 05_review.executed.ipynb --output-dir reports/phase5 --ExecutePreprocessor.timeout=180
```

The development/final-evaluation commands are documented for reproducibility,
not as routine rerun steps after completion:

```bash
python -m churn_analytics.phase5 develop
python -m churn_analytics.phase5 final-evaluate
```

Development refuses existing Phase 5 output directories. Final evaluation refuses
an existing claim or result. Preserve `reports/phase5/test_consumption.json` with the
artifacts; deleting it or cloning a pre-consumption project to get more test looks
violates the evaluation policy. Phase 4's original reserved-test flag is historical,
not current authorization. Phase 5's ledger is authoritative.

New modules:

- `phase5_data.py`: fingerprints, exact stored membership, ID-only coverage checks,
  selective record reading and exclusively created JSON evidence.
- `random_forest.py`: deterministic RF preprocessing/model builder, train-only
  fitting/CV, transformed and grouped impurity importance, validation permutation.
- `model_comparison.py`: shared metrics plus Brier/log loss, fixed threshold grid,
  calibration bins and the predeclared validation AP selection rule.
- `final_evaluation.py`: frozen selection/code/provenance checks, durable claim,
  one final prediction pass and consumption metadata.
- `phase5.py`: orchestrates development and exposes develop/final-evaluate/status.
- `phase5_plots.py`: validation-only ROC, precision-recall, calibration, threshold,
  comparison and importance figures.
- `phase5_reporting.py`: renders the business report from saved evidence, without
  training or scoring.
- `tests/test_phase5.py`: synthetic-only leakage, scoring, inference, search,
  preservation and final-evaluation guard regression cases.
- `notebooks/05_random_forest_model_comparison.ipynb`: concepts and executable saved-
  evidence analysis; normal reruns cannot evaluate test data.
- `reports/phase5/`: protocol, CV results/fold IDs, validation predictions/metrics,
  calibration/threshold/importance tables, selection manifest, final metrics,
  consumption ledger, final metadata, executed notebook and figures.
- `models/phase5/`: fixed, balanced and tuned forest artifacts. The original LR
  artifact stays at its Phase 4 path and is never overwritten or relabeled.

`inference.py` now accepts fitted LR or RF pipelines and an explicitly supplied
frozen threshold. Its default remains 0.50. It reuses the original normalization
and strict schema/category checks, preserving F01. F02's acquisition code is unchanged.
Use `load_churn_predictor(path, expected_sha256=..., threshold=...)` with trusted
local model metadata. A checksum verifies identity, not the safety of untrusted pickle.

## Questions and beginner-friendly answers

1. **What is a decision tree?** A sequence of feature-based splits that partitions
   customers into groups. A leaf estimates a class probability from its training
   observations; very specific leaves can overfit.
2. **What is Gini impurity?** `1 - sum(p_k**2)`. It is zero for a pure group and
   larger when labels are mixed; a tree seeks weighted reductions in impurity.
3. **What is entropy?** `-sum(p_k * log2(p_k))`, with zero-probability terms treated
   as zero. It is another measure of class uncertainty.
4. **What is information gain?** Parent impurity minus the sample-weighted
   impurity of the children. A useful split reduces uncertainty about the labels.
5. **What is a Random Forest?** An ensemble of randomized decision trees. This
   classifier averages their class-probability estimates; it can capture nonlinear
   relationships and interactions without manually specifying each interaction.
6. **What is bagging?** Bootstrap aggregating: fit learners on resampled training
   data and combine their predictions. It often reduces the variance of unstable trees.
7. **What is bootstrapping?** Sampling with replacement. A tree's sample may include
   one customer more than once and omit another; no new real customers are created.
8. **Why randomly choose features at splits?** It reduces similarity between trees.
   Averaging less-correlated errors can improve stability.
9. **What does n_estimators control?** The number of trees. More trees generally
   stabilize the ensemble at increased fit/inference/storage cost; improvement is
   not guaranteed on every dataset.
10. **How do depth and leaf size control overfitting?** Shallower trees and larger
    minimum leaves restrict very specific partitions. Too much restriction can underfit.
11. **What is the bias-variance trade-off?** Simpler models may miss real structure
    (bias), while overly flexible models fluctuate with samples (variance). Forest
    averaging and tree constraints address different parts of that balance.
12. **Does RF require feature scaling?** Split thresholds depend on feature order,
    not standard units, so this forest omits standardization. LR keeps its original
    scaling because regularization depends on predictor scales.
13. **Why use a Pipeline inside CV?** Each training fold learns its own medians and
    category vocabulary. Preprocessing the whole training partition before CV
    would let fold-validation information enter those learned transformations.
14. **Why preserve the same 18 features?** It makes comparison with LR fair and
    retains the existing eligibility policy. IDs, targets and uncertain/post-outcome
    amounts must not enter predictors.
15. **Why exclude TotalCharges?** Its pre-outcome availability is undocumented and
    it is a cumulative amount. Its eleven missing values are preserved; excluding
    it follows the baseline policy rather than a test-score-driven decision.
16. **How were the data partitions made here?** Phase 5 does not make them. It loads
    the stored seed-42 membership: 4,225 train, 1,409 validation and 1,409 test, then
    checks hashes, unique IDs, non-overlap, coverage and the assignment CSV.
17. **What is StratifiedKFold?** It approximately maintains class proportions in
    each fold. Three shuffled seed-42 folds are drawn only within original training.
18. **What does RandomizedSearchCV do?** It samples a bounded set of parameter
    configurations and compares cross-validation scores. We predeclare eight
    configurations, AP scoring and one worker, then refit the winner on train only.
19. **Why is the CV score different from the validation score?** They use different
    customers and different training sizes, and CV folds fluctuate. CV standard
    deviation is not a confidence interval for the final external score.
20. **What is class imbalance?** Churned customers are the minority, about 26.54% of
    the snapshot. Predicting everyone retained can look accurate while finding no
    churners, so accuracy alone is inadequate.
21. **How does balanced_subsample work?** It adjusts class weights for each tree's
    bootstrap sample. We compare it with unweighted fitting and include class
    weight in the controlled search; it does not establish business costs.
22. **What are precision and recall?** Precision is TP/(TP+FP); recall is TP/(TP+FN).
    A lower threshold often finds more churners but adds false positives. Costs and
    treatment outcomes are needed to choose a business outreach policy.
23. **What is F1?** The harmonic mean of precision and recall. It reflects a
    particular balance and omits true negatives and actual economic costs.
24. **What is ROC-AUC?** A ranking measure across true-positive/false-positive
    trade-offs. It should be considered alongside precision-recall metrics for
    this imbalanced task.
25. **What is Average Precision?** A recall-increment-weighted summary of precision
    across ranked examples, not trapezoidal PR area. It is our primary CV and
    validation selection metric, computed on probabilities rather than hard labels.
26. **Why compare thresholds separately from AP?** AP evaluates ranking, while a
    threshold turns probabilities into classifications. We describe the same fixed
    grid and keep 0.50; no profit objective or test-based threshold tuning is claimed.
27. **What is probability calibration?** Agreement between predicted probabilities
    and observed frequencies over groups. A good ranking does not imply reliable
    probabilities. We assess ten fixed bins, Brier score and log loss; no calibrator
    is fitted to the validation labels.
28. **What are Brier score and log loss?** Brier is mean squared probability error;
    log loss penalizes confidently wrong probabilities strongly. Lower is better
    on the same sample, but neither proves prospective calibration under shift.
29. **How would you safely fit a calibrator?** Use training-only CV or a distinct
    development calibration arrangement. Include the calibrator in the persisted
    inference workflow; never fit/evaluate it on the same validation labels as if
    they were independent, and never choose it using consumed test results.
30. **What does impurity importance measure?** Accumulated training split-purity
    improvements credited to features. We aggregate one-hot levels to original
    columns; continuous/high-cardinality fields and correlated inputs can bias it.
31. **What does permutation importance measure?** The AP drop after shuffling an
    original validation feature through the full trained pipeline. Five seed-42
    repeats show shuffle variability. It can be negative and can create inconsistent
    service combinations; the normal operational adapter remains strict.
32. **Can RF importance be compared numerically with LR coefficients?** No.
    Coefficients are signed conditional log-odds contrasts; impurity and permutation
    importance use different scales and meanings. Neither proves causation.
33. **Which model was selected and why?** Read the frozen selection manifest and
    actual report. The rule chooses highest unrounded validation AP; exact ties
    prefer LR, then fixed RF, balanced RF and tuned RF. Test scores do not choose.
34. **Why evaluate the final test once?** Repeatedly choosing from test results
    turns the test into another development set. We freeze model/threshold/code
    before a durable claim, perform one final prediction call and keep a consumed
    ledger. Failed claims also block automatic retries.
35. **What did F01 teach about serialization?** A fitted Pipeline may omit Python
    normalization performed before it. Our shared adapter converts numeric/string
    SeniorCitizen consistently and strictly validates schema/categories for LR and
    RF, while preserving the original LR artifact.
36. **How do you demonstrate reproducibility and integrity?** Persist membership,
    hashes, seeds, full parameters, CV fold IDs/results, versions, validation
    predictions, frozen code/artifact hashes, unit results and consumption evidence.
    Fixed seeds do not guarantee identical outputs across every package/platform.
37. **What is dataset shift?** Changes in customers, services, label definitions or
    relationships can degrade ranking and calibration. This snapshot does not
    demonstrate performance over future time or across providers.
38. **What business recommendation can you defend?** Test a prospective retention
    intervention with real eligibility, costs and treatment response. Model scores
    support prioritization hypotheses; classification performance alone cannot
    demonstrate saved revenue, retained customers or ROI.

## Actual results to discuss in an interview

The selected model is **tuned Random Forest**, trained on the original 4,225
training customers, with 150 trees, max_depth=6, min_samples_leaf=2,
min_samples_split=2, max_features=0.5, class_weight=None and random_state=42.
Validation AP was **0.624140**
versus LR **0.606301**. RF validation precision/recall were **0.617089 / 0.521390**,
versus LR **0.604938 / 0.524064**: a higher ranking score did not mean higher recall
at the same threshold. The balanced fixed forest raised recall to 0.625668 but
lowered precision to 0.553191 and AP to 0.578163.

The frozen 0.50-threshold model's one-time test AP/ROC-AUC were
**0.663154 / 0.843038**. Accuracy was
**0.790632**, precision **0.632107**, recall
**0.505348**, F1 **0.561664**, Brier **0.137102**.
Confusion matrix was `[[925, 110], [185, 189]]`. These test results were never
used to change the model or threshold. All 183 tests passed. The held-out set is
now consumed; the original datasets, memberships, SQL database and LR artifact
remain preserved. These figures describe this snapshot, not future guarantees.

## Two-minute interview presentation

“I extended an IBM Telco churn portfolio from validated analytics and SQL to a fair
Logistic Regression versus Random Forest comparison. I preserved the original
7,043-customer source, the cleaned file, existing model and the exact 60/20/20
customer membership. I kept the same 18 eligible predictors and reused a strict
normalization adapter so numeric and string SeniorCitizen values score consistently.

“I trained fixed unweighted and class-weighted forests, then used an eight-candidate,
three-fold stratified search within training only. The complete preprocessing
pipeline was inside CV. I compared models on validation Average Precision, supported
by ROC-AUC, confusion matrices and precision-recall trade-offs. I measured local
runtime, assessed calibration without fitting to validation labels, and explained
impurity and validation permutation importance with their limitations.

“I froze the selected model, training provenance, threshold and code before the
one-time held-out evaluation. A durable ledger prevents ordinary reruns from
reusing the test. The notebook only reads evidence, so it can be rerun safely.
My 183 passing tests check leakage boundaries, split integrity, serialization,
input normalization and repeat-evaluation refusal. The tuned forest achieved
validation AP 0.624140 versus LR 0.606301, and final test AP 0.663154 with ROC-AUC
0.843038. I distinguish retrospective associations from future risk or causal
retention benefits. There is no claimed ROI or production validation.”

Use the saved full-precision artifacts if asked to substantiate the rounded scores.

References: [Scikit-learn search/refit](https://sklearn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html),
[permutation importance](https://sklearn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html).
