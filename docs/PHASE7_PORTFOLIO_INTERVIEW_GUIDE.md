> Publication-copy note: this guide documents the historical complete local
> project. Trusted datasets/models/provenance and private audit logs are withheld
> from this source release. See [asset requirements](PUBLIC_RELEASE_ASSETS.md),
> [current verification](PUBLICATION_VERIFICATION.md) and
> [new authentic captures](screenshots/README.md). Historical development commands
> are for review; do not retrain frozen models or reopen the consumed test.

# Final portfolio interview guide

The project is a local, retrospective IBM Telco example application. All figures
below come from protected source/evidence. No public repository, cloud deployment,
causal retention effect, operational customer validation or fairness guarantee is claimed.

## 55 questions with beginner-friendly answers

1. **What business problem does this project address?** It explores which customer
   groups have higher observed churn and applies a frozen classifier to user-entered
   scenarios. The goal is analytical prioritization, not proof that an action prevents churn.
2. **What does one dataset row represent?** One customer in a public example
   snapshot, with service, billing, demographic and churn-label information. It is
   not a time series of that customer's history.
3. **What are the dataset dimensions?** The original has 7,043 rows and 21 columns.
   The analytical derivative keeps every row and has 25 columns, adding four documented fields.
4. **What is the target?** ChurnFlag maps No to 0 and Yes to 1. It is validated
   against the original Churn column and kept separate from predictors.
5. **What is the class distribution?** 1,869 churned and 5,174 did not churn,
   approximately 26.54% churn. A majority-class guess can look accurate while missing all churners.
6. **Why preserve raw bytes separately?** It supports provenance, repeatability
   and revisiting cleaning decisions. SHA-256 checks detect changes without altering the source.
7. **What did F02 fix?** Generic validation warned about noncanonical row counts,
   which was too permissive for downloads. Canonical acquisition now requires the
   documented 7,043×21 structure plus schema/ID/value checks; small intentional fixtures remain supported.
8. **Is a matching shape proof of authenticity?** No. Structural validation can
   accept a different file meeting the same rules. The protected original checksum
   establishes local identity, not universal authentication of every IBM source value.
9. **How can a column have no Pandas nulls but missing information?** Whitespace
   strings are values, not conventional nulls. The source had 11 blank TotalCharges
   entries, all at zero tenure, which needed explicit investigation.
10. **Why not fill those charges with zero?** Zero tenure suggests a plausible billing
    explanation but does not prove zero charges. The cleaning policy retains missingness
    with a flag and documents alternatives and observed-value coverage.
11. **What are missing versus invalid values?** Missing means information is absent;
    invalid means a supplied value violates a type, domain, range or consistency rule.
    Do not turn malformed values into valid categories by guessing.
12. **How are duplicates checked?** Check both customer-ID uniqueness and full-row
    duplicates. An identifier must be complete and unique even if other customer attributes match.
13. **What makes cleaning reproducible?** Explicit transformations, validation and
    deterministic derived fields, without editing raw data. Repeated cleaning gives
    the same analytical values rather than accumulating transformations.
14. **What is univariate EDA?** Summaries/distributions of one field, such as tenure
    median, quartiles or a histogram. It identifies coverage and shape before relating fields.
15. **What are bivariate and multivariate EDA?** Bivariate analysis relates two
    variables, such as contract and churn. Multivariate views examine combinations,
    such as contract/tenure groups, with sample sizes visible.
16. **Why are denominators so important?** Churn within a contract divides churned
    customers in that contract by all customers in the contract. Share of all churners
    uses a different denominator and answers a different business question.
17. **What does a Wilson interval tell you?** It describes uncertainty around a
    binomial proportion under assumptions. It does not prove the illustrative dataset
    is a random sample of a real customer population.
18. **How were categorical associations studied?** The EDA includes contingency-table
    chi-square summaries, effect-size/context checks and documented sparse-group limits.
    Multiple comparisons are handled as an exploratory family; statistical association is not causation.
19. **Why not encode arbitrary categories as 1, 2, 3 for correlation?** That invents
    an ordering and distance. Numerical correlations should use meaningful quantities;
    categorical relations need appropriate group/contingency summaries.
20. **What is a useful business insight here?** Month-to-month customers have 42.7%
    observed churn versus 2.8% for two-year contracts, with group counts shown. This
    motivates investigating contract/onboarding hypotheses, not claiming offers will work.
21. **Why use SQLite?** It is a lightweight local relational database built into
    Python, sufficient for this portfolio. There is no need for cloud database infrastructure.
22. **How is the relational data organized?** A constrained customer snapshot links
    to contract/payment/group lookups with primary/foreign keys. It organizes one
    source rather than fabricating transaction/customer-history tables.
23. **What is an INNER versus LEFT JOIN?** INNER retains matched rows; LEFT retains
    all left-side rows and fills absent matches with NULL. Counting matched IDs matters
    when preserving empty dictionary categories.
24. **How does SQL NULL differ from zero?** NULL is missing/unknown, while zero is
    a known numeric value. AVG/COUNT(column) use observed values, so missing coverage must be reported.
25. **What are window functions?** They calculate values across related ordered rows
    without collapsing them like GROUP BY. This project uses rankings, adjacent-band
    comparisons and quartile/cumulative summaries.
26. **What do RANK, DENSE_RANK and ROW_NUMBER mean?** RANK shares positions for ties
    and leaves gaps; DENSE_RANK shares positions without gaps; ROW_NUMBER assigns
    unique sequence positions. Define tie handling deliberately.
27. **Does LAG across tenure bands show a trend?** No. It compares adjacent
    cross-sectional groups, not the same customers over calendar time.
28. **How are SQL and Pandas reconciled?** Counts/rates are compared using existing
    definitions and exact counts or documented floating tolerances. Fresh Phase 7
    integration passed all 97 checks; disagreement must be investigated, not rounded away.
29. **How is dashboard SQL safe?** Only nine aggregate names are exposed, parameters
    are validated/bound, and readonly/query_only/authorizer controls deny writes,
    ATTACH and unsafe operations. There is no free-form SQL textbox.
30. **What does Logistic Regression estimate?** A linear score in feature space is
    mapped to a class probability through the logistic function. Regularized
    coefficients are conditional associations, not causal effects.
31. **Why one-hot encode categories?** It represents category membership without
    invented ordinal distances. LR drops a reference level; RF retains all indicators,
    while both use the same 18 eligible source features.
32. **Why scale numerical inputs for LR but not RF?** Scaling helps comparable
    regularization/numerical behavior in LR. Tree split order is not changed by simple
    monotonic unit scaling; RF retains training-only imputation without a scaler.
33. **What is data leakage?** Information unavailable at legitimate prediction time
    influences development/evaluation. Examples include target columns, post-outcome
    values, full-data learned preprocessing or choosing from final-test results.
34. **Why exclude customerID and TotalCharges?** ID is identity metadata rather than
    a predictor. TotalCharges has undocumented pre-outcome availability; the primary
    comparison preserves the earlier exclusion rather than choosing by test scores.
35. **How were partitions preserved?** Original seed-42 membership has 4,225 train,
    1,409 validation and 1,409 test customers. Later phases load exact IDs and hashes,
    not regenerate or reshuffle the split.
36. **What is a Random Forest?** It combines randomized decision trees trained on
    bootstrap samples. Averaging probabilities can reduce variance while capturing
    nonlinear patterns/interactions, but does not remove bias or leakage.
37. **What are Gini impurity and information gain?** Gini is `1-sum(p_k²)` within a
    node; a split improves class purity. Information gain compares parent uncertainty
    with weighted child uncertainty; entropy is another uncertainty measure.
38. **What does bootstrapping/feature randomness achieve?** Sampling training rows
    with replacement and considering random feature subsets makes trees less correlated.
    More trees cannot guarantee improvement or fix an unsuitable dataset.
39. **How was hyperparameter tuning done?** Historical Phase 5 used eight sampled
    RF configurations and three shuffled stratified training-only folds, AP scoring,
    seed 42 and one worker. Complete preprocessing stayed inside the searched Pipeline.
40. **Why StratifiedKFold?** It roughly maintains minority-class representation in
    each fold. All folds must remain inside original training; validation/test data
    must not teach fold imputers or category vocabularies.
41. **How was class imbalance handled?** Fixed unweighted and balanced_subsample
    forests were compared and weight was part of the controlled search. No SMOTE or
    fabricated operational customers were introduced.
42. **What are precision, recall and F1?** Precision is TP/(TP+FP); recall is TP/(TP+FN).
    F1 is their harmonic mean. They depend on the threshold and do not encode actual outreach economics.
43. **What do ROC-AUC and AP measure?** They summarize probability ranking across
    thresholds, with AP focused on precision-recall ordering. AP is not trapezoidal
    PR area and must use scores/probabilities rather than only hard predictions.
44. **How was the final model selected?** Highest unrounded validation AP, with a
    predeclared tie preference for simpler candidates. Tuned RF AP was 0.624140 versus
    LR 0.606301; a small difference on one sample is not proof of general superiority.
45. **Which RF configuration won?** 150 trees, depth 6, leaf minimum 2, split minimum
    2, max_features=0.5, unweighted, seed 42. It remains fitted on original train only;
    no train+validation refit followed selection.
46. **What is calibration?** Agreement between predicted probabilities and observed
    proportions over groups. Brier/log loss and fixed-bin reliability diagnostics
    were assessed; no calibrator was fitted to validation labels or chosen from test scores.
47. **Why keep threshold 0.50?** It is the frozen common policy. Descriptive validation
    trade-offs do not establish a profit optimum without costs, capacity and treatment-response evidence.
48. **What are impurity and permutation importance?** Impurity credits training
    split reductions; permutation measures validation AP decrease from shuffled original
    columns. Correlation, high-cardinality bias and synthetic combinations limit interpretation.
49. **How was final testing controlled?** A manifest froze model/code/inputs/threshold
    before a durable claim, then the selected model scored 1,409 test customers once.
    The ledger is consumed and later phases only read saved aggregate evidence.
50. **What are the final results?** Saved test accuracy 0.790632, precision 0.632107,
    recall 0.505348, F1 0.561664, ROC-AUC 0.843038 and AP 0.663154. The matrix is
    TN=925, FP=110, FN=185, TP=189; Phase 7 did not reproduce these by evaluating again.
51. **What did F01 teach about model serialization?** Pre-Pipeline Python normalization
    can be omitted from a saved object. The trusted adapter shares categorical conversion,
    including numeric/string SeniorCitizen, and strictly checks actual scoring inputs.
52. **How does the Streamlit architecture help?** Thin page entrypoints use separate
    analytics, SQL, artifacts, inference, cache and export services. Forms score only
    on submission, service dependencies are validated, and errors avoid local paths/stack traces.
53. **How do caching and batch validation work?** Public analytics are cached as
    data copies and the trusted fixed predictor as a resource. Uploads are not globally
    cached; UTF-8 CSVs are bounded to 1 MiB/500 rows with exact unique headers and preserved row order.
54. **What did final QA actually correct?** Numerical histogram categories had been
    sorted as strings; they now preserve numeric interval order. Narrow-viewport KPI
    text/header clearance was improved; final native responsive evidence remains incomplete.
55. **Is the project deployment/GitHub-ready?** It is a locally demonstrable portfolio
    with tests/docs/private review packaging. Public release still needs owner approval,
    licensing/privacy/artifact planning and a compatible fresh hosting build; no repo
    push, cloud deployment or production security validation occurred.

## 30-second explanation

“I built a local customer churn analytics and prediction application using the
public IBM Telco example. It combines reproducible cleaning/EDA, reconciled SQLite
analytics and a frozen Random Forest behind a six-page Streamlit dashboard. The
saved final test AP is 0.663154, evaluated once before the test was permanently
closed. Final QA preserved the data/model artifacts, corrected chart/layout issues
and passed 230 tests. It is a portfolio demonstration, not a production or causal
retention-success claim.”

## Two-minute explanation

“I started with a 7,043-customer public snapshot and treated data quality as part of
the analysis. Eleven whitespace-only TotalCharges values occurred at zero tenure;
I retained missingness instead of assuming zero. Shared Pandas functions define
KPIs and segment denominators, and the SQLite implementation is independently
reconciled. The dashboard provides an executive view, coordinated filters and a
curated readonly SQL explorer.

“For modeling, the original customer membership and 18-feature policy were preserved.
Preprocessing was fitted only within training, including CV folds. Historical
Phase 5 compared fixed/class-weighted forests and an eight-candidate RF search
against the preserved Logistic Regression. Validation AP selected a 150-tree,
depth-six forest at the unchanged 0.50 threshold: AP 0.624140 versus LR 0.606301.

“The model, code and protocol were frozen before one final test evaluation. Its
recorded AP is 0.663154 and ROC-AUC 0.843038, with recall 0.505348 at the chosen
threshold. That test is consumed. The application loads a checksum-verified trusted
artifact through the shared normalization adapter and scores only submitted scenarios
or bounded, feature-only uploads. It never retrains or reevaluates the test.

“Final QA passed 230 tests, checked all six pages with AppTest and all nine SQL
questions, and corrected numerical bin sorting and narrow-layout KPI text. Browser
capture was partial, so I document the missing final responsive evidence. I also
prepared deployment/distribution guidance without publishing. The key limitations
are snapshot timing, prior full-data EDA, validation reuse, demographic suitability
and dataset shift; I claim no causal churn reduction or ROI.”

## Five-minute technical walkthrough

Use the [demo guide](PHASE7_DEMO_GUIDE.md). Spend 45 seconds on KPIs and data-quality
policy, 45 seconds on filters/SQL denominators, one minute on invented form inputs,
45 seconds on safe batch parsing, one minute on model selection/saved performance,
and 45 seconds on integrity, tests and limitations. Mention the screenshot/browser
coverage distinction rather than presenting AppTest as a full device-browser test.

## Resume-ready description

- Built a six-page local Streamlit churn analytics/prediction portfolio using Python,
  Pandas, SQLite and scikit-learn on 7,043 public IBM example records.
- Reconciled 97 SQL/Pandas checks and integrated a frozen Random Forest with strict
  inference validation, bounded in-memory batch scoring and aggregate exports.
- Preserved train/validation/test provenance and one-time test policy; recorded
  final test AP 0.663154/ROC-AUC 0.843038 and completed 230 passing regression tests.

## Proposed GitHub repository description

“IBM Telco churn analytics portfolio: reproducible Pandas/SQLite analysis, guarded
LR/RF methodology and a local Streamlit dashboard with trusted frozen-model inference.”

This is proposed wording, not an existing public repository or a deployment claim.

## Likely follow-ups

- Why not call the probability a real-world risk guarantee? Discuss calibration,
  snapshot timing, shift and the lack of prospective validation (Q46/Q50).
- How do you know the test did not influence Phase 7? Show protected hashes,
  consumed ledger, guarded synthetic QA and saved-summary-only pages (Q49).
- What would you improve next? A separately approved prospective evaluation/fairness/
  monitoring plan and complete browser/platform review, without reusing this test.
- Why not publish the ZIP directly? Explain owner license, upstream data terms,
  provenance IDs/local historical paths and deliberate artifact distribution (Q55).
- What went wrong during development? Explain the documented earlier instrumentation
  issue, F01/F02 and the final chart-order/layout fixes with evidence, not invented outcomes.
