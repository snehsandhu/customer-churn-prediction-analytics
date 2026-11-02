> Publication-copy note: this guide documents the historical complete local
> project. Trusted datasets/models/provenance and private audit logs are withheld
> from this source release. See [asset requirements](PUBLIC_RELEASE_ASSETS.md),
> [current verification](PUBLICATION_VERIFICATION.md) and
> [new authentic captures](screenshots/README.md). Historical development commands
> are for review; do not retrain frozen models or reopen the consumed test.

# Phase 6 interview guide — analytics application integration

Read [the application guide](PHASE6_STREAMLIT_GUIDE.md) for setup, architecture and
safe usage, and verification (`../PHASE6_VERIFICATION.md`; owner-held historical artifact) for executed evidence.

1. **What does Streamlit do?** It turns Python scripts into an interactive data
   application. Widget interactions rerun the script; state and caches make that
   model practical without a separate frontend framework.
2. **How did you organize the dashboard?** A small Home entry defines six pages.
   Presentation modules use separate data, SQL, artifact, prediction and export
   services. This keeps UI changes from altering model-development logic.
3. **What makes a useful executive KPI?** A clear business definition, denominator,
   units and scope. Our customer counts, churn proportion, charge summaries and
   tenure reuse established analytics rather than new independent formulas.
4. **How do you calculate segment churn?** Churned customers within a segment
   divided by all customers in that segment. A group's share of all churners answers
   a different question; confusing them can reverse business interpretations.
5. **How do coordinated filters work?** Each selected category set intersects with
   the others, creating one population for all page charts and KPIs. No selected
   categories means zero matching rows, not an unexpected reset to the full data.
6. **How are empty and small groups handled?** Empty rates are undefined, displayed
   as an em dash or empty cell. Groups below 30 are flagged and counts are visible;
   a high rate from a few customers is not strong evidence.
7. **Why not show a monthly churn trend?** The dataset is a snapshot, without the
   historical observation sequence needed for a time trend. Tenure bands compare
   different customers, not changes in the same population over calendar time.
8. **Why reuse SQL instead of rewriting every calculation?** Existing checked-in
   queries are already tested and reconciled against Pandas. Reuse reduces divergent
   definitions and makes the application demonstrate the same analytical evidence.
9. **How is SQLite protected?** Connections use URI mode=ro, query_only and an
   authorizer that denies writes, schema changes, ATTACH and extension loading.
   Only registered aggregate SELECT questions are exposed; each connection is closed.
10. **Why no free-form SQL box?** It would widen the application from curated
    analytics to arbitrary database access. An allowlist provides useful exploration
    while keeping execution and export scope understandable and testable.
11. **How do bound SQL parameters help?** Parameters supply values separately from
    SQL syntax. The app also validates parameter names, types and ranges; it never
    interpolates user text into query paths or statements.
12. **What is cache_data for?** Reusing deterministic data calculations while returning
    copies. We cache a public, validated analytical snapshot without customer IDs;
    the cache key incorporates the source-file identity metadata.
13. **What is cache_resource for?** Reusing a heavyweight resource such as the
    already-fitted predictor. Resources can be shared, so we avoid mutation and
    revalidate trusted model evidence on each scoring request.
14. **Why not globally cache uploaded customer data?** A global cache can outlive a
    user's interaction and mix session data. Uploads are parsed in memory, and only
    that session's output scores are kept for display/download; Reset/new upload clears them.
15. **What happens on a model cache miss?** The existing trusted artifact is loaded
    after checksum/metadata checks. A miss never trains, searches, calibrates or
    reevaluates the final test.
16. **What is serialization, and why can it be unsafe?** Joblib stores a fitted
    Python object graph. Loading pickle-based files can execute code, so the app
    accepts only the project's trusted local artifact, never an uploaded model.
    A matching checksum proves identity relative to an expectation, not trust.
17. **How is inference different from training?** Training learns parameters and
    preprocessing statistics from labeled data. Inference applies those fixed
    transformations and parameters to new user inputs without learning anything.
18. **What did F01 change for the app?** The original Pipeline omitted prior Python
    categorical normalization. The shared adapter makes numeric/string SeniorCitizen
    consistent and validates exact schema/categories before scoring. The app does
    not call the raw unnormalized Pipeline.
19. **What do you validate before prediction?** All 18 unique required features,
    complete values, strict categories, numeric finiteness/ranges, binary senior
    status and service consistency. Invalid values fail; typos are not silently repaired.
20. **Why handle service dependencies in the form?** Customers without internet or
    phone require structural category labels expected by training. Dependent widgets
    show those labels explicitly; inconsistent uploaded combinations are rejected.
21. **Why use a form and Predict button?** Editing a field should not automatically
    score every intermediate scenario. A form makes submission intentional and
    reduces repeated work. Service selectors rerun first to keep dependent controls valid.
22. **How do you protect batch inputs?** Enforce 1 MiB/500-row bounds, UTF-8 CSV,
    exact unique headers before Pandas renaming, complete values and the same
    inference validation. Results preserve order through a generated 1-based row index.
23. **Why reject IDs and targets in uploads?** They are not eligible predictors.
    Targets would invite evaluation/reuse of historical labels; identifiers create
    needless privacy and export risks. The template contains one invented scenario.
24. **What does the 50% threshold mean?** The frozen policy assigns Churn when the
    estimated probability is at least 0.50. It is unchanged from Phase 5, not a new
    profit-optimal threshold chosen while building the app.
25. **What are precision and recall?** Precision is the share of predicted churners
    who actually churned; recall is the share of actual churners detected. They
    depend on the classification threshold and do not by themselves establish ROI.
26. **How do you explain the saved confusion matrix?** It has TN=925, FP=110, FN=185,
    TP=189. False negatives are missed churn labels; false positives are unnecessary
    churn flags. Outreach cost and treatment evidence are needed to value either.
27. **Why is accuracy insufficient?** Most customers did not churn. A model can
    achieve reasonable accuracy by favoring the majority class while missing churners.
    We show recall, precision, F1, ROC-AUC and AP alongside accuracy.
28. **Why read saved test summaries rather than recompute?** The test was consumed
    once after selection was frozen. Phase 6 is integration; recalculating predictions
    or choosing from test performance would cross the permanent modeling boundary.
29. **How do you preserve reproducibility?** Retain the exact data/split/model
    artifacts, selection manifest, threshold, consumption ledger, versions and earlier
    reports. New application evidence is written separately and protected hashes
    are compared before/after.
30. **What can feature importance explain?** Global model reliance on features,
    subject to correlated variables, impurity bias and permutation assumptions.
    It is not a causal effect and does not explain why this individual will churn.
31. **How do exports avoid leaking customer data?** Only curated aggregate tables
    and scrubbed evaluation summaries are offered. IDs, raw records, test labels,
    individual final predictions, model metadata and the original database are not
    public downloads. Formula-leading text is escaped for spreadsheet safety.
32. **How did you test the application?** Synthetic service tests validate counts,
    filters, SQL permissions, artifacts, inputs and exports. Native Streamlit AppTest
    exercises six-page navigation, submit/reset, service dependencies and empty
    filters. New app checks forbid fitting/evaluation; earlier regression tests remain.
33. **How are errors presented?** Expected failures use actionable messages without
    local paths or input values. The UI catches unexpected exceptions generically,
    and the Streamlit client suppresses stack-trace details. This is not a substitute
    for a complete production security review.
34. **What would monitoring involve after real deployment?** Track input validity,
    distribution shift, data quality, latency and failures; evaluate outcomes only
    when reliable delayed labels exist under a new evaluation plan. This local app
    implements none of that production infrastructure and claims no production validation.
35. **What business story is defensible?** Month-to-month contracts and shorter
    tenure show higher observed churn in this sample. A team could test onboarding
    or contract hypotheses prospectively. Observational comparisons do not show that
    offers will retain customers, reduce churn or save revenue.
36. **What are the portability limits?** Paths derive from the project rather than
    an author's machine. The app was verified locally on macOS; serialized-model
    compatibility still requires the recorded environment, and cross-platform/load/
    mobile behavior has not been comprehensively tested.

## Two-minute presentation

“I built a six-page local Streamlit application around an existing customer churn
analytics project using the public IBM Telco example. The Executive Overview shows
7,043 customers and 26.54% observed churn. Interactive filters coordinate segment
counts, churn rates and distributions with the correct denominators. A curated
SQL explorer reuses the project's business queries through a read-only connection.

“For prediction, I integrated the frozen tuned Random Forest through the shared
inference adapter. The form validates 18 eligible features and service dependencies,
then applies the unchanged 50% threshold. A separate bounded CSV workflow processes
uploads in memory and preserves row order without saving training data. I separated
public analytics caching from model-resource caching and never globally cache uploads.

“The performance page reads saved validation and final summaries. The selected
model's recorded test AP is 0.663154 and ROC-AUC is 0.843038, but I never reevaluated
that consumed test set during application development. Protected data, SQL, splits,
models, selection and the consumption ledger remain unchanged. Downloads are
aggregate summaries rather than customer records or test predictions.

“I tested validation, metadata drift, SQL write refusal, safe exports and real
Streamlit interactions, while retaining the earlier regression suite. The app is a
local portfolio demonstration, not a claim of production deployment, fairness,
causal retention benefits or ROI. My focus was to make the existing analytical
work understandable, reproducible and safe to demonstrate.”
