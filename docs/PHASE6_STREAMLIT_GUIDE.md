> Publication-copy note: this guide documents the historical complete local
> project. Trusted datasets/models/provenance and private audit logs are withheld
> from this source release. See [asset requirements](PUBLIC_RELEASE_ASSETS.md),
> [current verification](PUBLICATION_VERIFICATION.md) and
> [new authentic captures](screenshots/README.md). Historical development commands
> are for review; do not retrain frozen models or reopen the consumed test.

# Phase 6 — Streamlit application guide

This local portfolio application connects the existing IBM Telco analytics, SQLite
queries and frozen Random Forest to a six-page dashboard. It demonstrates application
integration, not public deployment or a newly trained model. Phase 7 is not implemented.

## Launch and environment

Run from the existing project root, with the existing environment:

```bash
cd customer-churn-analytics
source .venv/bin/activate
python -m streamlit run app/Home.py
```

Streamlit prints the local URL, normally `http://127.0.0.1:8501`. An explicit port:

```bash
python -m streamlit run app/Home.py --server.port=8516
```

The verified environment is macOS, Python 3.13.9, Streamlit 1.66.1, Altair 6.3.0,
Pandas 2.3.3, NumPy 2.5.4, scikit-learn 1.9.1 and joblib 1.6.0. No package was
installed/upgraded during Phase 6. Plotly is absent, so interactive charts use the
already installed Altair dependency behind native Streamlit chart components.
Requirements now declare Altair directly (`>=5.4,<7`) and require Streamlit >=1.50
for the width/navigation APIs used here. Compatibility was reviewed before changing
these declarations; installed versions did not change.

For a new environment, review `requirements.txt`, the saved Phase 5 environment in
`reports/phase5/development_protocol.json` and the serialized model's exact
scikit-learn version before installation. The model loader rejects a different
scikit-learn version; broad package bounds do not prove serialized-model portability.
On an approved compatible environment, normal setup is:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

These installation commands were **not** run during Phase 6. Prefer the existing
verified environment. No internet is needed for normal analytics/scoring once the
local dependencies and project artifacts are present. The application uses system
fonts, local assets and no external data/model API.

Required existing files include the Phase 2 processed CSV, Phase 3 SQLite database
and SQL definitions, selected Phase 5 model, final selection manifest, final model
metadata, completed consumption ledger, saved aggregate evaluation JSON/CSV and
figures. They are generated project assets and may be ignored by Git; restore the
trusted supplied bundle if missing. **Do not rerun model training/evaluation to
repair a missing application asset.**

## Architecture and files

```text
app/Home.py                          navigation and common layout
app/pages/1_Executive_Overview.py     overview entry
app/pages/2_Customer_Churn_Analytics.py
app/pages/3_SQL_Business_Insights.py
app/pages/4_Customer_Churn_Prediction.py
app/pages/5_Model_Performance.py
app/pages/6_Reports_and_Downloads.py
src/churn_analytics/dashboard/
    artifacts.py    allowlisted saved evidence and model identity validation
    data.py         read-only snapshot, established KPIs, filters and segments
    sql.py          curated original SELECT queries, read-only connections
    prediction.py   input validation, bounded CSV parsing and adapter scoring
    exports.py      aggregate CSV and scrubbed public summary
    cache.py        public-data cache and trusted immutable model cache
    components.py   shared cards, charts, filters and safe page errors
    views/          six presentation modules; no training or evaluation workflow
.streamlit/config.toml               light navy/teal theme and local defaults
tests/test_dashboard_services.py     analytics, SQL, artifacts and exports
tests/test_dashboard_prediction.py   feature validation and illustrative inference
tests/test_phase6.py                 native Streamlit AppTest interactions
reports/phase6/                      new execution/integrity evidence only
PHASE6_VERIFICATION.md               actual outcomes and limitations
```

The two `__init__.py` files define the new package boundaries. Page entry scripts
remain small; services reuse the Phase 2 analytics, Phase 3 registered SQL and Phase 4/5
inference interface. All asset paths derive from the existing project root; no
machine-specific author path is embedded in application code. Launch from the root
to pick up the theme configuration. The default server binds localhost and disables
usage telemetry. Browser errors suppress stack-trace details; expected service
failures have actionable messages without local paths or uploaded values.

[Streamlit navigation reference](https://docs.streamlit.io/develop/api-reference/navigation/st.navigation).

## Navigation

**Executive Overview** shows the six established customer KPIs, snapshot outcome
counts and contract, tenure, charge-group, payment and internet churn charts. Four
statements compare computed segment rates/counts and suggest investigations. No
calendar trend or causal benefit is inferred.

**Customer Analytics** applies all nine sidebar filters together. All categories
selected means unrestricted; clearing a filter means zero matching customers.
Counts, churn rates, segment comparisons, binned numerical distributions and a
contract/tenure heatmap share the same filtered population. Empty selections show
an explicit message; groups below 30 are flagged. Missing TotalCharges is retained
and excluded only from that observed-value distribution. Histograms are aggregated
before sending chart data to the browser; there is no raw-record table.

**SQL Business Insights** offers nine approved aggregate questions, including
contract/payment/service segments, above-average groups, rankings, window functions
and charge quartiles. Explanations distinguish rate, volume and churn contribution.
Only the two relevant queries expose a bounded minimum-group-size parameter. SQL
code is viewable in an expander; results can be exported as aggregate CSV.

**Customer Prediction** accepts all 18 actual eligible inputs. Phone and internet
service selectors rerun first so dependent fields become structurally not
applicable when needed. The remaining controls live in a proper form; scoring
happens only when Predict churn is submitted. Reset clears only that session's
widget values/results. The most recent result is clearly labeled; submit again
after edits. No new training data is saved.

**Model Performance** reads stored validation summaries for all four candidates,
stored final test metrics/confusion matrix and saved curves/importance. It never
opens per-customer final test predictions or recomputes test metrics. Global
importance describes model reliance across data, not why an individual will churn.

**Reports & Downloads** offers snapshot KPIs, segment summaries, approved SQL
aggregates, saved validation comparison and a scrubbed JSON summary. The original
database, raw records, original customer IDs, historical test labels/predictions,
full model metadata and private local paths are not exposed for download. It
explains methodology and analytical limitations without fictional ROI claims.

## KPI definitions and business storytelling

The app delegates to `analytics.customer_kpis()` and `churn_by_segment()`. For each
segment, churn rate is churned customers **in that segment** divided by all
customers **in that segment**. This is different from a segment's share of all
churned customers. Empty rates are undefined, not zero. Chart percentage axes start
at zero and show denominator/count tooltips; sparse groups are exploratory.

Reference counts are 7,043 customers, 1,869 churned, 5,174 no churn, about 26.54%
observed churn, 64.76 mean monthly charges and 32.37 months mean tenure. Charges
are dataset units; no currency conversion, lost revenue, CLV or savings is invented.
The new real-data verification reconciles these with the original SQL analytics.
Recommendations remain hypotheses needing prospective treatment/cost evidence.

## Frozen model and inference contract

The application validates the selection identity, consistency of final metadata,
completed one-attempt consumption ledger, threshold 0.50, exact 18-feature schema,
fixed selected model path, model SHA-256, saved summary SHA-256 and runtime
scikit-learn version **before loading**. It accepts only the trusted local
`models/phase5/random_forest_pipeline.joblib`. A checksum detects identity changes;
it does not make untrusted pickle/joblib content safe. No uploaded model is accepted.

`load_churn_predictor()` returns the existing `ChurnPredictor`. The app calls its
validated `predict_proba()`, never a raw unnormalized Pipeline. The existing F01
normalization converts numeric and exact-string SeniorCitizen 0/1 consistently;
the F02 acquisition implementation is unchanged. The app adds complete-input and
UI-range requirements without modifying the model's general imputation contract:

- Exactly `modeling.FEATURES`, 18 unique columns; no IDs/targets/extras.
- Tenure: whole months, 0–72 inclusive.
- MonthlyCharges: finite 0–200 dataset units inclusive. This is an application
  bound; observed sample values are approximately 18–119, so outside-sample inputs
  have additional uncertainty.
- SeniorCitizen: valid numeric 0/1 or exact strings "0"/"1"; no booleans or guessed forms.
- Other fields: exact supported categories and no missing/blank required values.
- No phone => MultipleLines="No phone service". No internet => all six add-ons
  equal "No internet service". Inconsistent uploaded combinations are rejected.

At probability >=0.50, the label is Churn; otherwise No Churn. Displayed probabilities
are predictive estimates, not individual guarantees, calibrated real-world risk
promises or causal intervention explanations. The estimator, its coefficients/
trees, thresholds, source data, split and ledger are never rewritten by inference.

## Optional batch scoring

A separate expander accepts UTF-8/BOM CSV, at most **1 MiB** and **500 rows**.
CSV header duplicates are detected before Pandas can rename them. Exactly the 18
eligible headers are required; malformed rows, targets/IDs, missing values, invalid
categories/ranges and service contradictions fail before scoring. An illustrative
one-row template is invented rather than drawn from any historical/test partition.

Uploads are processed in memory without global caching or permanent dataset writes.
The output preserves order with a generated **1-based row_index**, label,
probability and frozen threshold. Only output scores live in session state; a new
upload or Reset clears them. Inputs are not echoed in output tables or logs. There
is no uploaded user identifier field, avoiding accidental exposure or formula
injection through IDs. Uploaded records must not be used to replay the consumed
historical test set. Unit tests exercise parsing/order; native file-picker behavior
is not covered by AppTest's simulated upload controls.

## SQL, caching and export safety

SQLite uses the existing `connect_database(..., readonly=True)`, plus query_only
and a restrictive authorizer. The authorizer blocks DDL/DML, ATTACH, write PRAGMAs
and extension loading. Each request closes its own connection; connections are
not shared globally between threads. Registered original query definitions and
bound validated parameter values are reused. Row-level ranking query 16 is excluded.
There is no unrestricted SQL text box or user-controlled query file path.

`st.cache_data` stores copies of validated public analytics with source identity
invalidation. `st.cache_resource` stores the trusted fitted predictor with model/
metadata identity in its cache key; evidence is reverified on each scoring request.
Cache misses never train or evaluate. User inputs/uploads/results are not globally
cached. Session state is not authentication or production privacy infrastructure.
[Streamlit resource-cache reference](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource).

Exports are generated only by curated aggregate services. Identifying/outcome-row
column names are rejected; spreadsheet formula-leading strings are escaped. This
helper is defense in depth, not a generic anonymizer for arbitrary data. Prediction
exports contain only the current uploaded input's index and model scores, distinct
from historical final-test predictions, which are never offered.

## Tests and troubleshooting

```bash
python -m pytest -q
```

The full suite retains earlier synthetic estimator-fitting tests. New Phase 6
service/UI checks forbid training and final evaluation; no existing IBM model is
fitted/refitted. AppTest covers all pages, form submit/reset, structural service
dependencies, empty filters and SQL navigation.
[AppTest reference](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).

- **Required artifact missing/invalid:** restore the trusted existing model/data/
  report bundle. Do not rerun Phase 4/5 or delete the consumption ledger.
- **Incompatible model environment:** use the recorded sklearn 1.9.1 environment.
  Do not blindly upgrade or refit to bypass the identity check.
- **No matching customers:** restore sidebar categories; an empty selection is
  deliberately empty. Undefined rates show an em dash.
- **Prediction rejected:** check exact headers, supported categories, complete
  values, app numerical ranges and structural phone/internet representations.
- **SQL unavailable:** restore the existing database and registered SQL files;
  the app never initializes/rebuilds the database.
- **Port already used:** choose another localhost port with `--server.port`.
- **Batch output disappeared:** a new upload/reset clears that session's results.
- **Browser stack traces:** expected errors are sanitized; consult the local
  developer environment/files rather than enabling public diagnostic details.

Known limits: verified locally on macOS; cross-platform, touch/mobile, concurrency,
load testing, authentication, encryption for remote hosting and production monitoring
are not established. Model snapshot/EDA/fairness/shift limitations remain. No public
deployment or Phase 7 work is included.
