> Publication-copy note: this guide documents the historical complete local
> project. Trusted datasets/models/provenance and private audit logs are withheld
> from this source release. See [asset requirements](PUBLIC_RELEASE_ASSETS.md),
> [current verification](PUBLICATION_VERIFICATION.md) and
> [new authentic captures](screenshots/README.md). Historical development commands
> are for review; do not retrain frozen models or reopen the consumed test.

# Recruiter demo — approximately three to five minutes

## Start safely

From the existing project root:

```bash
source .venv/bin/activate
python -m streamlit run app/Home.py
```

Use the printed localhost URL. Do not rerun phase4/phase5 development or evaluation
commands to prepare a demo. Restore trusted missing assets from the supplied bundle;
the final test is permanently consumed. No real operational customer information
is needed. Native browser/file-picker and final mobile review remain incomplete;
see the screenshot status rather than implying all browser scenarios were verified.

## 0:00–0:45 — executive story

Show Executive Overview: 7,043 customers, 1,869 churned, 5,174 no churn and 26.54%
observed churn. Charges average 64.76 dataset units and tenure 32.37 months. Explain
that these are snapshot KPIs, not historical churn trends or retained-customer ROI.
Contrast month-to-month observed churn (42.7%, n=3,875) with two-year (2.8%, n=1,695),
framing a contract/onboarding investigation rather than a causal intervention.

## 0:45–1:30 — analytics and SQL

Open Customer Analytics. Select a contract/internet group, point to the filtered
customer count and within-segment denominator, and hover for group sizes. Clear a
filter to show the deliberate no-match state, then restore categories. Inspect a
numeric distribution; bins now retain numerical order, not string sorting.

Open SQL Business Insights. Choose contract segments, then the adjacent-tenure
window question. Show SQL in the expander and explain that neighboring tenure
bands are not a calendar trend. Queries are allowlisted/readonly; downloads contain
aggregate counts and rates. All nine exposed questions and 97 SQL/Pandas checks
were executed during final QA.

## 1:30–2:30 — invented prediction scenario

Use an invented record, for example the template defaults:

| Feature group | Illustrative input |
|---|---|
| Tenure / monthly charges | 12 months / 70 dataset units |
| Phone / lines | Yes / No |
| Internet / six add-ons | DSL / No for every add-on |
| Contract / paperless | Month-to-month / Yes |
| Payment | Electronic check |
| Gender / senior | Female / No (0) |
| Partner / dependents | No / No |

Submit Predict churn. Explain the displayed estimated probability and frozen 50%
classification threshold. Do not prestate a particular probability as a guarantee.
Change phone/internet to No to demonstrate structural not-applicable fields, then
Reset. The model includes demographic attributes; suitability/fairness is unverified.
Global importance is not an explanation of this individual scenario.

## 2:30–3:15 — optional batch demonstration

Download the app's invented feature-only CSV template and make a second invented
row, changing SeniorCitizen to 1 if desired. Upload only this synthetic file. The
service accepts 1 MiB/500 rows, exact unique headers, complete valid fields and
consistent services; it rejects IDs/targets. Scores preserve row order through a
1-based index. Download the current upload's scores, then choose a new file to
clear prior session output. Do not upload historical test records. Backend/order/
limit behavior is verified; native file-picker/new-upload UI behavior remains a
manual check when browser tools are available.

## 3:15–4:15 — methodology and performance

Show the saved validation comparison: AP 0.606301 for LR, 0.589939 fixed RF,
0.578163 balanced RF and 0.624140 tuned RF. The tuned forest was selected before
test access, using training-only three-fold CV and validation AP. Threshold stayed
0.50; no calibrator or train+validation refit was used.

The saved one-time final test AP is 0.663154 and ROC-AUC 0.843038; accuracy 0.790632,
precision 0.632107, recall 0.505348 and F1 0.561664. The matrix is TN=925, FP=110,
FN=185, TP=189. These are stored retrospective results, not a fresh evaluation or
future-performance guarantee. Explain missed churners and outreach-cost uncertainty.

## 4:15–5:00 — engineering and limitations

Show Reports & Downloads and the architecture diagram. Explain checksum-verified
trusted model loading, shared F01 normalization, F02 canonical validation, readonly
SQL and separated caches. The final test suite has 230 passing tests, including
all earlier regression cases. The test ledger/model/data hashes remain unchanged.

Close with limitations: earlier full-data EDA, uncertain feature timestamps,
calibration/shift/fairness questions, partial browser/responsive evidence, no fresh
cross-platform installation or public deployment. Suggest a prospective study with
real costs/treatment response; do not claim retention savings or causal effects.

## Troubleshooting

- Missing/invalid asset: restore the trusted bundle, not training/evaluation outputs.
- sklearn mismatch: use the recorded compatible environment; do not bypass checks.
- No filtered rows: restore category selections; empty selection is intentionally empty.
- Invalid upload: use exactly 18 feature columns, no labels/IDs and valid service combinations.
- Port busy: choose `--server.port=8517` or another approved localhost port.
- Source-only checkout: obtain the approved artifact bundle; ignored assets are essential.
- Browser capture unavailable: describe the limitation; do not substitute mockups.
