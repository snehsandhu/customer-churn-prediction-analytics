> Publication-copy note: this guide documents the historical complete local
> project. Trusted datasets/models/provenance and private audit logs are withheld
> from this source release. See [asset requirements](PUBLIC_RELEASE_ASSETS.md),
> [current verification](PUBLICATION_VERIFICATION.md) and
> [new authentic captures](screenshots/README.md). Historical development commands
> are for review; do not retrain frozen models or reopen the consumed test.

# Attribution and distribution decision record

This project uses the public CSV from IBM's
[telco-customer-churn-on-icp4d repository](https://github.com/IBM/telco-customer-churn-on-icp4d),
[source CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).
The original bytes were preserved; the analytical derivative retains 7,043 rows,
adds four analytical fields and keeps 11 TotalCharges entries missing.

The upstream repository identifies its **code pattern** as Apache 2.0 and provides
[the license text](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/LICENSE).
An unmodified copy is retained at `docs/third_party/IBM_TELCO_APACHE_LICENSE.txt`.
No separate dataset-specific grant was identified during this review. This review
therefore does not assert that every dataset, derivative or trained model has a
separately verified unrestricted redistribution license. The owner must review
applicable terms before public redistribution. No endorsement by IBM is implied.

The upstream license is third-party context, not a license chosen for the owner's
original portfolio source. There is no newly imposed root LICENSE; owner license
selection remains a pre-publication decision.

## Distribution matrix

| Material | Private independent-review ZIP | Proposed public GitHub/release treatment |
|---|---|---|
| Original source, tests, SQL and Streamlit app | Include | Candidate source, after owner license/secret review |
| New architecture/demo/interview guides | Include | Candidate docs; do not exaggerate deployment |
| Actual QA screenshot | Include, explicitly pre-layout-fix | Label accurately; obtain final screenshots before presenting them as final UI |
| Raw/processed public IBM data and DB | Include with attribution/upstream license | Keep ignored; approve terms and a deliberate release/retrieval plan first |
| LR and three RF artifacts | Include, unchanged | Keep ignored; approve derived-asset terms and trusted distribution first |
| Selection/model metadata and consumed ledger | Include; essential integrity/demo assets | Supply with approved model bundle; provenance contains sample IDs |
| Safe aggregate comparison/final metrics/figures | Include | Review/curate public copies; keep exact original evidence |
| Per-customer final test predictions | **Exclude** | Do not publish; original remains local and unchanged |
| Per-customer validation predictions | Include only as historical private audit evidence | Keep out of public source/release by default |
| Historical verification reports/notebook outputs | Include unchanged | Review for author-machine paths and row-level sample outputs; publish curated derivatives only |
| Environments/caches/secrets/backups | Exclude | Never commit/distribute |

The private archive is self-contained for reviewing/running the current app in a
compatible environment. It is deliberately **not** a ready-to-publish repository
snapshot. It includes requested historical evidence containing local execution
paths; those are not app outputs and need a public-copy privacy review. No private
operational customer dataset is part of this project; IBM sample identifiers are
not exposed by dashboard exports.

Do not remove essential assets from the local demo to satisfy Git ignore rules.
Ignoring a file is not deleting it. A future approved release must supply the same
trusted assets and checksums; do not replace the retrieval plan with retraining or
another held-out evaluation. Original protected files remain intact locally even
where the review ZIP deliberately omits a file.

`.gitignore` now covers environments, caches, .DS_Store, local .env variants,
Streamlit secrets, key/credential files, logs, temporary files and backup archives.
It is a preventive policy, not proof that every secret can be detected. No Git
repository was initialized and no commit, remote or push was created.
