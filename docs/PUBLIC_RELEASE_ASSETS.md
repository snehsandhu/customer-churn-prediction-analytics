# Trusted application assets and source-release boundary

The source candidate preserves the genuine application and every inference integrity
check. Assets are withheld because dataset-derived distribution and row-level privacy
require separate review. This checkout is useful for code/SQL review, historical
notebook reading, interview preparation and asset-independent synthetic tests.
It is **not a runnable full demo by itself**.

| Application component | Required artifact(s) | Public release policy |
|---|---|---|
| Overview / Customer Analytics | `data/processed/telco_churn_cleaned.csv`; `reports/phase4/split_manifest.json` for dataset fingerprint | Withheld; manifest contains customer memberships |
| SQL Insights | `data/database/customer_churn.db`; checked-in `sql/` queries/schema | DB withheld; SQL source included unchanged |
| Single/batch prediction | `models/phase5/random_forest_pipeline.joblib`; `reports/phase5/final_selection_manifest.json`, `final_model_metadata.json`, `test_consumption.json`, `final_test_metrics.json` | Model/metadata/ledger withheld as one trusted bundle; do not sanitize signed provenance in place |
| Model Performance | Same frozen model bundle; `reports/phase5/model_comparison.csv`, `feature_importance.csv`, `permutation_importance.csv`, `calibration_bins.csv`, `threshold_analysis.csv`; six allowlisted validation figures under `reports/phase5/figures/` | Original files withheld. Public scalar results are documentation, not runtime replacements |
| Reports / aggregate downloads | Analytical CSV and saved performance bundle as above | Withheld; no arbitrary files exposed |
| Historical ingestion / notebooks | Raw IBM CSV plus each notebook's historical reports/data | Not bundled; source-only notebooks are output-cleared and not reexecuted |
| Historical LR review | `models/logistic_regression_pipeline.joblib` and historical metadata | Withheld; not required by current RF inference page |

The raw CSV, split-assignment CSV, per-customer predictions and historical runtime
logs are **not** required by normal frozen dashboard inference. No held-out customer
features/labels should be supplied as demonstration inputs.

## Legitimate asset acquisition

No public model/DB download is approved or promised. A reviewer must obtain a
separately authorized trusted bundle from the project owner after applicable rights
and privacy review. The owner's complete local project is preserved. Any future
distribution must deliver the exact original relative paths and fingerprints through
an approved channel and retain the consumed ledger and frozen threshold 0.50.
The private audit ZIP is not automatically licensed for redistribution and is not a
public acquisition URL. Dataset acquisition code points to the original IBM source;
that alone cannot reconstruct the frozen model, database or provenance.

Never train replacement models, alter manifests/checksums, clear the ledger, or run
another final evaluation to make an asset-free checkout launch. Without authorized
assets, restrict use to source review and documented synthetic tests. The application
correctly reports missing evidence instead of bypassing these safeguards.

## Local launch with separately supplied assets

In a compatible existing environment, from the project root:

```bash
python -m streamlit run app/Home.py
```

The checked-in configuration binds localhost. This is not a cloud deployment plan.
Only the trusted original joblib model may be deserialized; never load arbitrary
downloaded or uploaded pickle/joblib files. A matching checksum authenticates bytes
only when the expected checksum and source are themselves trusted.
