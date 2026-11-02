# Supported saved-model inference after F01/F02

The supported scoring entry point is `load_churn_predictor()` in
`src/churn_analytics/inference.py`. It loads the existing trusted Logistic Regression
artifact, checks its expected SHA-256 and fitted feature/class contract, and wraps
it without fitting, changing coefficients, or writing an artifact. No new model
file is required. `ChurnPredictor` intentionally has no `fit()` method.

## Caller example

From the project root, with the existing project environment active:

```python
import json
from pathlib import Path
import pandas as pd
from churn_analytics.inference import load_churn_predictor

root = Path.cwd()  # customer-churn-analytics
audit = json.loads((root / "reports/phase4/model_audit.json").read_text())
predictor = load_churn_predictor(
    root / "models/logistic_regression_pipeline.joblib",
    expected_sha256=audit["model_sha256"],
)

# Illustrative input, not an IBM customer or a reported dataset finding.
records = pd.DataFrame([{
    "tenure": 12, "MonthlyCharges": 70.0,
    "gender": "Female", "SeniorCitizen": 1,
    "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}])
probabilities = predictor.predict_proba(records)
churn_probability = probabilities[:, 1]  # columns: P(No), P(Yes)
churn_label = predictor.predict(records)  # existing inclusive threshold >= 0.50
```

Only load a local artifact whose origin you trust. A matching checksum establishes
file identity relative to the recorded expectation; it does not make malicious
pickle/joblib content safe. Keep the known artifact, metadata, and package versions
together. The verified environment uses Python 3.13.9 and scikit-learn 1.9.1.

## Input contract

Pass a nonempty DataFrame containing exactly `modeling.FEATURES` (18 unique
columns). Order may vary; normalization restores the training order. Excluded
fields such as `customerID`, `Churn`, `ChurnFlag` and `TotalCharges` must be removed
explicitly by the caller. Preserve any customer identifier separately for joining
results; it is never used as a predictor. Inputs are copied, not modified.

- `tenure` and `MonthlyCharges` are converted with strict numeric parsing into
  float64; nonnumeric, infinite and negative values are rejected. Tenure must
  represent whole months. Actual missing values use the saved numeric imputer.
- `SeniorCitizen` accepts real numeric 0/1 (including 0.0/1.0), exact string
  `"0"`/`"1"`, or actual missing values. It normalizes the valid binary forms to
  strings. Booleans, 2, -1, fractional values, `"1.0"`, `"01"`, `" 1"` and `"yes"`
  are rejected. No category guessing or whitespace repair occurs.
- The other 15 categorical fields must contain exact supported strings from
  `cleaning.CATEGORICAL_DOMAINS`, or actual missing values. Spelling, case,
  whitespace and sentinel strings are not repaired. For example, `"Fiber optic"`
  is valid; `"Fiber Optic"` is rejected.
- Known phone/internet service contradictions are rejected when both relevant
  fields are present. This is an input contract check, not proof of real-world
  availability or correctness.
- Missing categorical values retain the saved constant-imputation behavior.
  If `__MISSING__` was absent during training, the preserved encoder may warn
  and encode that missing level as all zeros, the same representation as the
  dropped reference category. This is a limitation of the unchanged artifact;
  an explicit unknown input string is still rejected before prediction.

Outputs are finite two-column probability arrays in class order `[0, 1]` and
binary labels at the already established 0.50 threshold. No threshold tuning,
model selection or training occurs. Probabilities are not a demonstrated causal
retention effect or a validated prospective forecast.

## Canonical normalization and split guards

`modeling.select_features()` is the shared, deterministic, non-learning feature
selection/type conversion used by development preparation, guarded fitting,
guarded evaluation and this inference adapter. The adapter adds strict scoring
schema and supported-domain checks. Generic development feature selection still
permits the existing encoder's handling of unseen categories in synthetic
development tests; it is not the production scoring interface.

The saved legacy Pipeline does not itself contain this Python normalization.
Do not bypass the adapter with `joblib.load(...).predict_proba(raw_frame)`.
Direct Pipeline use is only equivalent after the documented canonical
normalization. The correction deliberately preserves the original serialized
artifact rather than refitting or replacing it.

The generic inference adapter has no split-role context. It does not authorize
development evaluation on arbitrary records. Continue using `DevelopmentSplit`
and the existing manifest/ID guards for development evaluation, and keep reserved
test records excluded. During correction verification, every real model prediction
was restricted to the existing 1,409 validation customers.

## F02 acquisition contract

`data.validate_dataset(frame, expected_rows=None)` remains the general validator
for intentional fixtures and filtered data. Its default row-count check is still
an advisory warning. `data.validate_canonical_dataset(frame)` is the stricter
acquisition policy: 7,043 rows, exactly the 21 expected column names with no
duplicates, unique/nonblank customer IDs, valid target categories and the existing
numeric-parse checks. Blank TotalCharges remains an investigation warning; it is
not silently cleaned during acquisition.

`download_dataset()` validates both cached and new files with that strict policy.
It never replaces an existing file. A failed new transfer removes its temporary
file; successful validation promotes the same-directory temporary file using an
atomic, non-overwriting hard link. If the destination appears during transfer,
promotion fails instead of overwriting it. Filesystems without hard-link support
will fail clearly and require the documented manual-download path.

This policy verifies structure and required values, not cryptographic authenticity
of every IBM source value. A different 7,043-row file satisfying the same checks
can pass. The protected project's actual raw-file checksum is recorded separately
in the correction report; it is not hardcoded as a universal source fingerprint.
The source URL and raw bytes were preserved, and no fresh download was needed.
