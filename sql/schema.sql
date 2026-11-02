-- SQLite >=3.37: STRICT types plus explicit domain/relationship constraints.
-- Grain: one snapshot row per customerID. No events/history/transactions invented.
CREATE TABLE IF NOT EXISTS contract_lookup (
    contract_name TEXT PRIMARY KEY NOT NULL
        CHECK (contract_name IN ('Month-to-month', 'One year', 'Two year')),
    display_order INTEGER NOT NULL UNIQUE CHECK (display_order BETWEEN 1 AND 3)
) STRICT;

CREATE TABLE IF NOT EXISTS payment_method_lookup (
    payment_method TEXT PRIMARY KEY NOT NULL CHECK (payment_method IN (
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ))
) STRICT;

-- These two lookups store the fixed Phase 2 analytical bins, not new source data.
CREATE TABLE IF NOT EXISTS tenure_group_lookup (
    tenure_group TEXT PRIMARY KEY NOT NULL CHECK (tenure_group IN (
        '0-12 months', '13-24 months', '25-36 months', '37-48 months', '49-60 months', '61+ months'
    )),
    display_order INTEGER NOT NULL UNIQUE CHECK (display_order BETWEEN 1 AND 6)
) STRICT;

CREATE TABLE IF NOT EXISTS monthly_charge_group_lookup (
    monthly_charge_group TEXT PRIMARY KEY NOT NULL CHECK (monthly_charge_group IN ('<50', '50 to <80', '80+')),
    display_order INTEGER NOT NULL UNIQUE CHECK (display_order BETWEEN 1 AND 3)
) STRICT;

CREATE TABLE IF NOT EXISTS customer_snapshot (
    customerID TEXT PRIMARY KEY NOT NULL CHECK (
        length(trim(customerID, char(9)||char(10)||char(13)||' ')) > 0
        AND customerID = trim(customerID, char(9)||char(10)||char(13)||' ')
    ),
    gender TEXT NOT NULL CHECK (gender IN ('Female', 'Male')),
    SeniorCitizen INTEGER NOT NULL CHECK (SeniorCitizen IN (0, 1)),
    Partner TEXT NOT NULL CHECK (Partner IN ('Yes', 'No')),
    Dependents TEXT NOT NULL CHECK (Dependents IN ('Yes', 'No')),
    tenure INTEGER NOT NULL CHECK (tenure >= 0),
    PhoneService TEXT NOT NULL CHECK (PhoneService IN ('Yes', 'No')),
    MultipleLines TEXT NOT NULL CHECK (MultipleLines IN ('Yes', 'No', 'No phone service')),
    InternetService TEXT NOT NULL CHECK (InternetService IN ('DSL', 'Fiber optic', 'No')),
    OnlineSecurity TEXT NOT NULL CHECK (OnlineSecurity IN ('Yes', 'No', 'No internet service')),
    OnlineBackup TEXT NOT NULL CHECK (OnlineBackup IN ('Yes', 'No', 'No internet service')),
    DeviceProtection TEXT NOT NULL CHECK (DeviceProtection IN ('Yes', 'No', 'No internet service')),
    TechSupport TEXT NOT NULL CHECK (TechSupport IN ('Yes', 'No', 'No internet service')),
    StreamingTV TEXT NOT NULL CHECK (StreamingTV IN ('Yes', 'No', 'No internet service')),
    StreamingMovies TEXT NOT NULL CHECK (StreamingMovies IN ('Yes', 'No', 'No internet service')),
    Contract TEXT NOT NULL REFERENCES contract_lookup(contract_name),
    PaperlessBilling TEXT NOT NULL CHECK (PaperlessBilling IN ('Yes', 'No')),
    PaymentMethod TEXT NOT NULL REFERENCES payment_method_lookup(payment_method),
    MonthlyCharges REAL NOT NULL CHECK (MonthlyCharges >= 0 AND MonthlyCharges <= 1.7976931348623157e308),
    TotalCharges REAL CHECK (TotalCharges >= 0 AND TotalCharges <= 1.7976931348623157e308),
    Churn TEXT NOT NULL CHECK (Churn IN ('Yes', 'No')),
    ChurnFlag INTEGER NOT NULL CHECK (ChurnFlag IN (0, 1)),
    TotalChargesMissing INTEGER NOT NULL CHECK (TotalChargesMissing IN (0, 1)),
    TenureGroup TEXT NOT NULL REFERENCES tenure_group_lookup(tenure_group),
    MonthlyChargeGroup TEXT NOT NULL REFERENCES monthly_charge_group_lookup(monthly_charge_group),
    CHECK (ChurnFlag = CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END),
    CHECK (TotalChargesMissing = (TotalCharges IS NULL)),
    CHECK (TotalCharges IS NOT NULL OR tenure = 0),
    CHECK (TenureGroup = CASE
        WHEN tenure <= 12 THEN '0-12 months' WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 36 THEN '25-36 months' WHEN tenure <= 48 THEN '37-48 months'
        WHEN tenure <= 60 THEN '49-60 months' ELSE '61+ months' END),
    CHECK (MonthlyChargeGroup = CASE WHEN MonthlyCharges < 50 THEN '<50'
        WHEN MonthlyCharges < 80 THEN '50 to <80' ELSE '80+' END),
    CHECK ((PhoneService = 'No') = (MultipleLines = 'No phone service')),
    CHECK ((InternetService = 'No') = (OnlineSecurity = 'No internet service')),
    CHECK ((InternetService = 'No') = (OnlineBackup = 'No internet service')),
    CHECK ((InternetService = 'No') = (DeviceProtection = 'No internet service')),
    CHECK ((InternetService = 'No') = (TechSupport = 'No internet service')),
    CHECK ((InternetService = 'No') = (StreamingTV = 'No internet service')),
    CHECK ((InternetService = 'No') = (StreamingMovies = 'No internet service'))
) STRICT;
