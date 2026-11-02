-- SQLite: snapshot KPIs. Median averages the middle one/two ordered observations.
-- Integer division in the position formula is intentional; percentages use 100.0.
WITH ordered_charges AS (
    SELECT MonthlyCharges,
           ROW_NUMBER() OVER (ORDER BY MonthlyCharges, customerID) AS position,
           COUNT(*) OVER () AS observations
    FROM customer_snapshot
)
SELECT COUNT(*) AS total_customers,
       COALESCE(SUM(ChurnFlag), 0) AS churn_count,
       COALESCE(SUM(CASE WHEN ChurnFlag = 0 THEN 1 ELSE 0 END), 0) AS retained_count,
       100.0 * SUM(ChurnFlag) / NULLIF(COUNT(*), 0) AS churn_pct,
       100.0 * SUM(CASE WHEN ChurnFlag = 0 THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0) AS retention_pct,
       AVG(MonthlyCharges) AS mean_monthly_charges,
       (SELECT AVG(MonthlyCharges) FROM ordered_charges
        WHERE position IN ((observations + 1) / 2, (observations + 2) / 2)) AS median_monthly_charges,
       AVG(tenure) AS mean_tenure_months,
       COALESCE(SUM(CASE WHEN TotalCharges IS NULL THEN 1 ELSE 0 END), 0) AS missing_total_charges,
       COALESCE(SUM(CASE WHEN tenure = 0 THEN 1 ELSE 0 END), 0) AS zero_tenure_customers
FROM customer_snapshot;
