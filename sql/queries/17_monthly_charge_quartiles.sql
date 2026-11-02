-- SQLite: row-count quartiles, not the fixed Phase 2 MonthlyChargeGroup bins.
-- Equal charges can fall in adjacent tiles; customerID makes ties deterministic.
WITH tiled AS (
    SELECT MonthlyCharges, ChurnFlag,
           NTILE(4) OVER (ORDER BY MonthlyCharges, customerID) AS charge_quartile
    FROM customer_snapshot
)
SELECT charge_quartile, COUNT(*) AS customer_count, SUM(ChurnFlag) AS churn_count,
       MIN(MonthlyCharges) AS minimum_monthly_charges,
       MAX(MonthlyCharges) AS maximum_monthly_charges,
       AVG(MonthlyCharges) AS mean_monthly_charges,
       100.0 * AVG(ChurnFlag) AS churn_pct
FROM tiled GROUP BY charge_quartile ORDER BY charge_quartile;
