-- SQLite: compare mean/median monthly charges by observed outcome.
-- NULL cumulative charges stay NULL; AVG excludes them and coverage is explicit.
WITH ranked AS (
    SELECT Churn, MonthlyCharges, TotalCharges,
           ROW_NUMBER() OVER (PARTITION BY Churn ORDER BY MonthlyCharges, customerID) AS position,
           COUNT(*) OVER (PARTITION BY Churn) AS observations
    FROM customer_snapshot
)
SELECT Churn, COUNT(*) AS customer_count, AVG(MonthlyCharges) AS mean_monthly_charges,
       AVG(CASE WHEN position IN ((observations + 1) / 2, (observations + 2) / 2)
                THEN MonthlyCharges END) AS median_monthly_charges,
       COUNT(TotalCharges) AS total_charges_observed,
       AVG(TotalCharges) AS mean_total_charges_observed
FROM ranked GROUP BY Churn ORDER BY Churn;
