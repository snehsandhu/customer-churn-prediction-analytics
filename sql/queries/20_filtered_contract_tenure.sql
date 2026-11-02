-- SQLite: parameterized onboarding/contract-segment summary for query-plan review.
SELECT COUNT(*) AS customer_count, COALESCE(SUM(ChurnFlag), 0) AS churn_count,
       100.0 * SUM(ChurnFlag) / NULLIF(COUNT(*), 0) AS churn_pct,
       AVG(MonthlyCharges) AS mean_monthly_charges
FROM customer_snapshot
WHERE Contract = :contract AND tenure BETWEEN :min_tenure AND :max_tenure;
