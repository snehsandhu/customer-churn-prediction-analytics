-- SQLite: analyst-defined minimum n; NOT an optimized model/risk threshold.
WITH segments AS (
    SELECT Contract, InternetService, COUNT(*) AS customer_count,
           SUM(ChurnFlag) AS churn_count, 100.0 * AVG(ChurnFlag) AS churn_pct
    FROM customer_snapshot
    GROUP BY Contract, InternetService
    HAVING COUNT(*) >= :min_customers
)
SELECT *, churn_pct - (SELECT 100.0 * AVG(ChurnFlag) FROM customer_snapshot) AS gap_from_overall_pp
FROM segments
WHERE churn_pct > (SELECT 100.0 * AVG(ChurnFlag) FROM customer_snapshot)
ORDER BY churn_pct DESC, customer_count DESC, Contract, InternetService;
