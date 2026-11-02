-- SQLite: churn proportion within each observed InternetService category.
SELECT InternetService, COUNT(*) AS customer_count, SUM(ChurnFlag) AS churn_count,
       100.0 * SUM(ChurnFlag) / NULLIF(COUNT(*), 0) AS churn_pct
FROM customer_snapshot
GROUP BY InternetService
ORDER BY InternetService;
