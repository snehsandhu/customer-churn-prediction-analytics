-- SQLite: meaningful-size service combinations, threshold explicitly parameterized.
SELECT InternetService, TechSupport, OnlineSecurity,
       COUNT(*) AS customer_count, SUM(ChurnFlag) AS churn_count,
       100.0 * SUM(ChurnFlag) / NULLIF(COUNT(*), 0) AS churn_pct
FROM customer_snapshot
GROUP BY InternetService, TechSupport, OnlineSecurity
HAVING COUNT(*) >= :min_customers
ORDER BY churn_pct DESC, customer_count DESC, InternetService, TechSupport, OnlineSecurity;
