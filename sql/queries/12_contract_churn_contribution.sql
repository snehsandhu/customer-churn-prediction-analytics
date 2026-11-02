-- SQLite: distinguish rate rank from volume rank and contribution to all churn.
WITH segments AS (
    SELECT Contract, COUNT(*) AS customer_count, SUM(ChurnFlag) AS churn_count,
           100.0 * AVG(ChurnFlag) AS churn_pct
    FROM customer_snapshot GROUP BY Contract
)
SELECT *, RANK() OVER (ORDER BY churn_pct DESC) AS churn_rate_rank,
       DENSE_RANK() OVER (ORDER BY churn_count DESC) AS churn_volume_rank,
       100.0 * churn_count / NULLIF(SUM(churn_count) OVER (), 0) AS share_of_all_churn_pct
FROM segments
ORDER BY churn_rate_rank, Contract;
