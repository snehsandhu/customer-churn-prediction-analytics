-- SQLite: audit missingness by tenure status without imputing financial amounts.
SELECT CASE WHEN tenure = 0 THEN 'Zero tenure' ELSE 'Positive tenure' END AS tenure_status,
       COUNT(*) AS customer_count,
       SUM(CASE WHEN TotalCharges IS NULL THEN 1 ELSE 0 END) AS missing_total_charges,
       COUNT(TotalCharges) AS total_charges_observed,
       AVG(TotalCharges) AS mean_total_charges_observed
FROM customer_snapshot
GROUP BY CASE WHEN tenure = 0 THEN 'Zero tenure' ELSE 'Positive tenure' END
ORDER BY tenure_status;
