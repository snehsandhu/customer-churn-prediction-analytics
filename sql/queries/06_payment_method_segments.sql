-- SQLite: INNER JOIN attaches the actual payment-method dictionary.
SELECT p.payment_method AS PaymentMethod, COUNT(*) AS customer_count,
       SUM(s.ChurnFlag) AS churn_count,
       100.0 * SUM(s.ChurnFlag) / NULLIF(COUNT(*), 0) AS churn_pct
FROM customer_snapshot AS s
INNER JOIN payment_method_lookup AS p ON s.PaymentMethod = p.payment_method
GROUP BY p.payment_method
ORDER BY p.payment_method;
