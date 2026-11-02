-- SQLite: all combinations of REAL source contract/payment labels.
-- Empty combinations are analytical cells, not fabricated customers.
SELECT c.contract_name AS Contract, p.payment_method AS PaymentMethod,
       COUNT(s.customerID) AS customer_count, COALESCE(SUM(s.ChurnFlag), 0) AS churn_count,
       100.0 * SUM(s.ChurnFlag) / NULLIF(COUNT(s.customerID), 0) AS churn_pct,
       100.0 * COUNT(s.customerID) /
           NULLIF(SUM(COUNT(s.customerID)) OVER (PARTITION BY c.contract_name), 0) AS within_contract_customer_pct
FROM contract_lookup AS c
CROSS JOIN payment_method_lookup AS p
LEFT JOIN customer_snapshot AS s ON s.Contract = c.contract_name AND s.PaymentMethod = p.payment_method
GROUP BY c.contract_name, c.display_order, p.payment_method
ORDER BY c.display_order, p.payment_method;
