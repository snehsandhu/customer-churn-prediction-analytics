-- SQLite: all defined Contract segments, including empty categories.
-- COUNT(customerID), not COUNT(*), avoids counting an unmatched LEFT JOIN row.
SELECT d.contract_name AS Contract, COUNT(s.customerID) AS customer_count,
       COALESCE(SUM(s.ChurnFlag), 0) AS churn_count,
       100.0 * SUM(s.ChurnFlag) / NULLIF(COUNT(s.customerID), 0) AS churn_pct
FROM contract_lookup AS d
LEFT JOIN customer_snapshot AS s ON s.Contract = d.contract_name
GROUP BY d.contract_name, d.display_order
ORDER BY d.display_order;
