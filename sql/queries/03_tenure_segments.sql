-- SQLite: all defined TenureGroup segments, including empty categories.
-- COUNT(customerID), not COUNT(*), avoids counting an unmatched LEFT JOIN row.
SELECT d.tenure_group AS TenureGroup, COUNT(s.customerID) AS customer_count,
       COALESCE(SUM(s.ChurnFlag), 0) AS churn_count,
       100.0 * SUM(s.ChurnFlag) / NULLIF(COUNT(s.customerID), 0) AS churn_pct
FROM tenure_group_lookup AS d
LEFT JOIN customer_snapshot AS s ON s.TenureGroup = d.tenure_group
GROUP BY d.tenure_group, d.display_order
ORDER BY d.display_order;
