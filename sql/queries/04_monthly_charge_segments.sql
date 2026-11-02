-- SQLite: all defined MonthlyChargeGroup segments, including empty categories.
-- COUNT(customerID), not COUNT(*), avoids counting an unmatched LEFT JOIN row.
SELECT d.monthly_charge_group AS MonthlyChargeGroup, COUNT(s.customerID) AS customer_count,
       COALESCE(SUM(s.ChurnFlag), 0) AS churn_count,
       100.0 * SUM(s.ChurnFlag) / NULLIF(COUNT(s.customerID), 0) AS churn_pct
FROM monthly_charge_group_lookup AS d
LEFT JOIN customer_snapshot AS s ON s.MonthlyChargeGroup = d.monthly_charge_group
GROUP BY d.monthly_charge_group, d.display_order
ORDER BY d.display_order;
