-- SQLite: LAG/LEAD compare adjacent tenure bands in a CROSS-SECTION.
-- They do not measure historical changes, transitions or survival hazards.
WITH bands AS (
    SELECT t.tenure_group AS TenureGroup, t.display_order,
           COUNT(s.customerID) AS customer_count,
           COALESCE(SUM(s.ChurnFlag), 0) AS churn_count,
           100.0 * SUM(s.ChurnFlag) / NULLIF(COUNT(s.customerID), 0) AS churn_pct
    FROM tenure_group_lookup AS t
    LEFT JOIN customer_snapshot AS s ON s.TenureGroup = t.tenure_group
    GROUP BY t.tenure_group, t.display_order
)
SELECT TenureGroup, customer_count, churn_count, churn_pct,
       LAG(churn_pct) OVER (ORDER BY display_order) AS previous_band_churn_pct,
       LEAD(churn_pct) OVER (ORDER BY display_order) AS next_band_churn_pct,
       churn_pct - LAG(churn_pct) OVER (ORDER BY display_order) AS difference_from_previous_pp,
       SUM(customer_count) OVER (
           ORDER BY display_order ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS cumulative_customers
FROM bands ORDER BY display_order;
