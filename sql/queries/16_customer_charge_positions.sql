-- SQLite: top n charges per contract for billing/plan review, not predictions.
-- ROW_NUMBER selects a stable number of rows. RANK preserves tied prices with
-- gaps; DENSE_RANK counts distinct price levels. customerID breaks selection ties.
WITH positions AS (
    SELECT customerID, Contract, MonthlyCharges,
           ROW_NUMBER() OVER (PARTITION BY Contract ORDER BY MonthlyCharges DESC, customerID) AS selection_position,
           RANK() OVER (PARTITION BY Contract ORDER BY MonthlyCharges DESC) AS charge_rank,
           DENSE_RANK() OVER (PARTITION BY Contract ORDER BY MonthlyCharges DESC) AS distinct_price_rank,
           AVG(MonthlyCharges) OVER (PARTITION BY Contract) AS contract_mean_monthly_charges
    FROM customer_snapshot
)
SELECT *, MonthlyCharges - contract_mean_monthly_charges AS gap_from_contract_mean
FROM positions WHERE selection_position <= :top_n
ORDER BY Contract, selection_position;
