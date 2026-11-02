-- SQLite: a covering index for equality on Contract, then a tenure range.
-- MonthlyCharges and ChurnFlag cover the filtered summary's aggregate inputs.
CREATE INDEX IF NOT EXISTS idx_snapshot_contract_tenure_charges
    ON customer_snapshot (Contract, tenure, MonthlyCharges, ChurnFlag);
-- Index the other commonly grouped/joined source lookup key.
CREATE INDEX IF NOT EXISTS idx_snapshot_payment_method
    ON customer_snapshot (PaymentMethod);
