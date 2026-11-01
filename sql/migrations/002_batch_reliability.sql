-- Back up the database before applying this migration to an existing install.
-- This script deliberately refuses to delete historical duplicate predictions.
BEGIN;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS content_hash TEXT;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM fraud_predictions
               GROUP BY transaction_id, model_version, threshold HAVING count(*) > 1) THEN
        RAISE EXCEPTION 'Duplicate prediction decisions exist. Archive/reconcile them before applying migration 002.';
    END IF;
END $$;
CREATE UNIQUE INDEX IF NOT EXISTS uq_prediction_decision
ON fraud_predictions (transaction_id, model_version, threshold);
COMMIT;
