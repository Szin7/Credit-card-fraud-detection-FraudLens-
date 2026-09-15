-- Reference queries against the FraudLens schema (sql/schema.sql).
-- These are the same shapes src/database/repository.py runs through
-- SQLAlchemy for the dashboard; kept here as plain, readable SQL for
-- anyone exploring the database directly (psql, a BI tool, etc).


-- 1. Highest-risk transactions currently on record.
SELECT
    p.transaction_id,
    t.amount,
    p.fraud_probability,
    p.risk_level,
    p.model_version,
    p.prediction_timestamp
FROM fraud_predictions p
JOIN transactions t ON t.transaction_id = p.transaction_id
ORDER BY p.fraud_probability DESC
LIMIT 20;


-- 2. Daily transaction counts.
-- This dataset's `Time` is elapsed seconds from the first transaction, not
-- a calendar timestamp, so "daily" here means 24-hour buckets from
-- ingestion rather than actual calendar days.
SELECT
    FLOOR(t.time_elapsed / 86400)::INT AS day_bucket,
    COUNT(*) AS transaction_count
FROM transactions t
GROUP BY day_bucket
ORDER BY day_bucket;


-- 3. Fraud counts and rate, predicted vs. actual (ground truth is known
-- here because this is historical/backtested data — a live system
-- typically wouldn't have actual_class available at prediction time).
SELECT
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN p.is_flagged THEN 1 ELSE 0 END) AS predicted_fraud_count,
    ROUND(100.0 * SUM(CASE WHEN p.is_flagged THEN 1 ELSE 0 END) / COUNT(*), 4) AS predicted_fraud_rate_pct,
    SUM(CASE WHEN t.actual_class = 1 THEN 1 ELSE 0 END) AS actual_fraud_count,
    ROUND(100.0 * SUM(CASE WHEN t.actual_class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS actual_fraud_rate_pct
FROM fraud_predictions p
JOIN transactions t ON t.transaction_id = p.transaction_id;


-- 4. Prediction (risk band) distribution.
SELECT
    risk_level,
    COUNT(*) AS prediction_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
FROM fraud_predictions
GROUP BY risk_level
ORDER BY CASE risk_level WHEN 'HIGH' THEN 1 WHEN 'REVIEW' THEN 2 WHEN 'LOW' THEN 3 END;


-- 5. Model-version comparison — every trained model's headline metrics,
-- most recent first, with the current champion flagged.
SELECT
    model_version,
    model_type,
    trained_at,
    is_champion,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    pr_auc
FROM model_runs
ORDER BY trained_at DESC;


-- 6. Recent high-risk transactions (last 24h of prediction activity).
SELECT
    p.transaction_id,
    t.amount,
    p.fraud_probability,
    p.model_version,
    p.prediction_timestamp
FROM fraud_predictions p
JOIN transactions t ON t.transaction_id = p.transaction_id
WHERE p.risk_level = 'HIGH'
  AND p.prediction_timestamp >= now() - INTERVAL '24 hours'
ORDER BY p.prediction_timestamp DESC;


-- Bonus: pipeline health — most recent run of each type, with duration.
SELECT DISTINCT ON (run_type)
    run_type,
    run_id,
    status,
    started_at,
    finished_at,
    EXTRACT(EPOCH FROM (finished_at - started_at)) AS duration_seconds,
    records_processed,
    records_rejected
FROM pipeline_runs
ORDER BY run_type, started_at DESC;
