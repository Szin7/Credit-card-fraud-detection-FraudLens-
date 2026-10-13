-- FraudLens application schema.
--
-- Deliberately excludes the anonymized V1-V28 model features: nothing in
-- the app/dashboard layer consumes them directly, they live in
-- data/processed/ as the model's feature store input, and duplicating 28
-- numeric columns into the serving database for no reader would just be
-- extra storage and write cost. See docs/architecture.md.
--
-- Applied automatically by the postgres container on first start (mounted
-- into /docker-entrypoint-initdb.d — see docker-compose.yml) and can be
-- re-applied locally with `make db-init`.

-- One row per execution of any pipeline stage. Every other table traces
-- its provenance back here.
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id              TEXT PRIMARY KEY,
    run_type            TEXT NOT NULL CHECK (run_type IN ('ingestion', 'validation', 'training', 'inference')),
    status              TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at          TIMESTAMPTZ NOT NULL,
    finished_at         TIMESTAMPTZ,
    records_processed   INTEGER,
    records_rejected    INTEGER,
    error_message       TEXT
);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_started_at ON pipeline_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_run_type ON pipeline_runs (run_type);


-- The detailed breakdown behind a validation-type pipeline_runs row, kept
-- separate so pipeline_runs itself stays generic across every run type.
CREATE TABLE IF NOT EXISTS validation_results (
    validation_id       SERIAL PRIMARY KEY,
    pipeline_run_id     TEXT NOT NULL REFERENCES pipeline_runs (run_id),
    total_rows          INTEGER NOT NULL,
    valid_rows          INTEGER NOT NULL,
    rejected_rows       INTEGER NOT NULL,
    duplicate_rows      INTEGER NOT NULL,
    missing_value_rows  INTEGER NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_validation_results_pipeline_run_id ON validation_results (pipeline_run_id);


-- Transaction-level summary for the app/serving layer.
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id      TEXT PRIMARY KEY,
    content_hash        TEXT,
    time_elapsed         NUMERIC NOT NULL,
    amount               NUMERIC NOT NULL,
    -- Ground-truth label, when known (this is historical/backtested data).
    -- A live production feed would leave this NULL until a chargeback or
    -- confirmed-fraud signal arrives, sometimes weeks later.
    actual_class          SMALLINT CHECK (actual_class IN (0, 1)),
    source_run_id          TEXT REFERENCES pipeline_runs (run_id),
    ingested_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_transactions_time_elapsed ON transactions (time_elapsed);
CREATE INDEX IF NOT EXISTS idx_transactions_ingested_at ON transactions (ingested_at DESC);


-- One row per trained model, including the comparison-table and curve data
-- the Model Performance dashboard page needs, so the dashboard only ever
-- talks to Postgres and never needs direct filesystem access to models/.
CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id        SERIAL PRIMARY KEY,
    model_version        TEXT NOT NULL UNIQUE,
    model_type           TEXT NOT NULL,
    trained_at           TIMESTAMPTZ NOT NULL,
    is_champion          BOOLEAN NOT NULL DEFAULT FALSE,
    decision_threshold   NUMERIC NOT NULL,
    random_seed          INTEGER NOT NULL,
    precision_score      NUMERIC NOT NULL,
    recall_score         NUMERIC NOT NULL,
    f1_score             NUMERIC NOT NULL,
    roc_auc              NUMERIC NOT NULL,
    pr_auc               NUMERIC NOT NULL,
    feature_columns      JSONB NOT NULL,
    curve_data           JSONB NOT NULL,
    model_comparison     JSONB NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_model_runs_trained_at ON model_runs (trained_at DESC);
-- Enforces "at most one champion" at the database level rather than only
-- in application code.
CREATE UNIQUE INDEX IF NOT EXISTS idx_model_runs_one_champion ON model_runs (is_champion) WHERE is_champion;


-- One row per (transaction, model_version) scored.
CREATE TABLE IF NOT EXISTS fraud_predictions (
    prediction_id          BIGSERIAL PRIMARY KEY,
    transaction_id         TEXT NOT NULL REFERENCES transactions (transaction_id),
    fraud_probability      NUMERIC NOT NULL CHECK (fraud_probability BETWEEN 0 AND 1),
    risk_level              TEXT NOT NULL CHECK (risk_level IN ('LOW', 'REVIEW', 'HIGH')),
    is_flagged               BOOLEAN NOT NULL,
    model_version            TEXT NOT NULL,
    threshold                NUMERIC NOT NULL,
    prediction_timestamp     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_prediction_decision UNIQUE (transaction_id, model_version, threshold)
);

CREATE INDEX IF NOT EXISTS idx_fraud_predictions_transaction_id ON fraud_predictions (transaction_id);
CREATE INDEX IF NOT EXISTS idx_fraud_predictions_risk_level ON fraud_predictions (risk_level);
CREATE INDEX IF NOT EXISTS idx_fraud_predictions_prediction_timestamp ON fraud_predictions (prediction_timestamp DESC);
