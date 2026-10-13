# FraudLens

A batch fraud-scoring project built with Python, scikit-learn, PostgreSQL,
Airflow, and Streamlit. It accepts transaction files, separates invalid rows,
compares models, and stores versioned predictions for review.

The included dataset is **synthetic**. This is a portfolio implementation,
not a validated payment-fraud product. Real-data benchmarking and a complete
Docker/Airflow acceptance run remain release gates.

## Quick start — no Docker required

Use **Python 3.12**. On Windows, WSL2 works well; native PowerShell also works.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m scripts.demo --dashboard
```

PowerShell activation: `.venv\Scripts\Activate.ps1`. Use `py -3.12` in place
of `python3.12` when creating the environment.

The demo generates labeled synthetic data, trains five candidates, scores an
unlabeled copy, and replays the batch to check that it inserts zero duplicate
predictions. It uses SQLite at `.local/demo.db` and opens the dashboard at
http://localhost:8501. Stop it with Ctrl+C. Subsequent launches can reuse the model:

```bash
python -m scripts.demo --skip-training --dashboard
```

## What is implemented

- Required-column and row validation, with a reason for each rejected row.
- Separate training and inference schemas: incoming scoring data needs no `Class` label.
- Stable transaction identity and database uniqueness for replay protection.
- An input snapshot and separate output directory for each pipeline execution.
- A dummy baseline, weighted Logistic Regression, Logistic Regression + SMOTE,
  Random Forest, and XGBoost.
- Train-only preprocessing/resampling, validation-based model and threshold selection,
  and a held-out test partition.
- Immutable model releases with a checksum, saved threshold, risk bands, and atomic pointer publication.
- Atomic persistence of transactions and predictions, with delayed-label support.
- Manual training and daily inference Airflow DAGs.
- A four-page dashboard and CI definitions for tests, PostgreSQL integration, and image builds.

## Flow

```mermaid
flowchart TD
  A[CSV snapshot] --> B[Row validation]
  B --> C[Rejected rows and reasons]
  B --> D[Valid transactions]
  D --> E[Training and validation]
  E --> F[Versioned model release]
  D --> G[Batch scoring]
  F --> G
  G --> H[Atomic database write]
  H --> I[Dashboard]
```

## Run with PostgreSQL and Airflow

Docker Desktop must be running. Copy the local development configuration:

```bash
cp .env.example .env
# Set DATASET_KIND=synthetic for the bundled sample; configure credentials locally.
docker compose up --build -d
```

Open Airflow at http://localhost:8080, then manually trigger
`fraud_training_pipeline`, followed by `fraud_inference_pipeline`.
Open the dashboard at http://localhost:8501. Airflow's default local login is
`admin` / `admin`; the settings are in `.env.example`. Ports bind to localhost.

The ML application runs in a separate Python environment inside the Airflow
image. This keeps SQLAlchemy 2 and the ML dependencies outside Airflow's own
dependency environment. Data and model releases use named volumes.

Each DAG invokes the shared Python pipeline as one external-Python task.
Individual stages have database run records. This keeps retry behavior and
application logic identical between the CLI and Airflow.

For an **existing database**, back it up and apply
`sql/migrations/002_batch_reliability.sql` before running this version. The
migration refuses to delete or silently merge duplicate historical decisions.
Fresh Docker volumes use `sql/schema.sql` automatically. See [the runbook](docs/runbook.md).

## Your own data

Input features: `Time`, `Amount`, `V1`–`V28`.
Training also requires binary `Class` labels. Inference may omit them or provide
partial labels. `transaction_id` is optional but recommended.

Put incoming CSVs in `data/incoming/` (gitignored). With an initialized database
and `DATABASE_URL` configured:

```bash
python -m scripts.run_pipeline training \
  --input data/incoming/creditcard.csv --dataset-kind real --split chronological
python -m scripts.run_pipeline inference --input data/incoming/incoming.csv
```

In Docker, `data/incoming/` is mounted read-only at `/incoming`. Set
`TRAINING_INPUT=/incoming/creditcard.csv`, `INFERENCE_INPUT=/incoming/incoming.csv`,
`TRAINING_SPLIT=chronological`, and `DATASET_KIND=real` in `.env`, then recreate the
services. Do not point inference at the labeled training dataset when evaluating
unseen performance.

If chronological partitions lack both classes, training fails with an explanation.
A stratified split is available for a demonstration, but it does not measure
performance on future transactions. No results on the real dataset are claimed.

## Results and verification

[Verification results](docs/verification.md) distinguish completed local checks
from pending container/PostgreSQL checks. [The model card](docs/model_card.md)
records the synthetic holdout metrics, class counts, and limitations.

```bash
pytest -q
ruff check .
# Optional; must be a disposable database with CREATE SCHEMA permission:
TEST_DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@localhost/fraudlens_test pytest -q
```

A model score is not a calibrated estimate of fraud probability. SMOTE and class
weighting change the training distribution. The dashboard uses LOW/REVIEW/HIGH
bands for triage; `is_flagged` uses the threshold stored with that model release.

## Repository map

| Path | Purpose |
|---|---|
| `src/ingestion`, `src/validation` | Snapshot files and validate records |
| `src/features`, `src/models` | Preprocess, train, evaluate, publish, score |
| `src/database` | Schema models, transactions, reads |
| `scripts/run_pipeline.py` | Tracked batch workflows |
| `airflow/dags` | Scheduling and retries |
| `dashboard` | Operational and model views |
| `tests` | Unit, regression, and PostgreSQL tests |
| `sql/migrations` | Upgrade an existing database |
| `docs` | Decisions, operations, model card, interview notes |

## Current limits

The implementation loads a batch into memory. It is intended for one deployment
host with shared volumes. Snapshot retention, streaming, drift alerts, probability
calibration, and authentication for the dashboard are not implemented.

Fallback IDs hash model features and exclude the label. This is useful for the
anonymous sample, but identical legitimate transactions can collapse. Use stable
upstream IDs for real payment data. Existing UUID records are not automatically
reconciled with new content-derived IDs.

Database registration and filesystem publication are separate operations; their
failure recovery is documented in the runbook. The system is not a distributed
model registry. Historical predictions remain available after retraining, so
prediction counts can exceed the number of unique transactions.

To publish, follow [the GitHub guide](docs/github.md). Git history retains the
original contributors and attribution.
