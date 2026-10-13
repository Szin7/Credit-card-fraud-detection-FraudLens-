# Architecture decisions

FraudLens is a single-host batch application. The code runs directly from the CLI
or inside an Airflow external-Python task. PostgreSQL serves the dashboard;
SQLite supports the local demonstration and most tests.

## Batch boundaries

Each run gets `data/processed/runs/<uuid>/`. Ingestion copies the incoming CSV to
`source.csv` and hashes that snapshot. Validation and scoring read only this copy.
The directory contains the manifest, valid/rejected rows, summary, and predictions.
Two runs never share an output filename. Input publishers should deliver complete
files atomically; ingestion cannot guarantee a consistent snapshot of a file being
modified during the copy.

Airflow schedules one application task per batch. Database records track ingestion,
validation, training, and inference separately. Every stage exception marks its run
failed. A hard process kill can leave a running record; recovery is an operator task.

## Identity and retries

An upstream `transaction_id` is preferred and preserved as a string. Without one,
validation hashes the numeric features in a fixed order, excluding `Class`.
Identical feature rows share an ID. This is appropriate for this anonymous fixture,
but may merge distinct real transactions: upstream identity is a deployment requirement.

The database enforces transaction primary keys and unique
`(transaction_id, model_version, threshold)` prediction decisions. Inserts use
`ON CONFLICT DO NOTHING`. Transaction and prediction writes share one transaction.
A failed prediction write rolls back inserted transactions. Retrying a completed
batch with the same release inserts no additional decisions.

An existing ID with different feature values or a contradictory known label raises
an error. A previously unknown label can be attached later. The first source-run
reference is retained. This is not a complete multi-source provenance graph.

## Model lifecycle

Preprocessing is part of each fitted pipeline. SMOTE sees training data only.
Model selection uses validation average precision (the metric labeled PR-AUC in
the UI). The threshold maximizes validation F1 on a fixed 0.05–0.95 grid; ties favor
higher precision, then a higher threshold. This is a demonstration policy, not a
business-approved payment decision rule.

Only then is the champion evaluated on the holdout. A release directory stores the
model, strict JSON metadata, model comparison, and validation threshold sweep.
Metadata records the source checksum, split strategy, dataset kind, sample counts,
threshold, risk bands, and artifact checksum.

Training uses a filesystem lock. Registration succeeds before an atomic pointer
update publishes the release. Inference resolves the pointer once, verifies the
artifact checksum, and loads the model and metadata from that same directory.
Checksums detect accidental corruption; they are not signatures. Joblib artifacts
must come from a trusted training process.

Database and filesystem updates are not one distributed transaction. A crash between
registration and pointer publication can leave the dashboard and inference referring
to different releases. See the recovery procedure in the runbook.

## Evaluation choices

The synthetic demo uses stratification to preserve its small positive class.
`--split chronological` partitions distinct timestamps in 64/16/20 proportions,
keeping timestamp ties together. Row counts may differ from those proportions.
Each partition must contain both classes. Neither strategy handles customer/card
entity leakage because this anonymized schema has no entity identifier.

## Deployment boundaries

The dashboard image installs only dashboard/database dependencies. The Airflow image
has a separate `/opt/fraudlens-venv`; application dependencies do not alter Airflow's
SQLAlchemy installation. Shared named volumes hold batch files and model releases.
This design assumes one host and shared local storage. It is not a multi-worker
or multi-region deployment design.

S3 mode uploads raw snapshots through boto3's standard credential chain. It does
not implement S3 event ingestion, remote model storage, or an AWS deployment.

Reference implementations used for conflict handling:
- https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#insert-on-conflict-upsert
- https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#insert-on-conflict-upsert
