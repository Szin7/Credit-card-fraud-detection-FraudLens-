# Runbook

## First run

Use Python 3.12 and `requirements-dev.txt`. Run `python -m scripts.demo --dashboard`
for the SQLite demonstration. It generates data under `.local/` and does not need
AWS or Docker. The dashboard process inherits the demo database URL.

For PostgreSQL/Airflow, follow README. `docker compose logs airflow-init` should
show a successful metadata migration and admin creation/check. Then trigger manual
training before inference. DAGs start paused; unpause inference only after a manual
run succeeds. Scheduled demo runs score the same configured CSV and should insert
zero repeated decisions for the same active model.

## Existing database

Back up the database. Apply the migration from the repository root:

```bash
docker compose exec -T postgres sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < sql/migrations/002_batch_reliability.sql
```

The migration adds the feature checksum and prediction uniqueness index. If older
predictions have duplicate decision keys, it stops without deleting them. Reconcile
and archive duplicates before retrying. Old UUID transactions do not match new
fallback IDs; retain the old dataset as a separate historical demonstration or start
with a separate fresh database. Do not claim cross-version deduplication of old IDs.

`init_db()` creates missing tables; it is not a schema migration engine. Never use
`docker compose down -v` as an upgrade step: that deletes named-volume data.

## Input and output

- Host input: `data/incoming/`; Docker input: `/incoming/`.
- Training requires complete binary labels; scoring does not.
- Each run directory contains a snapshot and validation evidence.
- Review rejected rows before accepting the batch. A batch with no valid rows fails.
- Match model metadata's source checksum with the ingestion manifest for training lineage.

Snapshots contain transaction features and are intentionally gitignored. Set a
retention policy before processing sensitive or large volumes; cleanup is not automated.

## Retry and incident checks

Use Pipeline Health to locate the failing stage. Inspect its error and logs. Correct
the input or configuration, then rerun. A scoring retry with the same model and
threshold does not duplicate existing decisions.

A hard-killed process can leave a stage marked running. Confirm that it has stopped
before marking the record failed. Do not modify records belonging to active tasks.

A failed database write rolls back transactions and predictions together. Run status
and CSV output are separate operations; a crash after commit can report a failed
stage even though decisions were saved. Retrying is the recovery path.

## Model publication recovery

Training is serialized by `models/.training.lock`. Do not delete an active lock.
Model files and metadata are immutable after publication.

If database registration succeeds but pointer publication fails:
1. Stop new training runs.
2. Find the release returned/logged by training; inspect its metadata and checksum.
3. Confirm its version is the registered champion in `model_runs`.
4. Publish that existing release with `publish_release(Path(...))` from
   `src.models.artifacts`, using the same application environment and model directory.
5. Run a small inference batch and verify its recorded version.

Rolling back requires registering the selected old release metadata with
`save_model_run()` and publishing that same directory. Back up first and keep
training stopped during the change. Historical predictions are retained.

## Release acceptance

- `pytest -q` and `ruff check .` pass.
- PostgreSQL tests pass against the real schema, not only SQLite.
- Both images build; both DAGs import and run in Docker.
- Trigger training, then inference twice; the second run inserts zero duplicate decisions.
- All four dashboard pages load after training and scoring.
- Real-data evaluation uses a held-out future partition and states its limitations.

The local verification report states which of these were actually performed.
