# Verification notes

This page separates evidence that exists from work still on the checklist. It is not a production-readiness certificate.

## Checks already recorded

| Check | Evidence |
|---|---|
| Synthetic end-to-end local demonstration | Completed: labeled training, unlabeled scoring, and replay with zero additional decisions |
| Four Streamlit pages | Passed AppTest execution without application exceptions; no browser visual acceptance recorded |
| Local source tests | Previous local report: 54 passed, two PostgreSQL tests skipped without a server |
| GitHub Actions on 2026-10-13, commit `7b8e377` | **Passed**: 56 tests, 85% line coverage, linting, PostgreSQL test service, Compose config, and Docker image builds |
| Airflow image DAG imports | **Passed** in GitHub Actions: both training and inference DAGs imported |
| Synthetic five-model benchmark | Completed; details in `docs/results/synthetic-benchmark.json` |

GitHub run: https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/38361858659

The October 13 GitHub Actions run completed successfully. The full Docker/Airflow execution remains a separate acceptance check.

## Not yet demonstrated

- A complete Docker Compose startup including actual Airflow training and inference execution against PostgreSQL, followed by a replay.
- Local acceptance of dashboard appearance and interactions in a browser, including responsive layout.
- Real-data chronological benchmark on the reference Kaggle dataset.
- Transfer and verification of a Kaggle-trained release in the local application and database.
- Authentication and drift monitoring for a deployed dashboard.

## Commands to repeat the checks

```bash
pytest -q
ruff check .
docker compose config --quiet
docker compose build dashboard airflow-scheduler
```

To verify against PostgreSQL, provide a disposable database with `CREATE SCHEMA` permission:

```bash
TEST_DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@localhost/fraudlens_test pytest -q
```

The CI run is evidence of the tested commit, not proof that future modifications will pass. Rerun the workflow after changes are committed and document the result with its commit SHA.
