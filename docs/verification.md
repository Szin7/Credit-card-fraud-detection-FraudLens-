# Verification notes

This page separates evidence that exists from work still on the checklist. It is not a production-readiness certificate.

## Checks already recorded

| Check | Evidence |
|---|---|
| Synthetic end-to-end local demonstration | Completed: labeled training, unlabeled scoring, and replay with zero additional decisions |
| Four Streamlit pages | AppTest passed and all four pages were opened and reviewed in a local browser |
| Local source tests | Most recent reported WSL check: 65 passed, 2 skipped; Ruff clean |
| GitHub Actions on 2026-10-13, commit `7b8e377` | **Passed**: 56 tests, 85% line coverage, linting, PostgreSQL test service, Compose config, and Docker image builds |
| Airflow image DAG imports | **Passed** in GitHub Actions: both training and inference DAGs imported |
| Synthetic five-model benchmark | Completed; details in `docs/results/synthetic-benchmark.json` |

GitHub run: https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/38361858659

The October 13 run established the initial baseline. A later full Docker/Airflow execution passed on 1 November 2026, as recorded below.

## Not yet demonstrated

- Responsive layout across mobile screen sizes has not been fully verified.
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

## Full-stack acceptance — 1 November 2026

Commit: `04542ad` on `fraudlens-work`.

- Tests and container builds passed:
  https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41313247146
- Full-stack check passed:
  https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41313247184

The full-stack run built the containers on GitHub, initialized
PostgreSQL and Airflow, registered both DAGs, executed training
and inference, and stored 6,080 synthetic transaction predictions.
Repeating inference did not add duplicate predictions.

The Streamlit health endpoint and PostgreSQL-backed dashboard
metadata check passed.

This is a synthetic-data deployment acceptance test. The separate
Kaggle benchmark is the source of the reported real-data performance.
Local Docker Desktop recovery and production security hardening are
not covered by this CI run.

## Kaggle-trained model in Docker — 1 November 2026

Commit: `a005209` on `main`. [Real-model pipeline check #4](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41322535119) passed.

- The runner downloaded the published `kaggle-model` archive and checked its SHA-256.
- The imported XGBoost release was registered as the real-data model in PostgreSQL.
- Both Airflow DAGs registered; inference ran using that imported release, without a new training run.
- Inference stored 6,080 predictions from synthetic integration-test transactions. Repeating the batch added zero duplicate decisions.
- Streamlit health and PostgreSQL-backed model metadata checks passed.

This demonstrates compatibility of the Kaggle-trained model with the Docker/Airflow application. Fraud-detection precision and recall come from the separate 59,888-row real-data holdout, not the synthetic scoring fixture. GitHub's test runner was temporary and does not host a live application.
