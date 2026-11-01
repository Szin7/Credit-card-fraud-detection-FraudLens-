# FraudLens: verification and remaining work

This is an engineering acceptance checklist, not a claim that all checks are complete.

## Verified locally on 1 November 2026

- [x] Kaggle real dataset: 284,807 rows, 1,081 exact duplicates rejected.
- [x] Five candidate models trained with a chronological train / validation / test split.
- [x] Real-data XGBoost test performance checked against the release metadata and confusion matrix.
- [x] Release exported, checksum validated, imported to WSL, and used for local predictions.
- [x] Local SQLite real-feature demonstration: 3,076 predictions, 54 flagged, repeat execution inserted zero new predictions.
- [x] Streamlit Overview, Transaction Explorer, Model Performance, and Pipeline Health were opened and reviewed.
- [x] Local Python suite previously reported 60 passed, 2 skipped; Ruff passed before and after dashboard adjustments. The latest chart-only edit was screenshot-checked, but its test output has not been provided.
- [x] PostgreSQL container started and reported healthy on host port 5433 before the Docker crash.

## Still to verify

- [ ] Recover the Docker Desktop Linux engine after the `SIGBUS` image-export fault and HTTP 500 error.
- [ ] Keep Docker's disk image off a nearly full C: drive, if another suitably sized drive is available.
- [ ] Run `ruff check .` and `pytest -q` after all documentation and script changes.
- [ ] Run the optional full-stack GitHub Actions workflow and retain the successful run URL / logs.
- [ ] Confirm PostgreSQL schema, Airflow initialization, scheduler, and both DAGs.
- [ ] Confirm repeated containerized inference stores no duplicate decisions.
- [ ] Confirm containerized Streamlit reads data and model metrics from PostgreSQL.
- [ ] Review container resources, vulnerabilities, and environment variables before any public deployment.
- [ ] Commit/push verified changes on `fraudlens-work`; review and merge into `main` when appropriate.

## Reporting rules

The 59,888-row chronological test metrics are the actual real-data evaluation. The 3,076-row deliberately enriched subset only demonstrates scoring and dashboard behavior. Keep the synthetic benchmark separate. Never include `.env`, raw Kaggle data, SQLite files, or serialized model releases in the repository.
