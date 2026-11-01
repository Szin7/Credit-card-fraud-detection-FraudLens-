# FraudLens: verification and remaining work

This is an engineering acceptance checklist, not a claim that all checks are complete.

## Verified locally on 1 November 2026

- [x] Kaggle real dataset: 284,807 rows, 1,081 exact duplicates rejected.
- [x] Five candidate models trained with a chronological train / validation / test split.
- [x] Real-data XGBoost test performance checked against the release metadata and confusion matrix.
- [x] Release exported, checksum validated, imported to WSL, and used for local predictions.
- [x] Local SQLite real-feature demonstration: 3,076 predictions, 54 flagged, repeat execution inserted zero new predictions.
- [x] Streamlit Overview, Transaction Explorer, Model Performance, and Pipeline Health were opened and reviewed.
- [x] Local integration checks: 65 tests passed, 2 skipped; Ruff clean.
- [x] PostgreSQL container started and reported healthy on host port 5433 before the Docker crash.

## Container verification and remaining work

- [ ] Recover the Docker Desktop Linux engine after the `SIGBUS` image-export fault and HTTP 500 error.
- [ ] Keep Docker's disk image off a nearly full C: drive, if another suitably sized drive is available.
- [x] Final local integration checks passed: 65 tests, 2 skipped, Ruff clean.
- [x] Full-stack GitHub Actions test passed on `main`: https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41317199864
- [x] Kaggle-trained XGBoost model imported and used in the Docker/Airflow/PostgreSQL test on `main`: https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41322535119
- [x] The imported real-trained release scored 6,080 synthetic integration-test rows; replay inserted no duplicate decisions.
- [x] PostgreSQL schema, Airflow initialization, scheduler and both DAGs verified on GitHub's runner.
- [x] Containerized replay inserted no duplicate predictions.
- [x] Containerized Streamlit health and PostgreSQL-backed model metadata checks passed.
- [ ] Review container resources, vulnerabilities, and environment variables before any public deployment.
- [x] Portfolio implementation and real-model integration merged into `main` through pull requests #1–#3.

## Reporting rules

The 59,888-row chronological test metrics are the actual real-data evaluation. The 3,076-row deliberately enriched subset only demonstrates scoring and dashboard behavior. Keep the synthetic benchmark separate. Do not commit `.env`, raw Kaggle data, SQLite files, or serialized model releases to Git. The checked model ZIP is distributed separately as a public GitHub Release asset.
