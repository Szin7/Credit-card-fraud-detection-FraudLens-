# Project changes

FraudLens is a batch fraud-detection research application. This page records major implementation milestones; it is not a substitute for test results or a claim of production deployment.

## 1 November 2026 — Real-data evaluation and integration

- Evaluated five classification candidates on the ULB/Kaggle credit-card dataset using a chronological train/validation/test split.
- Selected XGBoost using validation average precision, then selected a 0.95 decision threshold using validation F1.
- Recorded the held-out test results: precision 0.8852, recall 0.7105, F1 0.7883 and average precision 0.7954.
- Exported and checksum-verified the trained model release; used it for a local SQLite/Streamlit demonstration.
- Verified the imported Kaggle-trained model with Docker, Airflow and PostgreSQL on GitHub Actions. The model scored 6,080 synthetic integration-test transactions, and a repeated run added no duplicate decisions.
- Updated the README and supporting documentation to distinguish research results, demonstration data and integration tests.

## Core application

- Added transaction ingestion, schema validation and rejected-row reporting.
- Separated model training from batch inference and saved preprocessing with each versioned model release.
- Built the Streamlit Overview, Transaction Explorer, Model Performance and Pipeline Health pages.
- Implemented transaction and prediction persistence using SQLite locally and PostgreSQL in Docker.
- Added Airflow DAGs, Docker Compose configurations, regression tests and GitHub Actions checks.
- Made repeated inference idempotent for the same transaction, model version and decision threshold.

## Verification records

- [Current main-branch tests and container builds](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41326322968)
- [Synthetic full-stack acceptance](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41317199864)
- [Kaggle-trained model integration acceptance](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41322535119)

The integration checks run on temporary GitHub-hosted machines. FraudLens is not a persistent public payment-processing service; see [verification](docs/verification.md) and the [real-data model card](docs/real_model_card.md) for the evidence and limitations.
