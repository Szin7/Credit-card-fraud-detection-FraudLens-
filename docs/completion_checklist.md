# Project verification checklist

This checklist distinguishes the completed portfolio demonstration from work that would be required before a live deployment.

## Research and local demonstration

- [x] Validated the ULB/Kaggle dataset and recorded 1,081 rejected exact duplicates.
- [x] Compared five model candidates using a chronological train/validation/test split.
- [x] Selected XGBoost and the decision threshold using validation data only.
- [x] Evaluated the selected model on the 59,888-row held-out test partition.
- [x] Exported the trusted model release and verified its archive and model checksums.
- [x] Scored 3,076 fraud-enriched holdout-feature examples into local SQLite; the same batch produced no duplicates on replay.
- [x] Opened and reviewed the four Streamlit dashboard pages.

## Automated and container verification

- [x] Current `main` commit `ed60d4a`: Python tests, linting, Docker builds and DAG import checks passed in [CI](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41326322968).
- [x] Synthetic full-stack training, inference, replay protection and PostgreSQL/Streamlit checks passed on a [GitHub-hosted runner](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41317199864).
- [x] Trusted Kaggle-trained model was imported into Docker, registered in PostgreSQL, and used by Airflow for 6,080 synthetic test transactions in the [real-model acceptance run](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/actions/runs/41322535119).
- [x] Replaying that batch inserted zero duplicate decisions; dashboard/database verification passed.
- [ ] Rerun the manual real-model acceptance workflow on the current single-commit `main` snapshot to obtain a matching latest-commit run record (optional for the unchanged implementation).

## Outside the current portfolio scope

- [ ] Persistent hosted service with authentication and secure secrets management.
- [ ] Production monitoring, alerting, drift checks, audit retention and operational threshold policies.
- [ ] Long-term performance validation on genuinely future payment data.
- [ ] Recovery of the earlier local Docker Desktop Linux-engine issue, if local container demonstration is needed.

## Reporting rules

Use the 59,888-row chronological test results for real-data accuracy discussions. The 3,076-row local demonstration and 6,080-row synthetic container acceptance serve different purposes and must not be reported as independent real-world model evaluation. Do not commit `.env`, the raw Kaggle dataset, local SQLite files or serialized models. The checked model archive is a separate [public GitHub Release asset](https://github.com/Szin7/Credit-card-fraud-detection-FraudLens-/releases/tag/kaggle-model).
