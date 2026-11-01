# Explaining FraudLens in an interview

Use only statements that match the work you can demonstrate and explain. The examples here describe verified project capabilities, not a claim of experience running a live fraud-prevention service.

## Short project description

FraudLens is a credit-card fraud-detection research and batch-processing project. It compares classifiers on an imbalanced transaction dataset, saves a versioned model, scores new unlabeled batches and presents the decisions in a Streamlit dashboard. A Docker-based Airflow pipeline and PostgreSQL database handle the containerized workflow.

## Evidence you can discuss

- Compared five classifiers using validation average precision on the ULB/Kaggle credit-card dataset; selected XGBoost with a 0.95 validation-selected threshold.
- Evaluated the frozen model on 59,888 chronologically held-out transactions: precision **88.52%**, recall **71.05%** and F1 **78.83%** (54 true positives, 7 false positives and 22 false negatives).
- Used a local SQLite database and four Streamlit pages to demonstrate predictions from 3,076 deliberately fraud-enriched holdout-feature rows. That sample is a dashboard demonstration, **not** another independent accuracy estimate.
- Verified the saved Kaggle-trained XGBoost release inside a GitHub-hosted Docker/Airflow/PostgreSQL workflow, scoring 6,080 synthetic integration-test rows and confirming that replay introduced no duplicate decisions.
- Maintained tests, linting, container builds, DAG validation and runbooks in GitHub.

Adapt these points to the responsibilities you actually held. In particular, be ready to explain how input validation, threshold selection, persistence and replay protection work in code.

## Demonstration order

1. Describe the problem of rare fraud cases and why accuracy alone is misleading.
2. Explain the chronological split, validation-only model and threshold selection, and untouched test partition.
3. Open the Streamlit Overview and Transaction Explorer pages, then Model Performance and Pipeline Health.
4. Show how the model release is imported into Docker and used by Airflow without retraining.
5. Open the successful CI and real-model acceptance runs, then discuss limits and possible improvements.

## Common follow-up questions

**Why XGBoost?** It had the highest validation average precision among the five evaluated candidates. The final holdout results were calculated only after selection.

**Why a 0.95 threshold?** That threshold maximized validation F1 over the implemented search grid; it is not an approved operational banking threshold.

**Why both SQLite and PostgreSQL?** SQLite makes local demonstrations and tests easier. PostgreSQL is used for the Docker/Airflow integration and dashboard database checks.

**How is duplicate scoring avoided?** Prediction decisions have a unique key combining transaction identity, model version and threshold. Replaying the same batch does not insert another identical decision.

**Is it production-ready?** No. The application uses batch scoring and an anonymized research dataset; a live service would need access control, monitoring, privacy controls, threshold governance, retention policies and validation against future data.

For deeper walkthroughs, see the [interview guide](interview_guide.md), [architecture](architecture.md) and [real-data model card](real_model_card.md).
