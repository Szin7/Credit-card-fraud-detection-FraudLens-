"""Training pipeline: ingest -> validate -> train candidates -> register champion.

Not scheduled — retraining is a deliberate action (new data, a code
change, periodic review), not something that should happen automatically
every time inference runs. Trigger it from the Airflow UI or
`airflow dags trigger fraud_training_pipeline`. See fraud_inference_pipeline
for the scheduled daily DAG, and docs/architecture.md for why these are
two separate DAGs instead of one.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pendulum
from airflow.decorators import dag, task

default_args = {
    "owner": "fraudlens",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="fraud_training_pipeline",
    description="Ingest, validate, train candidate models, and register the champion.",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["fraudlens", "training"],
)
def fraud_training_pipeline():
    @task
    def ingest_transactions() -> str:
        from scripts.run_pipeline import run_ingestion_step

        return run_ingestion_step()

    @task
    def validate_transactions(ingestion_run_id: str) -> str:
        from scripts.run_pipeline import run_validation_step

        return run_validation_step()

    @task
    def train_and_register_model(validation_run_id: str) -> str:
        """Trains and compares Dummy/LogisticRegression(+SMOTE)/RandomForest/
        XGBoost, selects the champion by validation PR-AUC, and registers it
        in model_runs — see src/models/train.py for the comparison logic.
        """
        from src.database import repository as repo
        from src.models.train import main as train_models
        from src.utils.config import settings

        train_models()
        with open(settings.model_metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        repo.save_model_run(metadata)
        return metadata["model_type"]

    @task
    def pipeline_summary(champion_model_type: str) -> None:
        print(f"Training pipeline complete. Champion model: {champion_model_type}")

    ingestion_run_id = ingest_transactions()
    validation_run_id = validate_transactions(ingestion_run_id)
    champion_model_type = train_and_register_model(validation_run_id)
    pipeline_summary(champion_model_type)


fraud_training_pipeline()
