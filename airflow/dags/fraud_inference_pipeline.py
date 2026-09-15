"""Daily inference pipeline: ingest -> validate -> score -> persist -> summarize.

Feature transformation happens inside the persisted model pipeline
(src/features/build_features.py) as part of scoring, not as its own task
here — splitting it into a standalone task would mean re-implementing that
transformation outside the fitted pipeline, which is exactly the
training/serving skew ColumnTransformer is meant to prevent. Scoring and
storing predictions are one task for a similar reason: predictions are a
full DataFrame, and passing bulk data between Airflow tasks via XCom (which
persists through the metadata database) is a known anti-pattern — writing
straight to Postgres inside one task is the correct approach.
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.decorators import dag, task

default_args = {
    "owner": "fraudlens",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="fraud_inference_pipeline",
    description="Daily batch scoring of incoming transactions.",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=default_args,
    tags=["fraudlens", "inference"],
)
def fraud_inference_pipeline():
    @task
    def ingest_transactions() -> str:
        from scripts.run_pipeline import run_ingestion_step

        return run_ingestion_step()

    @task
    def validate_transactions(ingestion_run_id: str) -> str:
        from scripts.run_pipeline import run_validation_step

        return run_validation_step()

    @task
    def generate_and_store_predictions(validation_run_id: str) -> str:
        from scripts.run_pipeline import run_inference_step

        return run_inference_step(validation_run_id)

    @task
    def pipeline_summary(inference_run_id: str) -> None:
        from src.database import repository as repo

        runs = repo.get_pipeline_runs(limit=50)
        this_run = runs[runs["run_id"] == inference_run_id].iloc[0]
        print(
            f"Inference pipeline complete. status={this_run['status']} "
            f"records_processed={this_run['records_processed']}"
        )

    ingestion_run_id = ingest_transactions()
    validation_run_id = validate_transactions(ingestion_run_id)
    inference_run_id = generate_and_store_predictions(validation_run_id)
    pipeline_summary(inference_run_id)


fraud_inference_pipeline()
