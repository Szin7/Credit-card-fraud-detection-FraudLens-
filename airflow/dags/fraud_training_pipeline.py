"""Manual training; register and publish a complete immutable release."""
from datetime import timedelta

import pendulum
from airflow.decorators import dag, task


@dag(dag_id="fraud_training_pipeline", schedule=None, catchup=False,
     start_date=pendulum.datetime(2026, 1, 1, tz="UTC"), max_active_runs=1,
     default_args={"retries": 0, "retry_delay": timedelta(minutes=2)}, tags=["fraudlens"])
def fraud_training_pipeline():
    @task.external_python(python="/opt/fraudlens-venv/bin/python", expect_airflow=False)
    def train_batch():
        import os
        from pathlib import Path
        from scripts.run_pipeline import run_training_pipeline
        source = os.environ.get("TRAINING_INPUT")
        return run_training_pipeline(Path(source) if source else None,
                                     strategy=os.environ.get("TRAINING_SPLIT", "stratified"),
                                     dataset_kind=os.environ.get("DATASET_KIND", "unknown"))

    train_batch()


fraud_training_pipeline()
