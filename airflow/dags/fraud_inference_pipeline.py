"""Scheduled batch inference; application dependencies live in a separate venv."""
from datetime import timedelta

import pendulum
from airflow.decorators import dag, task


@dag(dag_id="fraud_inference_pipeline", schedule="@daily", catchup=False,
     start_date=pendulum.datetime(2026, 1, 1, tz="UTC"), max_active_runs=1,
     default_args={"retries": 2, "retry_delay": timedelta(minutes=2)}, tags=["fraudlens"])
def fraud_inference_pipeline():
    @task.external_python(python="/opt/fraudlens-venv/bin/python", expect_airflow=False)
    def score_batch():
        import os
        from pathlib import Path
        from scripts.run_pipeline import run_daily_inference_pipeline
        source = os.environ.get("INFERENCE_INPUT")
        return run_daily_inference_pipeline(Path(source) if source else None)

    score_batch()


fraud_inference_pipeline()
