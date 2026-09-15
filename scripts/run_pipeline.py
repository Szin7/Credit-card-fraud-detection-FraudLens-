"""Orchestration entry points for the batch pipeline.

Both the Makefile (local dev) and the Airflow DAGs (airflow/dags/) call
into these functions rather than duplicating the step sequence, so there's
exactly one place that defines what "the inference pipeline" or "the
training pipeline" actually means.

Training is intentionally a separate pipeline from daily inference — a
model doesn't need to be retrained every time new transactions are scored.
See airflow/dags/ and docs/architecture.md.
"""

from __future__ import annotations

import json
from uuid import uuid4

from src.database import repository as repo
from src.database.connection import init_db
from src.ingestion.ingest import IngestionError, ingest
from src.models.predict import generate_predictions
from src.models.train import main as train_models
from src.processing.transform import load_processed_transactions
from src.utils.config import settings
from src.utils.logging import get_logger, log_event
from src.validation.validator import ValidationError, run_validation

logger = get_logger(__name__)


def run_ingestion_step() -> str:
    run_id = uuid4().hex
    repo.start_pipeline_run(run_id, "ingestion")
    try:
        result = ingest()
        repo.finish_pipeline_run(run_id, "succeeded", records_processed=result.row_count)
    except IngestionError as exc:
        repo.finish_pipeline_run(run_id, "failed", error_message=str(exc))
        log_event(logger, "pipeline_failed", stage="ingestion", error=str(exc))
        raise
    return run_id


def run_validation_step() -> str:
    run_id = uuid4().hex
    repo.start_pipeline_run(run_id, "validation")
    try:
        summary = run_validation()
        repo.finish_pipeline_run(
            run_id, "succeeded", records_processed=summary.valid_rows, records_rejected=summary.rejected_rows
        )
        repo.save_validation_result(run_id, summary)
    except ValidationError as exc:
        repo.finish_pipeline_run(run_id, "failed", error_message=str(exc))
        log_event(logger, "pipeline_failed", stage="validation", error=str(exc))
        raise
    return run_id


def run_inference_step(validation_run_id: str) -> str:
    """Persist the validated transactions, score them, and store predictions."""
    run_id = uuid4().hex
    repo.start_pipeline_run(run_id, "inference")
    try:
        valid_df = load_processed_transactions()
        repo.save_transactions(valid_df, source_run_id=validation_run_id)

        predictions = generate_predictions(df=valid_df)
        repo.save_predictions(predictions)

        repo.finish_pipeline_run(run_id, "succeeded", records_processed=len(predictions))
    except Exception as exc:
        repo.finish_pipeline_run(run_id, "failed", error_message=str(exc))
        log_event(logger, "pipeline_failed", stage="inference", error=str(exc))
        raise
    return run_id


def run_daily_inference_pipeline() -> dict:
    """ingest -> validate -> persist transactions -> predict -> persist predictions."""
    init_db()
    log_event(logger, "pipeline_started", pipeline="daily_inference")

    ingestion_run_id = run_ingestion_step()
    validation_run_id = run_validation_step()
    inference_run_id = run_inference_step(validation_run_id)

    summary = {
        "ingestion_run_id": ingestion_run_id,
        "validation_run_id": validation_run_id,
        "inference_run_id": inference_run_id,
    }
    log_event(logger, "pipeline_summary", pipeline="daily_inference", **summary)
    return summary


def run_training_pipeline() -> dict:
    """ingest -> validate -> train -> persist model_run.

    Reuses the same ingest/validate steps as the inference pipeline (the
    model needs to train on the same validated data inference will later
    score) but adds model training and registration on top.
    """
    init_db()
    log_event(logger, "pipeline_started", pipeline="training")

    run_ingestion_step()
    run_validation_step()

    train_models()  # writes models/fraud_model.joblib + fraud_model_metadata.json

    with open(settings.model_metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    repo.save_model_run(metadata)

    log_event(logger, "pipeline_summary", pipeline="training", model_type=metadata["model_type"])
    return {"model_type": metadata["model_type"], "model_version": f"{metadata['model_type']}@{metadata['trained_at']}"}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a FraudLens batch pipeline.")
    parser.add_argument("pipeline", choices=["training", "inference"])
    args = parser.parse_args()

    if args.pipeline == "training":
        result = run_training_pipeline()
        print(f"Training pipeline complete. Champion: {result['model_type']}")
    else:
        result = run_daily_inference_pipeline()
        print(f"Inference pipeline complete. Run IDs: {result}")


if __name__ == "__main__":
    main()
