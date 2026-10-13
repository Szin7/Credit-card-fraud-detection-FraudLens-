"""Run a batch from an isolated input snapshot through tracked stages."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from src.database import repository as repo
from src.database.connection import init_db
from src.ingestion.ingest import ingest
from src.models.artifacts import publish_release
from src.models.predict import generate_predictions
from src.models.train import train_release
from src.processing.transform import load_processed_transactions
from src.utils.config import settings
from src.validation.validator import run_validation


@contextmanager
def tracked_stage(stage):
    run_id = uuid4().hex
    counts = {}
    repo.start_pipeline_run(run_id, stage)
    try:
        yield run_id, counts
    except Exception as exc:
        repo.finish_pipeline_run(run_id, "failed", error_message=str(exc))
        raise
    else:
        repo.finish_pipeline_run(run_id, "succeeded", **counts)


def prepare_batch(input_path=None, *, require_target=False):
    batch_dir = settings.processed_dir / "runs" / uuid4().hex
    with tracked_stage("ingestion") as (_, counts):
        source = ingest(input_path, require_target=require_target, output_dir=batch_dir)
        counts["records_processed"] = source.row_count
    with tracked_stage("validation") as (validation_id, counts):
        summary = run_validation(Path(source.source_path), require_target=require_target, output_dir=batch_dir)
        repo.save_validation_result(validation_id, summary)
        counts.update(records_processed=summary.valid_rows, records_rejected=summary.rejected_rows)
        if not summary.valid_rows:
            raise ValueError("No valid transactions remain after validation")
    return batch_dir, validation_id, source


def run_daily_inference_pipeline(input_path=None):
    init_db()
    batch_dir, validation_id, _ = prepare_batch(input_path, require_target=False)
    with tracked_stage("inference") as (run_id, counts):
        transactions = load_processed_transactions(batch_dir / "transactions_valid.csv")
        predictions = generate_predictions(transactions)
        result = repo.save_scored_batch(transactions, predictions, validation_id)
        predictions.to_csv(batch_dir / "predictions.csv", index=False)
        counts["records_processed"] = len(predictions)
    return {"inference_run_id": run_id, "batch_directory": str(batch_dir), **result}


def run_training_pipeline(input_path=None, *, strategy="stratified", dataset_kind="unknown"):
    from filelock import FileLock

    init_db()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    # Cross-process lock serializes promotions on this single-host deployment.
    with FileLock(str(settings.model_dir / ".training.lock"), timeout=0):
        batch_dir, _, source = prepare_batch(input_path, require_target=True)
        with tracked_stage("training") as (run_id, counts):
            valid_path = batch_dir / "transactions_valid.csv"
            release = train_release(valid_path, strategy=strategy, dataset_kind=dataset_kind,
                                    source_checksum=source.checksum_sha256)
            metadata = json.loads((release / "metadata.json").read_text())
            repo.save_model_run(metadata)
            publish_release(release)
            counts["records_processed"] = len(load_processed_transactions(valid_path))
    return {"training_run_id": run_id, "model_type": metadata["model_type"], "release": str(release)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", choices=["training", "inference"])
    parser.add_argument("--input", type=Path)
    parser.add_argument("--split", choices=["stratified", "chronological"], default="stratified")
    parser.add_argument("--dataset-kind", choices=["synthetic", "real", "unknown"], default="unknown")
    args = parser.parse_args()
    if args.pipeline == "training":
        result = run_training_pipeline(args.input, strategy=args.split, dataset_kind=args.dataset_kind)
    else:
        result = run_daily_inference_pipeline(args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
