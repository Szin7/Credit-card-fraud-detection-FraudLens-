"""Run the existing FraudLens training pipeline on a labeled CSV.

This script works without a database or Airflow. Use it in a Kaggle working
copy of the repository or in a local Python 3.12 environment.

Example:
    python -m scripts.benchmark_real_data --input /kaggle/input/creditcardfraud/creditcard.csv

The script does not publish the trained release unless --publish is specified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from src.ingestion.ingest import ingest
from src.models.artifacts import publish_release
from src.models.train import train_release
from src.utils.config import settings
from src.utils.json_io import write_json
from src.validation.validator import run_validation


def run_benchmark(input_path: Path, *, split: str = "chronological", publish: bool = False) -> dict:
    """Validate, train and report one benchmark without touching the database."""
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    if split not in {"chronological", "stratified"}:
        raise ValueError("split must be chronological or stratified")

    # Each run has its own copy of the original file and its own outputs.
    batch_dir = settings.processed_dir / "benchmarks" / uuid4().hex
    source = ingest(input_path, require_target=True, output_dir=batch_dir)
    summary = run_validation(Path(source.source_path), require_target=True, output_dir=batch_dir)
    if summary.valid_rows == 0:
        raise ValueError("No valid labeled rows remained after validation")

    release = train_release(
        batch_dir / "transactions_valid.csv",
        strategy=split,
        dataset_kind="real",
        source_checksum=source.checksum_sha256,
    )
    metadata = json.loads((release / "metadata.json").read_text(encoding="utf-8"))
    report = {
        "dataset_kind": "real",
        "split": split,
        "source_filename": input_path.name,
        "source_sha256": source.checksum_sha256,
        "original_rows": summary.total_rows,
        "valid_rows": summary.valid_rows,
        "rejected_rows": summary.rejected_rows,
        "champion": metadata["model_type"],
        "decision_threshold": metadata["decision_threshold"],
        "validation_metrics": metadata["validation_metrics"],
        "test_metrics": metadata["test_metrics"],
        "release_directory": str(release),
        "published": bool(publish),
    }
    write_json(batch_dir / "benchmark_summary.json", report)
    if publish:
        publish_release(release)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Labeled CSV with Time, V1-V28, Amount, Class")
    parser.add_argument("--split", choices=["chronological", "stratified"], default="chronological")
    parser.add_argument("--publish", action="store_true", help="Make the trained model the active local release")
    args = parser.parse_args()
    report = run_benchmark(args.input, split=args.split, publish=args.publish)
    print("\nReal-data benchmark completed")
    print(f"Champion: {report['champion']}")
    print(f"Selected threshold: {report['decision_threshold']:.2f}")
    print(f"Holdout average precision: {report['test_metrics']['pr_auc']:.4f}")
    print(f"Release: {report['release_directory']}")
    if not report["published"]:
        print("Not published: rerun with --publish after reviewing the results.")


if __name__ == "__main__":
    main()
