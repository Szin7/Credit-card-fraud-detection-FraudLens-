"""Ingestion: load an incoming transaction CSV, confirm it has the columns
the rest of the pipeline expects, and record what was ingested.

Ingestion only checks structure (right columns present, file parses, not
empty) — row-level rules (value ranges, duplicates, types) are the
validation module's job. Splitting these avoids re-reading the file twice
for two halves of the same concern.

"Preserving raw data" means different things per storage mode: in local
mode the file already sits in data/raw/ untouched by every later stage, so
ingestion just records a manifest proving what was ingested and when; in s3
mode, where the source may be a transient landing zone, ingestion uploads
the file to S3 as the immutable copy of record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.utils.config import settings
from src.utils.logging import get_logger, log_event
from src.validation.schema import REQUIRED_COLUMNS

logger = get_logger(__name__)


class IngestionError(Exception):
    """Raised when the input file can't be ingested at all (missing, empty,
    unparseable, or structurally wrong) — distinct from row-level rejects,
    which validation handles without failing the whole batch.
    """


@dataclass
class IngestionResult:
    run_id: str
    source_path: str
    storage_mode: str
    row_count: int
    column_count: int
    checksum_sha256: str
    ingested_at: str
    manifest_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise IngestionError(f"could not parse '{path}' as CSV: {exc}") from exc


def ingest(input_path: Path | None = None) -> IngestionResult:
    path = Path(input_path) if input_path is not None else settings.raw_data_path

    log_event(logger, "pipeline_started", stage="ingestion", source_path=str(path))

    if not path.exists():
        raise IngestionError(f"input file not found: {path}")
    if path.stat().st_size == 0:
        raise IngestionError(f"input file is empty: {path}")

    df = _read_csv(path)
    if df.empty:
        raise IngestionError(f"input file has no rows: {path}")

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise IngestionError(f"input file is missing required columns: {missing_columns}")

    run_id = uuid4().hex
    ingested_at = datetime.now(timezone.utc).isoformat()
    checksum = _checksum(path)

    if settings.storage_mode == "s3":
        from src.storage.s3 import upload_file

        upload_file(path, f"raw/{run_id}/{path.name}")

    manifests_dir = settings.raw_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"{run_id}.json"

    result = IngestionResult(
        run_id=run_id,
        source_path=str(path),
        storage_mode=settings.storage_mode,
        row_count=len(df),
        column_count=len(df.columns),
        checksum_sha256=checksum,
        ingested_at=ingested_at,
        manifest_path=str(manifest_path),
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    log_event(
        logger,
        "file_ingested",
        run_id=run_id,
        row_count=result.row_count,
        storage_mode=result.storage_mode,
        checksum_sha256=checksum,
    )

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a raw transaction CSV.")
    parser.add_argument("--input", type=Path, default=None, help="Path to the CSV file to ingest.")
    args = parser.parse_args()

    try:
        result = ingest(args.input)
    except IngestionError:
        log_event(logger, "pipeline_failed", stage="ingestion")
        raise

    print(f"Ingested {result.row_count} rows from {result.source_path} (run_id={result.run_id})")


if __name__ == "__main__":
    main()
