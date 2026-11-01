"""Install a trusted Kaggle model release in the current FraudLens environment.

The model is not deserialized by this command. Inference later loads the model
under the configured, pinned Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

from filelock import FileLock
from sqlalchemy import select

from scripts.import_model_release import import_release
from src.database import repository
from src.database.connection import init_db, session_scope
from src.database.models import ModelRun
from src.models.artifacts import publish_release
from src.processing.transform import FEATURE_COLUMNS
from src.utils.config import settings


@dataclass(frozen=True)
class ReleaseInfo:
    name: str
    metadata: dict
    archive_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_archive(archive: Path, expected_sha256: str) -> ReleaseInfo:
    """Check an independently supplied ZIP digest and FraudLens release metadata."""
    archive = Path(archive).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Model archive not found: {archive}")
    expected_sha256 = expected_sha256.strip().lower()
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("An expected 64-character archive SHA-256 is required")
    archive_hash = _sha256(archive)
    if archive_hash != expected_sha256:
        raise ValueError("Archive SHA-256 does not match the expected value")

    with ZipFile(archive) as zf:
        files = [i for i in zf.infolist() if not i.is_dir()]
        if not 2 <= len(files) <= 4 or any(i.file_size > 200 * 1024 * 1024 for i in files):
            raise ValueError("Unexpected model archive contents or size")
        names = [Path(i.filename) for i in files]
        allowed = {"model.joblib", "metadata.json", "model_comparison.csv", "threshold_analysis.csv"}
        if any(len(p.parts) != 3 or p.parts[0] != "releases" or p.name not in allowed or ".." in p.parts for p in names):
            raise ValueError("Unexpected paths in model archive")
        releases = {p.parts[1] for p in names}
        if len(releases) != 1 or len({p.name for p in names}) != len(names):
            raise ValueError("Expected a single release with no duplicate files")
        name = releases.pop()
        if len(name) != 32 or any(c not in "0123456789abcdef" for c in name):
            raise ValueError("Invalid model release identifier")
        if {"model.joblib", "metadata.json"} - {p.name for p in names}:
            raise ValueError("Model or metadata missing from archive")
        if zf.testzip() is not None:
            raise ValueError("Model archive has a damaged ZIP entry")
        metadata = json.loads(zf.read(f"releases/{name}/metadata.json"))
        if not isinstance(metadata, dict):
            raise ValueError("Invalid release metadata")
        if metadata.get("model_type") != "xgboost" or metadata.get("dataset_kind") != "real":
            raise ValueError("Expected the real-data XGBoost release")
        if metadata.get("split_strategy") != "chronological":
            raise ValueError("Expected a chronological evaluation split")
        if metadata.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError("Feature schema differs from the running application")
        for field in ("trained_at", "decision_threshold", "test_metrics", "model_comparison"):
            if field not in metadata:
                raise ValueError(f"Release metadata is missing {field}")
        if not 0 < float(metadata["decision_threshold"]) <= 1:
            raise ValueError("Invalid decision threshold")
        artifact_hash = hashlib.sha256(zf.read(f"releases/{name}/model.joblib")).hexdigest()
        if metadata.get("artifact_sha256") != artifact_hash:
            raise ValueError("Model checksum does not match its metadata")
    return ReleaseInfo(name=name, metadata=metadata, archive_sha256=archive_hash)


def _existing_release_matches(release: Path, metadata: dict) -> bool:
    if not (release / "metadata.json").is_file() or not (release / "model.joblib").is_file():
        return False
    try:
        current = json.loads((release / "metadata.json").read_text())
    except (ValueError, OSError):
        return False
    return current == metadata and _sha256(release / "model.joblib") == metadata["artifact_sha256"]


def register_model(metadata: dict) -> str:
    """Idempotently mark this release as the database champion."""
    version = f"{metadata['model_type']}@{metadata['trained_at']}"
    with session_scope() as session:
        existing = session.scalar(select(ModelRun).where(ModelRun.model_version == version))
        if existing is not None:
            if (existing.model_type != metadata["model_type"]
                    or float(existing.decision_threshold) != float(metadata["decision_threshold"])
                    or existing.curve_data.get("dataset_kind") != "real"
                    or existing.curve_data.get("split_strategy") != "chronological"
                    or existing.curve_data.get("source_checksum_sha256") != metadata.get("source_checksum_sha256")
                    or list(existing.feature_columns) != list(metadata["feature_columns"])
                    or abs(float(existing.pr_auc) - float(metadata["test_metrics"]["pr_auc"])) > 1e-9):
                raise ValueError("A conflicting model version already exists in the database")
            for champion in session.scalars(select(ModelRun).where(ModelRun.is_champion.is_(True))):
                champion.is_champion = False
            session.flush()
            existing.is_champion = True
            return version
    repository.save_model_run(metadata)
    return version


def deploy(archive: Path, expected_sha256: str, *, check_only: bool = False) -> dict:
    info = inspect_archive(archive, expected_sha256)
    version = f"{info.metadata['model_type']}@{info.metadata['trained_at']}"
    if check_only:
        return {"release": info.name, "model_version": version, "archive_sha256": info.archive_sha256, "status": "verified_only"}

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(settings.model_dir / ".deployment.lock"), timeout=60):
        destination = settings.model_dir / "releases" / info.name
        if destination.exists():
            if not _existing_release_matches(destination, info.metadata):
                raise ValueError("Existing model release differs from the verified archive")
        else:
            destination = import_release(archive, publish=False)
        init_db()
        register_model(info.metadata)
        publish_release(destination)

    return {"release": info.name, "model_version": version, "archive_sha256": info.archive_sha256, "status": "active_and_registered"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--check-only", action="store_true", help="Verify archive without copying, database access or publication")
    args = parser.parse_args()
    print(json.dumps(deploy(args.archive, args.expected_archive_sha256, check_only=args.check_only), indent=2))
    print("This command did not deserialize the model. Only score with releases you created and trust.")


if __name__ == "__main__":
    main()
