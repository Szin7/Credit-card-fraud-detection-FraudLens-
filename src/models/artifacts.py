"""Immutable model releases and an atomically published active pointer."""

import hashlib
import json
from pathlib import Path

import joblib

from src.utils.config import settings
from src.utils.json_io import write_json


def resolve_release(model_dir: Path | None = None) -> Path:
    root = model_dir or settings.model_dir
    pointer = root / "active_model.json"
    if not pointer.exists():
        raise FileNotFoundError("No active model. Run the training pipeline first.")
    release_name = json.loads(pointer.read_text())["release"]
    if Path(release_name).name != release_name:
        raise ValueError("Invalid model release path")
    return root / "releases" / release_name


def publish_release(release: Path, model_dir: Path | None = None) -> None:
    root = model_dir or settings.model_dir
    if release.parent.resolve() != (root / "releases").resolve():
        raise ValueError("Release must be inside the configured model directory")
    if not (release / "model.joblib").is_file() or not (release / "metadata.json").is_file():
        raise ValueError("Incomplete model release")
    write_json(root / "active_model.json", {"release": release.name})


def load_bundle(release: Path | None = None):
    """Resolve the pointer once so a concurrent promotion cannot mix versions.

    Only load joblib artifacts produced by a trusted local training run.
    """
    release = release or resolve_release()
    metadata = json.loads((release / "metadata.json").read_text())
    model_path = release / "model.joblib"
    expected = metadata.get("artifact_sha256")
    if expected and hashlib.sha256(model_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Model artifact checksum does not match metadata")
    return joblib.load(model_path), metadata
