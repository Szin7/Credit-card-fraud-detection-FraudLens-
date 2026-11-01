"""Import a trusted model release exported from the FraudLens Kaggle notebook.

Checks the package structure and model checksum before copying any files.
Never unpickles the model and does not change the active pointer by default.

Example:
    python -m scripts.import_model_release --archive fraudlens-real-model-release.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from src.models.artifacts import publish_release
from src.processing.transform import FEATURE_COLUMNS
from src.utils.config import settings

ALLOWED_FILES = {"model.joblib", "metadata.json", "model_comparison.csv", "threshold_analysis.csv"}


def import_release(archive: Path, *, publish: bool = False) -> Path:
    """Install a checked release without running code from the artifact."""
    archive = Path(archive).expanduser().resolve()
    with ZipFile(archive, "r") as zf:
        entries = [i for i in zf.infolist() if not i.is_dir()]
        if len(entries) < 2:
            raise ValueError("Model archive is incomplete")
        if any(i.file_size > 200 * 1024 * 1024 for i in entries):
            raise ValueError("An archive entry is unexpectedly large")
        paths = [Path(i.filename) for i in entries]
        if any(p.is_absolute() or ".." in p.parts or len(p.parts) != 3 or p.parts[0] != "releases"
               or p.name not in ALLOWED_FILES for p in paths):
            raise ValueError("Unexpected release archive paths")
        release_names = {p.parts[1] for p in paths}
        if len(release_names) != 1:
            raise ValueError("Archive must contain exactly one model release")
        names = {p.name for p in paths}
        if len(paths) != len(names) or not {"model.joblib", "metadata.json"}.issubset(names):
            raise ValueError("Missing or duplicate release files")
        name = release_names.pop()
        if len(name) != 32 or any(ch not in "0123456789abcdef" for ch in name):
            raise ValueError("Unexpected model release identifier")
        metadata = json.loads(zf.read(f"releases/{name}/metadata.json"))
        if metadata.get("dataset_kind") != "real":
            raise ValueError("This import command is for a real-data benchmark release")
        if metadata.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError("Model feature schema does not match this checkout")
        expected = metadata.get("artifact_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Model checksum is missing or malformed")
        digest = hashlib.sha256()
        with zf.open(f"releases/{name}/model.joblib") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Model checksum does not match its metadata")

        releases = settings.model_dir / "releases"
        releases.mkdir(parents=True, exist_ok=True)
        final = releases / name
        if final.exists():
            raise FileExistsError(f"Release is already installed: {final}")
        with TemporaryDirectory(dir=releases, prefix=".import-") as tmp:
            temp = Path(tmp)
            for path in paths:
                with zf.open(path.as_posix()) as source, open(temp / path.name, "wb") as out:
                    shutil.copyfileobj(source, out)
            temp.rename(final)

    if publish:
        publish_release(final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--publish", action="store_true", help="Set the active model pointer; does not register model metrics in the database")
    args = parser.parse_args()
    release = import_release(args.archive, publish=args.publish)
    print(f"Installed release: {release}")
    print("No model was deserialized. Load only artifacts you created or trust.")
    if not args.publish:
        print("Active release unchanged. Use --publish after reviewing the model and environment.")
    else:
        print("Active release updated. Model-performance database registration is a separate step.")


if __name__ == "__main__":
    main()
