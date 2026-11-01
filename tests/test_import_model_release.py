"""Check that model imports reject corrupt and unexpected packages."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from scripts import import_model_release as importer
from src.processing.transform import FEATURE_COLUMNS


def archive(tmp_path, *, corrupt=False, entry_name="model.joblib"):
    name = "a" * 32
    artifact = b"inert model bytes, never deserialized during import"
    data = {
        "dataset_kind": "real",
        "feature_columns": FEATURE_COLUMNS,
        "artifact_sha256": hashlib.sha256(artifact if not corrupt else b"different").hexdigest(),
    }
    result = tmp_path / "release.zip"
    with ZipFile(result, "w", ZIP_DEFLATED) as zf:
        zf.writestr(f"releases/{name}/metadata.json", json.dumps(data))
        zf.writestr(f"releases/{name}/{entry_name}", artifact)
    return result


def test_import_does_not_activate_without_explicit_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "settings", SimpleNamespace(model_dir=tmp_path / "models"))
    promoted = []
    monkeypatch.setattr(importer, "publish_release", lambda path: promoted.append(path))
    release = importer.import_release(archive(tmp_path))
    assert (release / "model.joblib").exists()
    assert (release / "metadata.json").exists()
    assert promoted == []
    assert not (tmp_path / "models" / "active_model.json").exists()


def test_rejects_mismatched_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "settings", SimpleNamespace(model_dir=tmp_path / "models"))
    with pytest.raises(ValueError, match="checksum"):
        importer.import_release(archive(tmp_path, corrupt=True))
    assert not (tmp_path / "models").exists()


def test_rejects_unexpected_archive_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "settings", SimpleNamespace(model_dir=tmp_path / "models"))
    with pytest.raises(ValueError, match="Unexpected release archive paths"):
        importer.import_release(archive(tmp_path, entry_name="../model.joblib"))
