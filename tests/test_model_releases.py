import hashlib
import json

import joblib
import pytest

from src.models import artifacts
from src.utils.json_io import write_json


def release(root, name):
    folder = root / "releases" / name
    folder.mkdir(parents=True)
    joblib.dump({"test_artifact": name}, folder / "model.joblib")
    write_json(folder / "metadata.json", {
        "model_type": name,
        "artifact_sha256": hashlib.sha256((folder / "model.joblib").read_bytes()).hexdigest(),
    })
    return folder


def test_publish_rejects_partial_release_and_preserves_active_version(tmp_path):
    old = release(tmp_path, "old")
    artifacts.publish_release(old, tmp_path)
    incomplete = tmp_path / "releases" / "incomplete"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="Incomplete"):
        artifacts.publish_release(incomplete, tmp_path)
    assert artifacts.resolve_release(tmp_path) == old


def test_model_checksum_is_checked_before_deserialization(tmp_path):
    folder = release(tmp_path, "one")
    (folder / "model.joblib").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        artifacts.load_bundle(folder)


def test_new_pointer_does_not_change_a_resolved_release(tmp_path):
    old = release(tmp_path, "old")
    new = release(tmp_path, "new")
    artifacts.publish_release(old, tmp_path)
    resolved = artifacts.resolve_release(tmp_path)
    artifacts.publish_release(new, tmp_path)
    model, metadata = artifacts.load_bundle(resolved)
    assert model["test_artifact"] == metadata["model_type"] == "old"
    assert artifacts.resolve_release(tmp_path) == new


def test_pointer_cannot_escape_release_directory(tmp_path):
    (tmp_path / "active_model.json").write_text(json.dumps({"release": "../../external"}))
    with pytest.raises(ValueError, match="Invalid"):
        artifacts.resolve_release(tmp_path)
