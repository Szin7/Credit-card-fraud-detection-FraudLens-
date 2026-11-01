"""Release checks and repeatable promotion without deserializing a model."""

import hashlib
import json
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from sqlalchemy import create_engine, select

from scripts import deploy_real_model as deployer
from scripts import import_model_release as importer
from src.database import repository
from src.database.connection import create_session_factory, init_db, session_scope
from src.database.models import ModelRun
from src.processing.transform import FEATURE_COLUMNS


def make_archive(tmp_path, *, dataset_kind="real", split="chronological"):
    name = "a" * 32
    model = b"inert example, not a loadable joblib model"
    metrics = {
        "precision": 0.88, "recall": 0.71, "f1": 0.79,
        "roc_auc": 0.97, "pr_auc": 0.79,
        "true_negatives": 100, "false_positives": 3,
        "false_negatives": 2, "true_positives": 5,
    }
    metadata = {
        "model_type": "xgboost", "dataset_kind": dataset_kind,
        "split_strategy": split, "feature_columns": FEATURE_COLUMNS,
        "artifact_sha256": hashlib.sha256(model).hexdigest(),
        "trained_at": "2026-11-01T08:48:38+00:00",
        "decision_threshold": 0.95, "random_seed": 42,
        "source_checksum_sha256": "b" * 64,
        "validation_metrics": metrics, "test_metrics": metrics,
        "precision_recall_curve": {"precision": [1, 0.5], "recall": [0, 1]},
        "roc_curve": {"fpr": [0, 1], "tpr": [0, 1]},
        "model_comparison": [{"model": "xgboost", "pr_auc": 0.8}],
    }
    path = tmp_path / "release.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr(f"releases/{name}/metadata.json", json.dumps(metadata))
        zf.writestr(f"releases/{name}/model.joblib", model)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_check_does_not_write_model_or_touch_database(tmp_path, monkeypatch):
    archive, checksum = make_archive(tmp_path)
    monkeypatch.setattr(deployer, "init_db", lambda: pytest.fail("Unexpected DB access"))
    result = deployer.deploy(archive, checksum, check_only=True)
    assert result["status"] == "verified_only"
    assert result["model_version"].startswith("xgboost@")
    assert not (tmp_path / "models").exists()


def test_archive_rejects_wrong_sha_and_wrong_dataset(tmp_path):
    archive, checksum = make_archive(tmp_path)
    with pytest.raises(ValueError, match="Archive SHA-256"):
        deployer.inspect_archive(archive, "0" * 64)
    assert deployer.inspect_archive(archive, checksum).name == "a" * 32
    bad, bad_sha = make_archive(tmp_path, dataset_kind="synthetic")
    with pytest.raises(ValueError, match="real-data XGBoost"):
        deployer.inspect_archive(bad, bad_sha)


def test_import_and_registration_are_idempotent(tmp_path, monkeypatch):
    archive, checksum = make_archive(tmp_path)
    model_dir = tmp_path / "models"
    monkeypatch.setattr(deployer, "settings", SimpleNamespace(model_dir=model_dir))
    monkeypatch.setattr(importer, "settings", SimpleNamespace(model_dir=model_dir))

    engine = create_engine(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)
    factory = create_session_factory(engine)
    monkeypatch.setattr(deployer, "init_db", lambda: None)
    monkeypatch.setattr(deployer, "session_scope", lambda: session_scope(factory))
    original_save_model_run = repository.save_model_run
    monkeypatch.setattr(
        deployer.repository, "save_model_run",
        lambda m: original_save_model_run(m, session_factory=factory),
    )
    published = []
    monkeypatch.setattr(deployer, "publish_release", lambda path: published.append(path))

    result1 = deployer.deploy(archive, checksum)
    result2 = deployer.deploy(archive, checksum)
    assert result1 == result2
    assert len(published) == 2
    with session_scope(factory) as session:
        rows = session.scalars(select(ModelRun)).all()
        assert len(rows) == 1
        assert rows[0].is_champion is True
        assert rows[0].curve_data["dataset_kind"] == "real"


def test_conflicting_installed_model_is_not_overwritten(tmp_path, monkeypatch):
    archive, checksum = make_archive(tmp_path)
    model_dir = tmp_path / "models"
    dest = model_dir / "releases" / ("a" * 32)
    dest.mkdir(parents=True)
    (dest / "metadata.json").write_text('{"fake":true}')
    (dest / "model.joblib").write_bytes(b"different")
    monkeypatch.setattr(deployer, "settings", SimpleNamespace(model_dir=model_dir))
    monkeypatch.setattr(deployer, "init_db", lambda: pytest.fail("Unexpected DB access"))
    with pytest.raises(ValueError, match="differs"):
        deployer.deploy(archive, checksum)
