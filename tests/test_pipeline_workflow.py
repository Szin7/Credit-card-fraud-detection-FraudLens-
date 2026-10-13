"""Exercise persisted workflows without relying on a prebuilt model artifact."""
from dataclasses import replace
import importlib

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from scripts import run_pipeline
from scripts.generate_sample_data import generate_transactions
from src.database import connection, repository
from src.features.build_features import build_feature_pipeline
from src.models import train
from src.models.artifacts import load_bundle
from src.utils.config import settings


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    configured = replace(settings, raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed",
                         rejected_dir=tmp_path / "rejected", model_dir=tmp_path / "models",
                         database_url=f"sqlite:///{tmp_path / 'workflow.db'}")
    for module in ["scripts.run_pipeline", "src.ingestion.ingest", "src.validation.validator",
                   "src.models.train", "src.models.predict", "src.models.artifacts",
                   "src.database.connection", "src.processing.transform"]:
        monkeypatch.setattr(importlib.import_module(module), "settings", configured)
    monkeypatch.setattr(connection, "_default_session_factory", None)
    monkeypatch.setattr(connection, "_default_engine", None)
    # Keep this regression test quick; the full five-candidate benchmark is a separate run.
    monkeypatch.setattr(train, "_build_model_pipelines", lambda seed, weight: {
        "dummy_baseline": Pipeline([("features", build_feature_pipeline()), ("model", DummyClassifier())]),
        "logistic_regression": Pipeline([("features", build_feature_pipeline()), ("model", LogisticRegression(max_iter=1000))]),
    })
    yield configured
    if connection._default_engine is not None:
        connection._default_engine.dispose()


def test_train_score_replay_and_failure_status(workspace, tmp_path):
    sample = generate_transactions()
    sample = sample.groupby("Class").head(100).copy()
    labeled = tmp_path / "train.csv"
    sample.to_csv(labeled, index=False)
    result = run_pipeline.run_training_pipeline(labeled, dataset_kind="synthetic")
    assert result["model_type"] == "logistic_regression"
    _, metadata = load_bundle()
    assert metadata["dataset_kind"] == "synthetic"
    assert metadata["source_checksum_sha256"]
    unlabeled = tmp_path / "incoming.csv"
    sample.drop(columns=["Class"]).to_csv(unlabeled, index=False)
    assert run_pipeline.run_daily_inference_pipeline(unlabeled)["predictions_inserted"] == len(sample)
    assert run_pipeline.run_daily_inference_pipeline(unlabeled)["predictions_inserted"] == 0
    with pytest.raises(Exception, match="not found"):
        run_pipeline.run_daily_inference_pipeline(tmp_path / "missing.csv")
    runs = repository.get_pipeline_runs(limit=100)
    assert "failed" in set(runs.status)
    assert "running" not in set(runs.status)
    assert "training" in set(runs.run_type)
