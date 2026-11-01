"""Lightweight regression checks for the database-free benchmark entry point."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import benchmark_real_data as benchmark
from src.utils.config import settings


def _fake_pipeline(tmp_path, monkeypatch):
    base = replace(settings, processed_dir=tmp_path / "processed")
    monkeypatch.setattr(benchmark, "settings", base)
    calls = {"publish": 0, "split": None}

    def fake_ingest(path, *, require_target, output_dir):
        assert require_target
        output_dir.mkdir(parents=True)
        (output_dir / "source.csv").write_text("small synthetic fixture")
        return SimpleNamespace(source_path=str(output_dir / "source.csv"), checksum_sha256="test-checksum")

    def fake_validate(path, *, require_target, output_dir):
        assert require_target
        return SimpleNamespace(total_rows=10, valid_rows=9, rejected_rows=1)

    def fake_train(path, *, strategy, dataset_kind, source_checksum):
        assert dataset_kind == "real"
        assert source_checksum == "test-checksum"
        calls["split"] = strategy
        release = tmp_path / "models" / "releases" / "test-release"
        release.mkdir(parents=True)
        (release / "metadata.json").write_text(json.dumps({
            "model_type": "logistic_regression_smote",
            "decision_threshold": 0.65,
            "validation_metrics": {"pr_auc": 0.75},
            "test_metrics": {"pr_auc": 0.70},
        }))
        return release

    monkeypatch.setattr(benchmark, "ingest", fake_ingest)
    monkeypatch.setattr(benchmark, "run_validation", fake_validate)
    monkeypatch.setattr(benchmark, "train_release", fake_train)
    monkeypatch.setattr(benchmark, "publish_release", lambda release: calls.__setitem__("publish", calls["publish"] + 1))
    return calls


def test_chronological_default_leaves_active_model_untouched(tmp_path, monkeypatch):
    calls = _fake_pipeline(tmp_path, monkeypatch)
    dataset = tmp_path / "creditcard.csv"
    dataset.write_text("a,b\n1,2\n")
    report = benchmark.run_benchmark(dataset)
    assert report["split"] == "chronological"
    assert report["dataset_kind"] == "real"
    assert report["test_metrics"]["pr_auc"] == 0.70
    assert report["published"] is False
    assert calls == {"publish": 0, "split": "chronological"}
    assert len(list((tmp_path / "processed" / "benchmarks").glob("*/benchmark_summary.json"))) == 1


def test_publication_requires_explicit_flag(tmp_path, monkeypatch):
    calls = _fake_pipeline(tmp_path, monkeypatch)
    dataset = tmp_path / "creditcard.csv"
    dataset.write_text("a,b\n1,2\n")
    report = benchmark.run_benchmark(dataset, split="stratified", publish=True)
    assert report["published"] is True
    assert calls == {"publish": 1, "split": "stratified"}


def test_missing_dataset_fails_before_ingestion(tmp_path, monkeypatch):
    calls = _fake_pipeline(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        benchmark.run_benchmark(tmp_path / "not-here.csv")
    assert calls["publish"] == 0
