from datetime import datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine, select

from src.database import repository as repo
from src.database.connection import create_session_factory, init_db, session_scope
from src.database.models import ValidationResult
from src.validation.validator import ValidationSummary


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return create_session_factory(engine)


def _sample_transactions(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"tx-{i}" for i in range(n)],
            "Time": [100.0 * i for i in range(n)],
            "Amount": [10.0 * (i + 1) for i in range(n)],
            "Class": [1 if i == 0 else 0 for i in range(n)],
        }
    )


def _sample_predictions(n: int = 5) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    probabilities = [0.9, 0.5, 0.1, 0.05, 0.8]
    risk_levels = ["HIGH", "REVIEW", "LOW", "LOW", "HIGH"]
    return pd.DataFrame(
        {
            "transaction_id": [f"tx-{i}" for i in range(n)],
            "fraud_probability": probabilities[:n],
            "risk_level": risk_levels[:n],
            "is_flagged": [p >= 0.6 for p in probabilities[:n]],
            "model_version": ["test_model@v1"] * n,
            "threshold": [0.6] * n,
            "prediction_timestamp": [now] * n,
        }
    )


def test_pipeline_run_lifecycle(session_factory):
    repo.start_pipeline_run("run-1", "ingestion", session_factory=session_factory)
    repo.finish_pipeline_run(
        "run-1", "succeeded", records_processed=100, records_rejected=2, session_factory=session_factory
    )

    runs = repo.get_pipeline_runs(session_factory=session_factory)
    assert len(runs) == 1
    assert runs.iloc[0]["status"] == "succeeded"
    assert runs.iloc[0]["records_processed"] == 100


def test_finish_pipeline_run_without_start_raises(session_factory):
    with pytest.raises(ValueError):
        repo.finish_pipeline_run("missing-run", "succeeded", session_factory=session_factory)


def test_save_validation_result_persists_summary(session_factory):
    repo.start_pipeline_run("run-2", "validation", session_factory=session_factory)
    summary = ValidationSummary(
        total_rows=10,
        valid_rows=8,
        rejected_rows=2,
        duplicate_rows=1,
        missing_value_rows=1,
        invalid_class_rows=0,
        negative_amount_rows=0,
        invalid_time_rows=0,
        non_numeric_feature_rows=0,
        unexpected_columns=[],
    )
    repo.save_validation_result("run-2", summary, session_factory=session_factory)

    with session_scope(session_factory) as session:
        stored = session.execute(select(ValidationResult)).scalar_one()
        assert stored.total_rows == 10
        assert stored.valid_rows == 8
        assert stored.duplicate_rows == 1


def test_save_transactions_and_predictions_feed_overview_stats(session_factory):
    repo.start_pipeline_run("run-3", "inference", session_factory=session_factory)
    repo.save_transactions(_sample_transactions(), source_run_id="run-3", session_factory=session_factory)
    repo.save_predictions(_sample_predictions(), session_factory=session_factory)

    stats = repo.get_overview_stats(session_factory=session_factory)
    assert stats["total_transactions"] == 5
    assert stats["flagged_count"] == 2  # probabilities 0.9 and 0.8 are >= 0.6

    distribution = repo.get_risk_distribution(session_factory=session_factory)
    counts = dict(zip(distribution["risk_level"], distribution["count"]))
    assert counts == {"HIGH": 2, "REVIEW": 1, "LOW": 2}


def test_search_predictions_filters_by_risk_and_probability(session_factory):
    repo.start_pipeline_run("run-4", "inference", session_factory=session_factory)
    repo.save_transactions(_sample_transactions(), source_run_id="run-4", session_factory=session_factory)
    repo.save_predictions(_sample_predictions(), session_factory=session_factory)

    high_only = repo.search_predictions(risk_levels=["HIGH"], session_factory=session_factory)
    assert set(high_only["transaction_id"]) == {"tx-0", "tx-4"}

    above_threshold = repo.search_predictions(min_probability=0.6, session_factory=session_factory)
    assert len(above_threshold) == 2

    assert repo.count_predictions(risk_levels=["HIGH"], session_factory=session_factory) == 2
    assert repo.count_predictions(min_probability=0.6, session_factory=session_factory) == 2


def test_get_transaction_detail_joins_transaction_and_prediction(session_factory):
    repo.start_pipeline_run("run-5", "inference", session_factory=session_factory)
    repo.save_transactions(_sample_transactions(), source_run_id="run-5", session_factory=session_factory)
    repo.save_predictions(_sample_predictions(), session_factory=session_factory)

    detail = repo.get_transaction_detail("tx-0", session_factory=session_factory)
    assert detail is not None
    assert detail["risk_level"] == "HIGH"
    assert detail["amount"] == 10.0

    assert repo.get_transaction_detail("does-not-exist", session_factory=session_factory) is None


def test_save_model_run_marks_champion_and_replaces_previous(session_factory):
    base_metadata = {
        "model_type": "logistic_regression",
        "trained_at": "2026-01-01T00:00:00+00:00",
        "decision_threshold": 0.5,
        "random_seed": 42,
        "feature_columns": ["Time", "Amount"],
        "validation_metrics": {"precision": 0.8, "recall": 0.7},
        "test_metrics": {
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "roc_auc": 0.95,
            "pr_auc": 0.6,
            "true_negatives": 100,
            "false_positives": 5,
            "false_negatives": 3,
            "true_positives": 7,
        },
        "precision_recall_curve": {"precision": [1.0, 0.5], "recall": [0.0, 1.0], "thresholds": [0.9, 0.1]},
        "roc_curve": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [0.9, 0.1]},
        "model_comparison": [{"model": "logistic_regression", "pr_auc": 0.6}],
    }
    repo.save_model_run(base_metadata, session_factory=session_factory)

    newer_metadata = {**base_metadata, "model_type": "xgboost", "trained_at": "2026-02-01T00:00:00+00:00"}
    repo.save_model_run(newer_metadata, session_factory=session_factory)

    champion = repo.get_champion_model(session_factory=session_factory)
    assert champion["model_type"] == "xgboost"

    history = repo.get_model_run_history(session_factory=session_factory)
    assert len(history) == 2
    assert history.loc[history["model_type"] == "logistic_regression", "is_champion"].iloc[0] == False  # noqa: E712
