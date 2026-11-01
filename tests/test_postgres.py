"""Exercise the production SQL schema and JSONB with a real PostgreSQL server."""
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy import create_engine, text

from src.database import repository as repo
from src.database.connection import create_session_factory
from src.models.evaluate import compute_metrics, roc_curve_points, precision_recall_curve_points
from src.validation.validator import validate
from tests.test_reliability import frame, predictions

pytestmark = pytest.mark.integration


@pytest.fixture
def postgres():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not configured")
    schema = "test_" + uuid4().hex
    admin = create_engine(url)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(Path("sql/schema.sql").read_text())
        yield create_session_factory(engine)
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def test_postgres_replay_and_atomic_rollback(postgres):
    repo.start_pipeline_run("batch", "validation", session_factory=postgres)
    tx, _, _ = validate(frame(), require_target=False)
    assert repo.save_scored_batch(tx, predictions(tx), "batch", postgres)["predictions_inserted"] == 3
    assert repo.save_scored_batch(tx, predictions(tx), "batch", postgres)["predictions_inserted"] == 0
    fresh = tx.copy()
    fresh["transaction_id"] += "-new"
    with pytest.raises(Exception):
        repo.save_scored_batch(fresh, predictions(fresh, probability=2), "batch", postgres)
    assert repo.get_overview_stats(postgres)["total_transactions"] == 3
    with postgres() as session:
        assert session.execute(text("SELECT count(*) FROM transactions")).scalar_one() == 3


def test_postgres_stores_real_roc_metadata_in_jsonb(postgres):
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.2, 0.9])
    metrics = compute_metrics(y, p).to_dict()
    metadata = dict(model_type="test", trained_at="2026-01-01T00:00:00+00:00",
                    decision_threshold=0.5, random_seed=42, feature_columns=["Amount"],
                    validation_metrics=metrics, test_metrics=metrics,
                    precision_recall_curve=precision_recall_curve_points(y, p),
                    roc_curve=roc_curve_points(y, p), model_comparison=[])
    repo.save_model_run(metadata, postgres)
    champion = repo.get_champion_model(postgres)
    assert champion["curve_data"]["roc_curve"]["thresholds"][0] is None
    assert champion["pr_auc"] == 1.0
