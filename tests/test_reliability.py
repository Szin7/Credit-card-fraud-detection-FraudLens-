"""Regression tests for batch identity, replay, and model metadata."""
import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select, func

from src.database import repository as repo
from src.database.connection import create_db_engine, create_session_factory, init_db
from src.database.models import Transaction, FraudPrediction
from src.models.evaluate import roc_curve_points
from src.models.train import split_data
from src.utils.json_io import write_json
from src.validation.schema import INFERENCE_COLUMNS
from src.validation.validator import validate, ValidationError


def frame(n=3, labeled=False):
    df = pd.DataFrame([{**dict.fromkeys(INFERENCE_COLUMNS, 0.1), "Time": i, "Amount": 10 + i} for i in range(n)])
    if labeled:
        df["Class"] = 0
    return df


@pytest.fixture
def database(tmp_path):
    engine = create_db_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    init_db(engine)
    factory = create_session_factory(engine)
    repo.start_pipeline_run("batch", "validation", session_factory=factory)
    yield factory
    engine.dispose()


def predictions(tx, probability=0.9):
    return pd.DataFrame({
        "transaction_id": tx.transaction_id, "fraud_probability": probability,
        "risk_level": "HIGH", "is_flagged": True, "model_version": "v1",
        "threshold": 0.6, "prediction_timestamp": "2026-01-01T00:00:00+00:00",
    })


def test_inference_accepts_unlabeled_data_but_training_requires_labels():
    valid, rejected, _ = validate(frame(), require_target=False)
    assert len(valid) == 3 and rejected.empty and valid.Class.isna().all()
    with pytest.raises(ValidationError, match="Class"):
        validate(frame())


@pytest.mark.parametrize("column,value", [("V1", np.inf), ("Amount", np.inf), ("Time", -np.inf)])
def test_nonfinite_features_are_rejected(column, value):
    df = frame()
    df[column] = df[column].astype(float)
    df.loc[0, column] = value
    valid, rejected, _ = validate(df, require_target=False)
    assert len(valid) == 2 and len(rejected) == 1


def test_fallback_ids_are_stable_across_order_and_label_changes():
    a, _, _ = validate(frame(), require_target=False)
    b, _, _ = validate(frame(labeled=True).iloc[::-1])
    assert dict(zip(a.Time, a.transaction_id)) == dict(zip(b.Time, b.transaction_id))


def test_upstream_ids_preserve_distinct_identical_transactions():
    df = pd.concat([frame(1), frame(1)], ignore_index=True)
    df["transaction_id"] = ["0001", "0002"]
    valid, _, _ = validate(df, require_target=False)
    assert list(valid.transaction_id) == ["0001", "0002"]


def test_replaying_batch_does_not_duplicate_transactions_or_predictions(database):
    tx, _, _ = validate(frame(), require_target=False)
    first = repo.save_scored_batch(tx, predictions(tx), "batch", database)
    second = repo.save_scored_batch(tx, predictions(tx), "batch", database)
    assert first == {"transactions_inserted": 3, "predictions_inserted": 3}
    assert second == {"transactions_inserted": 0, "predictions_inserted": 0}
    with database() as session:
        assert session.scalar(select(func.count()).select_from(Transaction)) == 3
        assert session.scalar(select(func.count()).select_from(FraudPrediction)) == 3


def test_failed_prediction_rolls_back_transaction_insertions(database):
    tx, _, _ = validate(frame(), require_target=False)
    with pytest.raises(Exception):
        repo.save_scored_batch(tx, predictions(tx, probability=1.5), "batch", database)
    with database() as session:
        assert session.scalar(select(func.count()).select_from(Transaction)) == 0


def test_upstream_id_cannot_silently_change_features(database):
    tx, _, _ = validate(frame(), require_target=False)
    repo.save_scored_batch(tx, predictions(tx), "batch", database)
    tx.loc[0, "V1"] = 4.5
    with pytest.raises(ValueError, match="different features"):
        repo.save_scored_batch(tx, predictions(tx), "batch", database)


def test_delayed_label_can_be_attached_without_duplicate_prediction(database):
    tx, _, _ = validate(frame(), require_target=False)
    repo.save_scored_batch(tx, predictions(tx), "batch", database)
    tx["Class"] = 1
    repo.save_scored_batch(tx, predictions(tx), "batch", database)
    with database() as session:
        assert all(row.actual_class == 1 for row in session.scalars(select(Transaction)))


def test_roc_metadata_roundtrips_as_strict_json(tmp_path):
    curve = roc_curve_points(np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.2, 0.9]))
    json.dumps(curve, allow_nan=False)
    target = tmp_path / "metadata.json"
    write_json(target, {"curve": curve, "optional": float("nan")})
    assert json.loads(target.read_text())["optional"] is None


def test_chronological_split_has_no_timestamp_overlap():
    df = frame(100, labeled=True)
    df["Class"] = np.arange(100) % 2
    train, val, test, *_ = split_data(df, 42, "chronological")
    assert train.Time.max() < val.Time.min() < test.Time.min()
    assert val.Time.max() < test.Time.min()


def test_chronological_split_refuses_a_single_class_holdout():
    df = frame(100, labeled=True)
    df.loc[:10, "Class"] = 1
    with pytest.raises(ValueError, match="both classes"):
        split_data(df, 42, "chronological")
