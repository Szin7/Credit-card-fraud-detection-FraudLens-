"""Read/write access to the app database, used by both the batch pipeline
(writers) and the dashboard (readers). Plain functions rather than a
Repository class — there's no shared state to wrap, so a class would just
be ceremony around what's really a handful of independent queries.

Every function accepts an optional `session_factory` so tests can point it
at a SQLite-backed factory instead of the real Postgres database (see
tests/test_repository.py) without any other code changing.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from src.database.connection import session_scope
from src.database.models import FraudPrediction, ModelRun, PipelineRun, Transaction, ValidationResult
from src.validation.validator import ValidationSummary
from src.validation.identity import feature_digest
from src.validation.schema import INFERENCE_COLUMNS
from src.utils.json_io import json_safe

# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def start_pipeline_run(run_id: str, run_type: str, session_factory: sessionmaker | None = None) -> None:
    with session_scope(session_factory) as session:
        session.add(
            PipelineRun(
                run_id=run_id,
                run_type=run_type,
                status="running",
                started_at=datetime.now(timezone.utc),
            )
        )


def finish_pipeline_run(
    run_id: str,
    status: str,
    records_processed: int | None = None,
    records_rejected: int | None = None,
    error_message: str | None = None,
    session_factory: sessionmaker | None = None,
) -> None:
    with session_scope(session_factory) as session:
        run = session.get(PipelineRun, run_id)
        if run is None:
            raise ValueError(f"no pipeline_runs row with run_id={run_id!r} — call start_pipeline_run() first")
        run.status = status
        run.finished_at = datetime.now(timezone.utc)
        run.records_processed = records_processed
        run.records_rejected = records_rejected
        run.error_message = error_message


def save_validation_result(
    pipeline_run_id: str,
    summary: ValidationSummary,
    session_factory: sessionmaker | None = None,
) -> None:
    with session_scope(session_factory) as session:
        session.add(
            ValidationResult(
                pipeline_run_id=pipeline_run_id,
                total_rows=summary.total_rows,
                valid_rows=summary.valid_rows,
                rejected_rows=summary.rejected_rows,
                duplicate_rows=summary.duplicate_rows,
                missing_value_rows=summary.missing_value_rows,
                created_at=datetime.now(timezone.utc),
            )
        )


def _insert_once(session, table, rows, key_columns):
    if not rows:
        return 0
    dialect = session.bind.dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        raise ValueError(f"Unsupported database dialect: {dialect}")
    count = 0
    # Bound parameter counts also work on SQLite builds with a low limit.
    for start in range(0, len(rows), 50):
        statement = insert(table).values(rows[start:start + 50])
        result = session.execute(statement.on_conflict_do_nothing(index_elements=key_columns))
        count += max(result.rowcount, 0)
    return count


def _transaction_rows(df, source_run_id):
    now = datetime.now(timezone.utc)
    rows = []
    has_features = set(INFERENCE_COLUMNS).issubset(df.columns)
    for row in df.to_dict(orient="records"):
        label = row.get("Class")
        rows.append(dict(
            transaction_id=str(row["transaction_id"]),
            content_hash=feature_digest(row) if has_features else None,
            time_elapsed=float(row["Time"]), amount=float(row["Amount"]),
            actual_class=None if pd.isna(label) else int(label),
            source_run_id=source_run_id, ingested_at=now,
        ))
    return rows


def _persist_transactions(session, rows):
    inserted = _insert_once(session, Transaction.__table__, rows, ["transaction_id"])
    # An upstream ID must never silently refer to different feature values.
    for start in range(0, len(rows), 300):
        batch = rows[start:start + 300]
        stored = {r.transaction_id: r for r in session.scalars(
            select(Transaction).where(Transaction.transaction_id.in_([r["transaction_id"] for r in batch]))
        )}
        for row in batch:
            old = stored[row["transaction_id"]]
            if (float(old.time_elapsed) != row["time_elapsed"] or float(old.amount) != row["amount"]
                    or (old.content_hash and row["content_hash"] and old.content_hash != row["content_hash"])):
                raise ValueError(f"Transaction ID reused with different features: {row['transaction_id']}")
            if old.actual_class is not None and row["actual_class"] is not None and old.actual_class != row["actual_class"]:
                raise ValueError(f"Conflicting label for transaction: {row['transaction_id']}")
            if old.actual_class is None and row["actual_class"] is not None:
                old.actual_class = row["actual_class"]
    return inserted


def _prediction_rows(df):
    return [dict(
        transaction_id=str(row.transaction_id), fraud_probability=float(row.fraud_probability),
        risk_level=row.risk_level, is_flagged=bool(row.is_flagged), model_version=row.model_version,
        threshold=float(row.threshold), prediction_timestamp=pd.Timestamp(row.prediction_timestamp).to_pydatetime(),
    ) for row in df.itertuples()]


def save_transactions(df, source_run_id, session_factory=None):
    with session_scope(session_factory) as session:
        return _persist_transactions(session, _transaction_rows(df, source_run_id))


def save_predictions(df, session_factory=None):
    with session_scope(session_factory) as session:
        return _insert_once(session, FraudPrediction.__table__, _prediction_rows(df),
                            ["transaction_id", "model_version", "threshold"])


def save_scored_batch(transactions, predictions, source_run_id, session_factory=None):
    """Commit transactions and predictions together; replay is a no-op."""
    if set(transactions.transaction_id) != set(predictions.transaction_id):
        raise ValueError("Every transaction must have exactly one prediction")
    if predictions.transaction_id.duplicated().any():
        raise ValueError("Duplicate prediction IDs")
    with session_scope(session_factory) as session:
        tx_count = _persist_transactions(session, _transaction_rows(transactions, source_run_id))
        prediction_count = _insert_once(session, FraudPrediction.__table__, _prediction_rows(predictions),
                                       ["transaction_id", "model_version", "threshold"])
    return {"transactions_inserted": tx_count, "predictions_inserted": prediction_count}


def save_model_run(metadata: dict, session_factory: sessionmaker | None = None) -> None:
    """Persist a trained model's metrics/curves and mark it champion,
    unsetting any previous champion in the same transaction.
    """
    metadata = json_safe(metadata)
    with session_scope(session_factory) as session:
        session.query(ModelRun).filter(ModelRun.is_champion.is_(True)).update({"is_champion": False})

        val_metrics = metadata["validation_metrics"]
        test_metrics = metadata["test_metrics"]
        session.add(
            ModelRun(
                model_version=f"{metadata['model_type']}@{metadata['trained_at']}",
                model_type=metadata["model_type"],
                trained_at=pd.Timestamp(metadata["trained_at"]).to_pydatetime(),
                is_champion=True,
                decision_threshold=metadata["decision_threshold"],
                random_seed=metadata["random_seed"],
                precision_score=test_metrics["precision"],
                recall_score=test_metrics["recall"],
                f1_score=test_metrics["f1"],
                roc_auc=test_metrics["roc_auc"],
                pr_auc=test_metrics["pr_auc"],
                feature_columns=metadata["feature_columns"],
                curve_data={
                    "precision_recall_curve": metadata["precision_recall_curve"],
                    "roc_curve": metadata["roc_curve"],
                    "validation_metrics": val_metrics,
                    "threshold_sweep": metadata.get("threshold_sweep", []),
                    "feature_importance": metadata.get("feature_importance", {"native": [], "shap": None}),
                    "dataset_kind": metadata.get("dataset_kind", "unknown"),
                    "split_strategy": metadata.get("split_strategy", "unknown"),
                    "test_rows": metadata.get("test_rows"),
                    "test_fraud": metadata.get("test_fraud"),
                    "source_checksum_sha256": metadata.get("source_checksum_sha256"),
                    "test_confusion_matrix": {
                        "true_negatives": test_metrics["true_negatives"],
                        "false_positives": test_metrics["false_positives"],
                        "false_negatives": test_metrics["false_negatives"],
                        "true_positives": test_metrics["true_positives"],
                    },
                },
                model_comparison=metadata["model_comparison"],
                created_at=datetime.now(timezone.utc),
            )
        )


# ---------------------------------------------------------------------------
# Readers — dashboard
# ---------------------------------------------------------------------------


def get_overview_stats(session_factory: sessionmaker | None = None) -> dict:
    with session_scope(session_factory) as session:
        total = session.scalar(select(func.count()).select_from(FraudPrediction)) or 0
        flagged = session.scalar(select(func.count()).where(FraudPrediction.is_flagged.is_(True))) or 0
        avg_probability = session.scalar(select(func.avg(FraudPrediction.fraud_probability))) or 0.0
        fraud_rate = (flagged / total) if total else 0.0

    return {
        "total_transactions": int(total),
        "flagged_count": int(flagged),
        "flagged_rate": float(fraud_rate),
        "avg_probability": float(avg_probability),
    }


def get_risk_distribution(session_factory: sessionmaker | None = None) -> pd.DataFrame:
    with session_scope(session_factory) as session:
        rows = session.execute(
            select(FraudPrediction.risk_level, func.count().label("count")).group_by(FraudPrediction.risk_level)
        ).all()
    return pd.DataFrame(rows, columns=["risk_level", "count"])


def get_predictions_over_time(session_factory: sessionmaker | None = None) -> pd.DataFrame:
    """One row per transaction time-bucket (by elapsed hour, since this
    batch data has no real calendar timestamp) with prediction counts and
    average fraud probability, for the Overview page's trend chart.
    """
    with session_scope(session_factory) as session:
        rows = session.execute(
            select(
                Transaction.time_elapsed,
                FraudPrediction.fraud_probability,
                FraudPrediction.risk_level,
            ).join(FraudPrediction, FraudPrediction.transaction_id == Transaction.transaction_id)
        ).all()

    df = pd.DataFrame(rows, columns=["time_elapsed", "fraud_probability", "risk_level"])
    if df.empty:
        return df
    df["hour_bucket"] = (df["time_elapsed"] // 3600).astype(int)
    return (
        df.groupby("hour_bucket")
        .agg(transaction_count=("fraud_probability", "count"), avg_probability=("fraud_probability", "mean"))
        .reset_index()
    )


def get_recent_high_risk(limit: int = 20, session_factory: sessionmaker | None = None) -> pd.DataFrame:
    with session_scope(session_factory) as session:
        rows = session.execute(
            select(
                FraudPrediction.transaction_id,
                Transaction.amount,
                FraudPrediction.fraud_probability,
                FraudPrediction.risk_level,
                FraudPrediction.prediction_timestamp,
            )
            .join(Transaction, Transaction.transaction_id == FraudPrediction.transaction_id)
            .where(FraudPrediction.risk_level == "HIGH")
            .order_by(FraudPrediction.prediction_timestamp.desc(), FraudPrediction.fraud_probability.desc())
            .limit(limit)
        ).all()

    return pd.DataFrame(
        rows, columns=["transaction_id", "amount", "fraud_probability", "risk_level", "prediction_timestamp"]
    )


def _prediction_search_filters(
    query,
    risk_levels: list[str] | None,
    transaction_id_contains: str | None,
    min_probability: float | None,
    max_probability: float | None,
    min_amount: float | None,
    max_amount: float | None,
):
    if risk_levels:
        query = query.where(FraudPrediction.risk_level.in_(risk_levels))
    if transaction_id_contains:
        query = query.where(FraudPrediction.transaction_id.contains(transaction_id_contains))
    if min_probability is not None:
        query = query.where(FraudPrediction.fraud_probability >= min_probability)
    if max_probability is not None:
        query = query.where(FraudPrediction.fraud_probability <= max_probability)
    if min_amount is not None:
        query = query.where(Transaction.amount >= min_amount)
    if max_amount is not None:
        query = query.where(Transaction.amount <= max_amount)
    return query


def search_predictions(
    risk_levels: list[str] | None = None,
    transaction_id_contains: str | None = None,
    min_probability: float | None = None,
    max_probability: float | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    limit: int = 100,
    offset: int = 0,
    session_factory: sessionmaker | None = None,
) -> pd.DataFrame:
    """Filtered, paginated transaction+prediction listing for the
    Transaction Explorer page.
    """
    query = select(
        FraudPrediction.transaction_id,
        Transaction.amount,
        Transaction.time_elapsed,
        FraudPrediction.fraud_probability,
        FraudPrediction.risk_level,
        FraudPrediction.is_flagged,
        FraudPrediction.model_version,
        FraudPrediction.prediction_timestamp,
    ).join(Transaction, Transaction.transaction_id == FraudPrediction.transaction_id)
    query = _prediction_search_filters(
        query, risk_levels, transaction_id_contains, min_probability, max_probability, min_amount, max_amount
    )
    query = query.order_by(FraudPrediction.fraud_probability.desc()).limit(limit).offset(offset)

    with session_scope(session_factory) as session:
        rows = session.execute(query).all()

    return pd.DataFrame(
        rows,
        columns=[
            "transaction_id",
            "amount",
            "time_elapsed",
            "fraud_probability",
            "risk_level",
            "is_flagged",
            "model_version",
            "prediction_timestamp",
        ],
    )


def count_predictions(
    risk_levels: list[str] | None = None,
    transaction_id_contains: str | None = None,
    min_probability: float | None = None,
    max_probability: float | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    session_factory: sessionmaker | None = None,
) -> int:
    """Total rows matching the same filters as search_predictions(), for
    pagination controls.
    """
    query = select(func.count()).select_from(FraudPrediction).join(
        Transaction, Transaction.transaction_id == FraudPrediction.transaction_id
    )
    query = _prediction_search_filters(
        query, risk_levels, transaction_id_contains, min_probability, max_probability, min_amount, max_amount
    )
    with session_scope(session_factory) as session:
        return int(session.scalar(query) or 0)


def get_transaction_detail(transaction_id: str, session_factory: sessionmaker | None = None) -> dict | None:
    with session_scope(session_factory) as session:
        row = session.execute(
            select(
                Transaction.transaction_id,
                Transaction.amount,
                Transaction.time_elapsed,
                Transaction.actual_class,
                FraudPrediction.fraud_probability,
                FraudPrediction.risk_level,
                FraudPrediction.model_version,
                FraudPrediction.threshold,
                FraudPrediction.prediction_timestamp,
            )
            .join(FraudPrediction, FraudPrediction.transaction_id == Transaction.transaction_id)
            .where(Transaction.transaction_id == transaction_id)
            .order_by(FraudPrediction.prediction_timestamp.desc(), FraudPrediction.prediction_id.desc())
        ).first()

    return dict(row._mapping) if row is not None else None


def get_champion_model(session_factory: sessionmaker | None = None) -> dict | None:
    with session_scope(session_factory) as session:
        run = session.execute(select(ModelRun).where(ModelRun.is_champion.is_(True))).scalar_one_or_none()
        if run is None:
            return None
        return {
            "model_version": run.model_version,
            "model_type": run.model_type,
            "trained_at": run.trained_at,
            "decision_threshold": run.decision_threshold,
            "precision": run.precision_score,
            "recall": run.recall_score,
            "f1": run.f1_score,
            "roc_auc": run.roc_auc,
            "pr_auc": run.pr_auc,
            "feature_columns": run.feature_columns,
            "curve_data": run.curve_data,
            "model_comparison": run.model_comparison,
        }


def get_model_run_history(session_factory: sessionmaker | None = None) -> pd.DataFrame:
    with session_scope(session_factory) as session:
        rows = session.execute(
            select(
                ModelRun.model_version,
                ModelRun.model_type,
                ModelRun.trained_at,
                ModelRun.is_champion,
                ModelRun.precision_score,
                ModelRun.recall_score,
                ModelRun.f1_score,
                ModelRun.roc_auc,
                ModelRun.pr_auc,
            ).order_by(ModelRun.trained_at.desc())
        ).all()

    return pd.DataFrame(
        rows,
        columns=["model_version", "model_type", "trained_at", "is_champion", "precision", "recall", "f1", "roc_auc", "pr_auc"],
    )


def get_pipeline_runs(limit: int = 20, session_factory: sessionmaker | None = None) -> pd.DataFrame:
    with session_scope(session_factory) as session:
        rows = session.execute(
            select(
                PipelineRun.run_id,
                PipelineRun.run_type,
                PipelineRun.status,
                PipelineRun.started_at,
                PipelineRun.finished_at,
                PipelineRun.records_processed,
                PipelineRun.records_rejected,
                PipelineRun.error_message,
            )
            .order_by(PipelineRun.started_at.desc())
            .limit(limit)
        ).all()

    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "run_type",
            "status",
            "started_at",
            "finished_at",
            "records_processed",
            "records_rejected",
            "error_message",
        ],
    )


def get_latest_successful_ingestion(session_factory: sessionmaker | None = None) -> dict | None:
    with session_scope(session_factory) as session:
        row = session.execute(
            select(PipelineRun)
            .where(PipelineRun.run_type == "ingestion", PipelineRun.status == "succeeded")
            .order_by(PipelineRun.started_at.desc())
            .limit(1)
        ).scalar_one_or_none()

    if row is None:
        return None
    return {
        "run_id": row.run_id,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "records_processed": row.records_processed,
    }
