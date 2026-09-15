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


def save_transactions(df: pd.DataFrame, source_run_id: str, session_factory: sessionmaker | None = None) -> int:
    """Insert one row per transaction. `df` must have transaction_id, Time,
    Amount, and Class columns (the shape of transactions_valid.csv).
    """
    now = datetime.now(timezone.utc)
    with session_scope(session_factory) as session:
        session.add_all(
            [
                Transaction(
                    transaction_id=row.transaction_id,
                    time_elapsed=float(row.Time),
                    amount=float(row.Amount),
                    actual_class=int(row.Class),
                    source_run_id=source_run_id,
                    ingested_at=now,
                )
                for row in df.itertuples()
            ]
        )
    return len(df)


def save_predictions(df: pd.DataFrame, session_factory: sessionmaker | None = None) -> int:
    """Insert one row per prediction. `df` must match the shape returned by
    src.models.predict.generate_predictions().
    """
    with session_scope(session_factory) as session:
        session.add_all(
            [
                FraudPrediction(
                    transaction_id=row.transaction_id,
                    fraud_probability=float(row.fraud_probability),
                    risk_level=row.risk_level,
                    is_flagged=bool(row.is_flagged),
                    model_version=row.model_version,
                    threshold=float(row.threshold),
                    prediction_timestamp=pd.Timestamp(row.prediction_timestamp).to_pydatetime(),
                )
                for row in df.itertuples()
            ]
        )
    return len(df)


def save_model_run(metadata: dict, session_factory: sessionmaker | None = None) -> None:
    """Persist a trained model's metrics/curves and mark it champion,
    unsetting any previous champion in the same transaction.
    """
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
