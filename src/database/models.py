"""SQLAlchemy ORM models mirroring sql/schema.sql.

Uses the generic `JSON` type (not Postgres-specific JSONB) so the same
models work against SQLite in tests as well as Postgres in docker-compose —
see tests/test_repository.py and src/database/connection.py.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, CheckConstraint, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    __table_args__ = (
        CheckConstraint("run_type IN ('ingestion','validation','training','inference')", name="ck_pipeline_runs_run_type"),
        CheckConstraint("status IN ('running','succeeded','failed')", name="ck_pipeline_runs_status"),
    )

    run_id: Mapped[str] = mapped_column(primary_key=True)
    run_type: Mapped[str]
    status: Mapped[str]
    started_at: Mapped[datetime]
    finished_at: Mapped[datetime | None] = mapped_column(default=None)
    records_processed: Mapped[int | None] = mapped_column(default=None)
    records_rejected: Mapped[int | None] = mapped_column(default=None)
    error_message: Mapped[str | None] = mapped_column(default=None)


class ValidationResult(Base):
    __tablename__ = "validation_results"

    validation_id: Mapped[int] = mapped_column(primary_key=True)
    pipeline_run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.run_id"))
    total_rows: Mapped[int]
    valid_rows: Mapped[int]
    rejected_rows: Mapped[int]
    duplicate_rows: Mapped[int]
    missing_value_rows: Mapped[int]
    created_at: Mapped[datetime]


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (CheckConstraint("actual_class IN (0,1)", name="ck_transactions_actual_class"),)

    transaction_id: Mapped[str] = mapped_column(primary_key=True)
    time_elapsed: Mapped[float]
    amount: Mapped[float]
    actual_class: Mapped[int | None] = mapped_column(default=None)
    source_run_id: Mapped[str | None] = mapped_column(ForeignKey("pipeline_runs.run_id"), default=None)
    ingested_at: Mapped[datetime]


class ModelRun(Base):
    """`is_champion` uniqueness (at most one true at a time) is enforced by
    a partial unique index in sql/schema.sql on the real Postgres database,
    and by repository.save_model_run() explicitly unsetting the previous
    champion inside the same transaction — not replicated as an ORM-level
    constraint here since SQLite's test database doesn't need to mirror
    every Postgres-specific index.
    """

    __tablename__ = "model_runs"

    model_run_id: Mapped[int] = mapped_column(primary_key=True)
    model_version: Mapped[str] = mapped_column(unique=True)
    model_type: Mapped[str]
    trained_at: Mapped[datetime]
    is_champion: Mapped[bool] = mapped_column(default=False)
    decision_threshold: Mapped[float]
    random_seed: Mapped[int]
    precision_score: Mapped[float]
    recall_score: Mapped[float]
    f1_score: Mapped[float]
    roc_auc: Mapped[float]
    pr_auc: Mapped[float]
    feature_columns: Mapped[list] = mapped_column(JSON)
    curve_data: Mapped[dict] = mapped_column(JSON)
    model_comparison: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[datetime]


class FraudPrediction(Base):
    __tablename__ = "fraud_predictions"
    __table_args__ = (
        CheckConstraint("fraud_probability BETWEEN 0 AND 1", name="ck_fraud_predictions_probability_range"),
        CheckConstraint("risk_level IN ('LOW','REVIEW','HIGH')", name="ck_fraud_predictions_risk_level"),
    )

    prediction_id: Mapped[int] = mapped_column(primary_key=True)
    transaction_id: Mapped[str] = mapped_column(ForeignKey("transactions.transaction_id"))
    fraud_probability: Mapped[float]
    risk_level: Mapped[str]
    is_flagged: Mapped[bool]
    model_version: Mapped[str]
    threshold: Mapped[float]
    prediction_timestamp: Mapped[datetime]
