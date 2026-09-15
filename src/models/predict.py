"""Run the trained model on validated transactions and produce fraud
probabilities, risk levels, and prediction metadata ready to persist.

`risk_level` (LOW/REVIEW/HIGH) and `is_flagged` are deliberately different
concepts: risk_level is a triage category for a human reviewer (configured
via config/settings.yaml > risk_bands), while is_flagged is the actual
operating decision at the tuned decision threshold (config > model >
decision_threshold, chosen by scripts/optimize_threshold.py). A transaction
can show as HIGH risk without crossing the decision threshold, and vice
versa near the boundary — the dashboard and any downstream action should
use whichever one is actually appropriate for that context.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.utils.config import settings
from src.utils.logging import get_logger, log_event
from src.validation.schema import V_FEATURE_COLUMNS

logger = get_logger(__name__)

FEATURE_COLUMNS = ["Time", "Amount", *V_FEATURE_COLUMNS]
PREDICTIONS_FILENAME = "predictions.csv"


def load_model():
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"no trained model at {settings.model_path} - run `python -m src.models.train` first"
        )
    return joblib.load(settings.model_path)


def load_model_version() -> str:
    if not settings.model_metadata_path.exists():
        return "unknown"
    with open(settings.model_metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    return f"{metadata['model_type']}@{metadata['trained_at']}"


def generate_predictions(df: pd.DataFrame | None = None, input_path: Path | None = None) -> pd.DataFrame:
    """One row per input transaction: transaction_id, fraud_probability,
    risk_level, is_flagged, model_version, threshold, prediction_timestamp.
    """
    if df is None:
        path = input_path or (settings.processed_dir / "transactions_valid.csv")
        df = pd.read_csv(path)

    pipeline = load_model()
    model_version = load_model_version()

    fraud_probability = pipeline.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    threshold = settings.decision_threshold

    predictions = pd.DataFrame(
        {
            "transaction_id": df["transaction_id"],
            "amount": df["Amount"],
            "time_elapsed": df["Time"],
            "fraud_probability": fraud_probability,
            "risk_level": [settings.risk_bands.classify(float(p)) for p in fraud_probability],
            "is_flagged": fraud_probability >= threshold,
            "model_version": model_version,
            "threshold": threshold,
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    log_event(
        logger,
        "prediction_completed",
        row_count=len(predictions),
        model_version=model_version,
        flagged_count=int(predictions["is_flagged"].sum()),
        high_risk_count=int((predictions["risk_level"] == "HIGH").sum()),
    )

    return predictions


def main() -> None:
    predictions = generate_predictions()

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.processed_dir / PREDICTIONS_FILENAME
    predictions.to_csv(output_path, index=False)

    print(f"Wrote {len(predictions)} predictions to {output_path}")
    print(predictions["risk_level"].value_counts().rename("count"))
    print(f"Flagged (>= threshold {settings.decision_threshold}): {int(predictions['is_flagged'].sum())}")


if __name__ == "__main__":
    main()
