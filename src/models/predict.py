"""Score an unlabeled batch with one immutable model release."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.processing.transform import FEATURE_COLUMNS, load_processed_transactions
from src.utils.config import settings, RiskBands
from src.models.artifacts import load_bundle
from src.utils.logging import get_logger, log_event

logger = get_logger(__name__)

PREDICTIONS_FILENAME = "predictions.csv"


def generate_predictions(df: pd.DataFrame | None = None, input_path: Path | None = None) -> pd.DataFrame:
    """One row per input transaction: transaction_id, fraud_probability,
    risk_level, is_flagged, model_version, threshold, prediction_timestamp.
    """
    if df is None:
        df = load_processed_transactions(input_path)

    pipeline, metadata = load_bundle()
    model_version = f"{metadata['model_type']}@{metadata['trained_at']}"
    bands = RiskBands(**metadata.get("risk_bands", {"low_max": settings.risk_bands.low_max, "review_max": settings.risk_bands.review_max}))

    fraud_probability = pipeline.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    threshold = float(metadata["decision_threshold"])

    predictions = pd.DataFrame(
        {
            "transaction_id": df["transaction_id"],
            "amount": df["Amount"],
            "time_elapsed": df["Time"],
            "fraud_probability": fraud_probability,
            "risk_level": [bands.classify(float(p)) for p in fraud_probability],
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
