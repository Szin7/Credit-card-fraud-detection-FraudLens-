"""Sweep decision thresholds on the validation split to choose a default,
and quantify the precision/recall/cost tradeoff at each one.

Threshold selection uses the VALIDATION split, never the test split — the
test set is reserved for reporting the champion's final numbers exactly
once (see src/models/train.py). The chosen threshold is not written back
into config/settings.yaml automatically: a human reviews this output and
updates `model.decision_threshold` there, the same way a promotion gate
would in a real pipeline, rather than letting a script silently change the
production operating point.
"""

from __future__ import annotations

import joblib
import pandas as pd

from src.models.evaluate import threshold_sweep
from src.models.train import load_training_data, split_data
from src.utils.config import settings
from src.utils.logging import get_logger, log_event

logger = get_logger(__name__)


def run_threshold_analysis() -> pd.DataFrame:
    df = load_training_data()
    _, X_val, _, _, y_val, _ = split_data(df, settings.random_seed)

    pipeline = joblib.load(settings.model_path)
    val_proba = pipeline.predict_proba(X_val)[:, 1]

    sweep = threshold_sweep(y_val.to_numpy(), val_proba)
    sweep["expected_cost"] = [
        settings.cost_model.expected_cost(int(fn), int(fp))
        for fn, fp in zip(sweep["false_negatives"], sweep["false_positives"])
    ]
    return sweep


def main() -> None:
    sweep = run_threshold_analysis()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(settings.model_dir / "threshold_analysis.csv", index=False)

    best_f1_row = sweep.loc[sweep["f1"].idxmax()]
    best_cost_row = sweep.loc[sweep["expected_cost"].idxmin()]

    log_event(
        logger,
        "threshold_analysis_completed",
        best_f1_threshold=float(best_f1_row["threshold"]),
        best_f1=float(best_f1_row["f1"]),
        best_cost_threshold=float(best_cost_row["threshold"]),
        min_expected_cost=float(best_cost_row["expected_cost"]),
    )

    display_cols = ["threshold", "precision", "recall", "f1", "false_positives", "false_negatives", "expected_cost"]
    print(sweep[display_cols].to_string(index=False))
    print(f"\nBest F1 at threshold={best_f1_row['threshold']:.2f} (F1={best_f1_row['f1']:.4f})")
    print(
        f"Lowest expected cost at threshold={best_cost_row['threshold']:.2f} "
        f"(cost={best_cost_row['expected_cost']:.1f}, "
        f"FN_cost={settings.cost_model.false_negative_cost}, FP_cost={settings.cost_model.false_positive_cost})"
    )
    print(
        "\nexpected_cost uses illustrative, configurable FN/FP costs "
        "(config/settings.yaml > cost_model) to demonstrate the tradeoff - "
        "not real business figures. See docs/model_card.md."
    )


if __name__ == "__main__":
    main()
