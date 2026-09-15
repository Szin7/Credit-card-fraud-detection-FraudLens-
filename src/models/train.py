"""Train and compare fraud models on the validated transaction set.

Split strategy: stratified random train/val/test (64/16/20), not
chronological. The dataset spans only ~48 hours with no calendar/seasonal
structure, and with fraud under 1% of rows, a chronological cut risks
starving val/test of positive examples depending on which hours land where.
Stratified splitting keeps the fraud ratio stable across all three sets,
which matters more here than simulating a train-on-past/test-on-future
production setup. See docs/model_card.md for the full reasoning, and
docs/architecture.md for why this would change with a longer-history
dataset.

Imbalance handling: every model except one uses class weighting
(LogisticRegression's / RandomForest's `class_weight="balanced"`, XGBoost's
`scale_pos_weight`) — it needs no extra library and can't leak between
folds. SMOTE is compared head-to-head against class weighting on the same
base model (Logistic Regression) via an imblearn Pipeline (not sklearn's —
sklearn's Pipeline doesn't support a resampling step that changes row
counts), fit only on the training fold, so the comparison table shows
whether oversampling earns its added complexity rather than assuming it
does.

Validation selects the champion (by PR-AUC); the test set is touched
exactly once, to report the champion's final numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features.build_features import build_feature_pipeline
from src.models.evaluate import compute_metrics, precision_recall_curve_points, roc_curve_points, threshold_sweep
from src.models.explain import get_feature_importance, get_shap_importance
from src.processing.transform import FEATURE_COLUMNS, TARGET_COLUMN, load_processed_transactions
from src.utils.config import settings
from src.utils.logging import get_logger, log_event

logger = get_logger(__name__)

VAL_FRACTION_OF_REMAINDER = 0.2
TEST_FRACTION = 0.2


def load_training_data(path: Path | None = None) -> pd.DataFrame:
    return load_processed_transactions(path)


def split_data(df: pd.DataFrame, seed: int):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VAL_FRACTION_OF_REMAINDER, stratify=y_train_val, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _build_model_pipelines(seed: int, scale_pos_weight: float) -> dict:
    return {
        "dummy_baseline": Pipeline(
            [
                ("features", build_feature_pipeline()),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression_class_weight": Pipeline(
            [
                ("features", build_feature_pipeline()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
            ]
        ),
        "logistic_regression_smote": ImbPipeline(
            [
                ("features", build_feature_pipeline()),
                ("smote", SMOTE(random_state=seed)),
                ("model", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("features", build_feature_pipeline()),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced",
                        n_estimators=300,
                        max_depth=12,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("features", build_feature_pipeline()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.1,
                        scale_pos_weight=scale_pos_weight,
                        random_state=seed,
                        eval_metric="logloss",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    val_metrics: dict


def train_and_compare(seed: int = settings.random_seed):
    df = load_training_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, seed)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    log_event(
        logger,
        "training_started",
        train_rows=len(X_train),
        val_rows=len(X_val),
        test_rows=len(X_test),
        train_fraud=int(y_train.sum()),
        val_fraud=int(y_val.sum()),
        test_fraud=int(y_test.sum()),
    )

    specs = _build_model_pipelines(seed, scale_pos_weight)
    trained: list[TrainedModel] = []
    comparison_rows = []

    for name, pipeline in specs.items():
        pipeline.fit(X_train, y_train)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val.to_numpy(), val_proba, threshold=0.5)

        trained.append(TrainedModel(name=name, pipeline=pipeline, val_metrics=metrics.to_dict()))
        comparison_rows.append({"model": name, **metrics.to_dict()})

        log_event(logger, "model_trained", model=name, pr_auc=metrics.pr_auc, roc_auc=metrics.roc_auc)

    comparison_df = pd.DataFrame(comparison_rows)
    return trained, comparison_df, (X_val, y_val), (X_test, y_test)


def select_champion(trained: list[TrainedModel]) -> TrainedModel:
    """Pick the best non-dummy model by validation PR-AUC — the metric that
    stays informative under severe class imbalance, unlike accuracy.
    """
    candidates = [t for t in trained if t.name != "dummy_baseline"]
    return max(candidates, key=lambda t: t.val_metrics["pr_auc"])


def main() -> None:
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    trained, comparison_df, (X_val, y_val), (X_test, y_test) = train_and_compare()
    comparison_df.to_csv(settings.model_dir / "model_comparison.csv", index=False)

    champion = select_champion(trained)
    val_proba = champion.pipeline.predict_proba(X_val)[:, 1]
    test_proba = champion.pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test.to_numpy(), test_proba, threshold=settings.decision_threshold)

    joblib.dump(champion.pipeline, settings.model_path)

    # Threshold sweep on the validation set (never test) for the dashboard's
    # interactive threshold-tradeoff table — see scripts/optimize_threshold.py,
    # which computes the same thing standalone for a human reviewing whether
    # to change config/settings.yaml's decision_threshold.
    val_sweep = threshold_sweep(y_val.to_numpy(), val_proba)
    val_sweep["expected_cost"] = [
        settings.cost_model.expected_cost(int(fn), int(fp))
        for fn, fp in zip(val_sweep["false_negatives"], val_sweep["false_positives"])
    ]

    # Computed here (not in the dashboard) so the dashboard only ever reads
    # from Postgres and never needs the model file, ML libraries, or
    # data/processed/ mounted into its own container.
    native_importance = get_feature_importance(champion.pipeline).to_dict(orient="records")
    shap_importance_df = get_shap_importance(champion.pipeline, X_val)
    shap_importance = shap_importance_df.to_dict(orient="records") if shap_importance_df is not None else None

    metadata = {
        "model_type": champion.name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": settings.random_seed,
        "decision_threshold": settings.decision_threshold,
        "feature_columns": FEATURE_COLUMNS,
        "validation_metrics": champion.val_metrics,
        "test_metrics": test_metrics.to_dict(),
        "precision_recall_curve": precision_recall_curve_points(y_test.to_numpy(), test_proba),
        "roc_curve": roc_curve_points(y_test.to_numpy(), test_proba),
        "threshold_sweep": val_sweep.to_dict(orient="records"),
        "model_comparison": comparison_df.to_dict(orient="records"),
        "feature_importance": {"native": native_importance, "shap": shap_importance},
    }
    with open(settings.model_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log_event(
        logger,
        "training_completed",
        champion=champion.name,
        test_pr_auc=test_metrics.pr_auc,
        test_roc_auc=test_metrics.roc_auc,
    )

    print(comparison_df.to_string(index=False))
    print(f"\nChampion (by validation PR-AUC): {champion.name}")
    print(f"Test PR-AUC: {test_metrics.pr_auc:.4f}   Test ROC-AUC: {test_metrics.roc_auc:.4f}")
    print(f"Test precision: {test_metrics.precision:.4f}   Test recall: {test_metrics.recall:.4f}")


if __name__ == "__main__":
    main()
