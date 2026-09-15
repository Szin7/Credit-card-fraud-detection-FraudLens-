import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.build_features import build_feature_pipeline
from src.models import predict as predict_module
from src.models.evaluate import compute_metrics
from src.utils.config import RiskBands, settings
from src.validation.schema import V_FEATURE_COLUMNS


def test_risk_band_boundaries_are_half_open_low_and_review():
    bands = RiskBands(low_max=0.30, review_max=0.70)

    assert bands.classify(0.0) == "LOW"
    assert bands.classify(0.2999) == "LOW"
    assert bands.classify(0.30) == "REVIEW"
    assert bands.classify(0.6999) == "REVIEW"
    assert bands.classify(0.70) == "HIGH"
    assert bands.classify(1.0) == "HIGH"


def test_compute_metrics_on_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.9, 0.95])
    metrics = compute_metrics(y_true, y_proba, threshold=0.5)

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert (metrics.true_positives, metrics.true_negatives) == (2, 2)
    assert (metrics.false_positives, metrics.false_negatives) == (0, 0)


def test_compute_metrics_confusion_counts():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.3, 0.8])  # threshold 0.5 -> preds [0,1,0,0,1]

    metrics = compute_metrics(y_true, y_proba, threshold=0.5)

    assert metrics.true_negatives == 2  # indices 0, 2
    assert metrics.false_positives == 1  # index 1
    assert metrics.false_negatives == 1  # index 3
    assert metrics.true_positives == 1  # index 4


def _tiny_fitted_pipeline() -> tuple[Pipeline, pd.DataFrame]:
    rng = np.random.default_rng(1)
    n = 40
    data = {col: rng.normal(size=n) for col in V_FEATURE_COLUMNS}
    data["Time"] = rng.uniform(0, 172_792, size=n)
    data["Amount"] = rng.lognormal(mean=3.0, sigma=1.0, size=n)
    X = pd.DataFrame(data)
    y = pd.Series(rng.integers(0, 2, size=n))

    pipeline = Pipeline([("features", build_feature_pipeline()), ("model", LogisticRegression(max_iter=200))])
    pipeline.fit(X, y)
    return pipeline, X


def test_generate_predictions_output_schema(monkeypatch):
    pipeline, X = _tiny_fitted_pipeline()
    monkeypatch.setattr(predict_module, "load_model", lambda: pipeline)
    monkeypatch.setattr(predict_module, "load_model_version", lambda: "test-model@2024-01-01")

    df = X.copy()
    df.insert(0, "transaction_id", [f"tx-{i}" for i in range(len(df))])

    predictions = predict_module.generate_predictions(df=df)

    expected_columns = {
        "transaction_id",
        "amount",
        "time_elapsed",
        "fraud_probability",
        "risk_level",
        "is_flagged",
        "model_version",
        "threshold",
        "prediction_timestamp",
    }
    assert expected_columns.issubset(predictions.columns)
    assert len(predictions) == len(df)
    assert predictions["fraud_probability"].between(0, 1).all()
    assert set(predictions["risk_level"]) <= {"LOW", "REVIEW", "HIGH"}
    assert predictions["model_version"].eq("test-model@2024-01-01").all()


def test_is_flagged_matches_configured_decision_threshold(monkeypatch):
    pipeline, X = _tiny_fitted_pipeline()
    monkeypatch.setattr(predict_module, "load_model", lambda: pipeline)
    monkeypatch.setattr(predict_module, "load_model_version", lambda: "test-model")

    df = X.copy()
    df.insert(0, "transaction_id", [f"tx-{i}" for i in range(len(df))])

    predictions = predict_module.generate_predictions(df=df)

    expected_flags = predictions["fraud_probability"] >= settings.decision_threshold
    pd.testing.assert_series_equal(predictions["is_flagged"], expected_flags, check_names=False)
