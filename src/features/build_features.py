"""Reusable feature preprocessing, shared by training and inference.

Built as a single sklearn ColumnTransformer rather than ad hoc pandas code
so the exact same fitted object runs at both training and prediction time —
that's what prevents training/serving skew (e.g. Amount being scaled with
one mean/std during training and a different one at inference).

V1-V28 are already PCA-anonymized (real dataset) or synthetic stand-ins for
that (sample dataset) — they're passed through unchanged. This module never
assigns them semantic meaning.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.validation.schema import V_FEATURE_COLUMNS

SECONDS_PER_DAY = 86_400
HOURS_PER_DAY = 24


def _cyclical_hour_of_day(df):
    """Fold elapsed-seconds `Time` into an hour-of-day, then encode it as
    sin/cos so hour 23 and hour 0 sit next to each other in feature space
    instead of at opposite ends of a raw 0-23 scale.
    """
    hour = (df["Time"].to_numpy() % SECONDS_PER_DAY) / 3600.0
    radians = 2 * np.pi * hour / HOURS_PER_DAY
    return np.column_stack([np.sin(radians), np.cos(radians)])


def _cyclical_hour_feature_names(transformer, input_features=None) -> list[str]:
    return ["time_hour_sin", "time_hour_cos"]


def build_feature_pipeline() -> ColumnTransformer:
    """Build the (unfitted) feature ColumnTransformer.

    - Amount: log1p (transaction amounts are heavily right-skewed) then
      standard-scaled.
    - Time: cyclical hour-of-day encoding.
    - V1-V28: passthrough.

    Call `.fit()` on this once, during training; the fitted object is
    persisted as part of the model pipeline and reused unchanged for every
    later prediction.
    """
    amount_pipeline = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )

    time_cyclical = FunctionTransformer(
        _cyclical_hour_of_day,
        feature_names_out=_cyclical_hour_feature_names,
    )

    transformer = ColumnTransformer(
        transformers=[
            ("amount", amount_pipeline, ["Amount"]),
            ("time_cyclical", time_cyclical, ["Time"]),
            ("pca_features", "passthrough", V_FEATURE_COLUMNS),
        ],
        remainder="drop",
    )
    transformer.set_output(transform="pandas")
    return transformer
