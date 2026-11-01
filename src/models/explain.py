"""Lightweight model explainability: which model inputs most influenced
predictions — not what those inputs "mean" in the real world. V1-V28 are
anonymized in both the real and synthetic datasets, and this module never
assigns them invented semantic labels (no "merchant", "device", "country").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def get_feature_importance(pipeline: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """Native feature importance: absolute coefficients for linear models,
    impurity-based importance for tree ensembles. Always available, unlike
    SHAP, which needs a model type it explicitly supports.
    """
    features_step = pipeline.named_steps["features"]
    feature_names = list(features_step.get_feature_names_out())
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_)).ravel()
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    else:
        raise ValueError(f"model {type(model).__name__} exposes neither coef_ nor feature_importances_")

    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def get_shap_importance(pipeline: Pipeline, X_background: pd.DataFrame, top_n: int = 15) -> pd.DataFrame | None:
    """Mean absolute SHAP value per feature for the champion model.

    Returns None — rather than raising — if SHAP isn't installed or errors
    on this particular model/data combination, so callers can fall back to
    get_feature_importance() without special-casing failures.
    """
    try:
        import shap

        features_step = pipeline.named_steps["features"]
        model = pipeline.named_steps["model"]
        feature_names = list(features_step.get_feature_names_out())

        background = X_background.sample(n=min(100, len(X_background)), random_state=0)
        X_transformed = features_step.transform(background)

        explainer = shap.Explainer(model, X_transformed, feature_names=feature_names)
        shap_values = explainer(X_transformed)

        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            # (n_samples, n_features, n_classes) - use the positive (fraud) class.
            values = values[:, :, -1]

        mean_abs = np.abs(values).mean(axis=0)
        df = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    except Exception:
        return None
