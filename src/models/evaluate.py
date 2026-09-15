"""Classification metrics for the fraud model.

Leads with precision, recall, F1, and PR-AUC rather than accuracy — with
fraud under 1% of transactions, accuracy stays near-perfect even for a
model that never flags fraud (see notebooks/01_exploration.ipynb for the
concrete numbers on this dataset).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> ClassificationMetrics:
    """Threshold-dependent metrics (precision/recall/F1/confusion matrix) at
    `threshold`, plus threshold-independent ROC-AUC and PR-AUC computed over
    the full probability range.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return ClassificationMetrics(
        threshold=threshold,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


MAX_CURVE_POINTS = 500


def _downsample(*arrays: np.ndarray, max_points: int = MAX_CURVE_POINTS) -> list[np.ndarray]:
    """Evenly subsample curve arrays so metadata stays a reasonable size
    even at the real dataset's scale (a raw precision-recall curve has one
    point per unique score, which can be tens of thousands on 284k rows).
    Always keeps the first and last point.
    """
    n = len(arrays[0])
    if n <= max_points:
        return list(arrays)
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return [arr[idx] for arr in arrays]


def precision_recall_curve_points(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one fewer threshold than precision/recall points
    thresholds = np.append(thresholds, 1.0)
    precision, recall, thresholds = _downsample(precision, recall, thresholds)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def roc_curve_points(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    fpr, tpr, thresholds = _downsample(fpr, tpr, thresholds)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }
