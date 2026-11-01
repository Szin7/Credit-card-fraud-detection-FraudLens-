"""Shape validated transactions into the canonical model-ready frame.

Centralizes the feature-column list and load path so training and
inference can't quietly drift out of sync about which columns the model
expects — before this existed, FEATURE_COLUMNS was defined independently
in both src/models/train.py and src/models/predict.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import settings
from src.validation.schema import V_FEATURE_COLUMNS

FEATURE_COLUMNS = ["Time", "Amount", *V_FEATURE_COLUMNS]
ID_COLUMN = "transaction_id"
TARGET_COLUMN = "Class"


def load_processed_transactions(path: Path | None = None) -> pd.DataFrame:
    path = path or (settings.processed_dir / "transactions_valid.csv")
    return pd.read_csv(path, dtype={"transaction_id": "string"})


def model_frame(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Select and order exactly the columns the model pipeline expects,
    so callers can't accidentally pass extra columns through to sklearn.
    """
    columns = [ID_COLUMN, *FEATURE_COLUMNS]
    if include_target and TARGET_COLUMN in df.columns:
        columns.append(TARGET_COLUMN)
    return df[columns].copy()
