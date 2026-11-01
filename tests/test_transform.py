import pandas as pd

from src.processing.transform import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN, model_frame


def _raw_frame() -> pd.DataFrame:
    row = {col: 0.1 for col in FEATURE_COLUMNS}
    row[ID_COLUMN] = "tx-1"
    row[TARGET_COLUMN] = 0
    row["unexpected_extra_column"] = "should be dropped"
    return pd.DataFrame([row])


def test_model_frame_selects_only_expected_columns():
    df = _raw_frame()
    frame = model_frame(df)

    assert list(frame.columns) == [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    assert "unexpected_extra_column" not in frame.columns


def test_model_frame_can_omit_target_for_inference():
    df = _raw_frame()
    frame = model_frame(df, include_target=False)

    assert TARGET_COLUMN not in frame.columns
    assert list(frame.columns) == [ID_COLUMN, *FEATURE_COLUMNS]
