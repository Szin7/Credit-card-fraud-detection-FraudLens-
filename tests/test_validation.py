import pandas as pd
import pytest

from src.validation.schema import REQUIRED_COLUMNS, V_FEATURE_COLUMNS
from src.validation.validator import ValidationError, validate


def _base_row(**overrides) -> dict:
    row = {col: 0.5 for col in V_FEATURE_COLUMNS}
    row["Time"] = 100.0
    row["Amount"] = 25.0
    row["Class"] = 0
    row.update(overrides)
    return row


def _frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=REQUIRED_COLUMNS)


def test_all_valid_rows_pass_through():
    df = _frame(_base_row(), _base_row(Time=200.0, Class=1))
    valid_df, rejected_df, summary = validate(df)

    assert len(valid_df) == 2
    assert len(rejected_df) == 0
    assert summary.total_rows == 2
    assert summary.rejected_rows == 0


def test_missing_required_column_raises():
    df = _frame(_base_row()).drop(columns=["Amount"])
    with pytest.raises(ValidationError):
        validate(df)


def test_negative_amount_is_rejected():
    df = _frame(_base_row(), _base_row(Amount=-10.0))
    valid_df, rejected_df, summary = validate(df)

    assert len(valid_df) == 1
    assert len(rejected_df) == 1
    assert "negative_or_invalid_amount" in rejected_df.iloc[0]["rejection_reason"]
    assert summary.negative_amount_rows == 1


def test_invalid_class_value_is_rejected():
    df = _frame(_base_row(), _base_row(Class=2))
    valid_df, rejected_df, summary = validate(df)

    assert len(valid_df) == 1
    assert "invalid_class" in rejected_df.iloc[0]["rejection_reason"]
    assert summary.invalid_class_rows == 1


def test_missing_value_is_rejected():
    df = _frame(_base_row(), _base_row(Time=None))
    valid_df, rejected_df, summary = validate(df)

    assert len(valid_df) == 1
    assert "missing_required_value" in rejected_df.iloc[0]["rejection_reason"]
    assert summary.missing_value_rows == 1


def test_non_numeric_feature_value_is_rejected():
    df = _frame(_base_row(), _base_row(V1="not_a_number"))
    valid_df, rejected_df, summary = validate(df)

    assert len(valid_df) == 1
    assert "non_numeric_feature_value" in rejected_df.iloc[0]["rejection_reason"]
    assert summary.non_numeric_feature_rows == 1


def test_duplicate_rows_are_rejected_keeping_the_first():
    row = _base_row()
    df = _frame(row, dict(row))
    valid_df, rejected_df, summary = validate(df)

    assert len(valid_df) == 1
    assert len(rejected_df) == 1
    assert "duplicate_row" in rejected_df.iloc[0]["rejection_reason"]
    assert summary.duplicate_rows == 1


def test_high_duplicate_ratio_is_flagged_when_over_the_configured_threshold():
    # 1 duplicate out of 2 rows = 50%, far above the default 5% warning
    # threshold (config/settings.yaml > validation > max_duplicate_ratio_warning).
    row = _base_row()
    df = _frame(row, dict(row))
    _, _, summary = validate(df)

    assert summary.high_duplicate_ratio is True
    assert "WARNING" in summary.as_text()


def test_low_duplicate_ratio_is_not_flagged():
    row = _base_row()
    rows = [row] + [_base_row(Time=float(i)) for i in range(1, 25)] + [dict(row)]
    df = _frame(*rows)  # 1 duplicate out of 26 rows = ~3.8%, under the threshold
    _, _, summary = validate(df)

    assert summary.duplicate_rows == 1
    assert summary.high_duplicate_ratio is False
    assert "WARNING" not in summary.as_text()


def test_unexpected_columns_are_reported_but_not_fatal():
    df = _frame(_base_row())
    df["merchant_note"] = "unused"
    valid_df, rejected_df, summary = validate(df)

    assert len(valid_df) == 1
    assert "merchant_note" not in valid_df.columns
    assert summary.unexpected_columns == ["merchant_note"]


def test_summary_counts_are_internally_consistent():
    df = _frame(_base_row(), _base_row(Amount=-1.0), _base_row(Class=1))
    _, _, summary = validate(df)

    assert summary.total_rows == summary.valid_rows + summary.rejected_rows
