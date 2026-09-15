"""Row-level validation: split an ingested batch into rows that are safe to
feed into the model pipeline and rows that aren't, with a machine-readable
reason attached to every rejection.

Every check below produces a boolean mask aligned to the input's index, so a
row can be caught by more than one rule (e.g. missing Time *and* a negative
Amount) without the checks interfering with each other.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.utils.config import settings
from src.utils.logging import get_logger, log_event
from src.validation.schema import NUMERIC_COLUMNS, REQUIRED_COLUMNS, V_FEATURE_COLUMNS, VALID_CLASS_VALUES

logger = get_logger(__name__)

VALID_TRANSACTIONS_FILENAME = "transactions_valid.csv"
REJECTED_TRANSACTIONS_FILENAME = "transactions_rejected.csv"
VALIDATION_SUMMARY_FILENAME = "validation_summary.json"


class ValidationError(Exception):
    """Raised when the batch can't be validated at all (required columns
    absent) — as opposed to individual rows failing rules, which are
    rejected rather than treated as a fatal error.
    """


@dataclass
class ValidationSummary:
    total_rows: int
    valid_rows: int
    rejected_rows: int
    duplicate_rows: int
    missing_value_rows: int
    invalid_class_rows: int
    negative_amount_rows: int
    invalid_time_rows: int
    non_numeric_feature_rows: int
    unexpected_columns: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def as_text(self) -> str:
        return (
            f"Total rows:        {self.total_rows}\n"
            f"Valid rows:        {self.valid_rows}\n"
            f"Rejected rows:     {self.rejected_rows}\n"
            f"Duplicate rows:    {self.duplicate_rows}\n"
            f"Missing values:    {self.missing_value_rows}"
        )


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, ValidationSummary]:
    """Split `df` into (valid_rows, rejected_rows, summary).

    `rejected_rows` carries every original column plus `rejection_reason`.
    `valid_rows` is restricted to REQUIRED_COLUMNS with clean numeric dtypes.
    """
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValidationError(f"batch is missing required columns: {missing_columns}")

    total_rows = len(df)
    unexpected_columns = sorted(set(df.columns) - set(REQUIRED_COLUMNS))
    working = df.copy()

    missing_mask = working[REQUIRED_COLUMNS].isna().any(axis=1)

    class_numeric = pd.to_numeric(working["Class"], errors="coerce")
    invalid_class_mask = class_numeric.isna() | ~class_numeric.isin(VALID_CLASS_VALUES)

    amount_numeric = pd.to_numeric(working["Amount"], errors="coerce")
    negative_amount_mask = amount_numeric.isna() | (amount_numeric < 0)

    time_numeric = pd.to_numeric(working["Time"], errors="coerce")
    invalid_time_mask = time_numeric.isna() | (time_numeric < 0)

    v_numeric = working[V_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    # Present but unparseable (e.g. text in a V column) — distinct from an
    # empty cell, which missing_mask already caught.
    non_numeric_feature_mask = v_numeric.isna().any(axis=1) & working[V_FEATURE_COLUMNS].notna().all(axis=1)

    duplicate_mask = working.duplicated(subset=REQUIRED_COLUMNS, keep="first")

    flags = pd.DataFrame(
        {
            "missing_required_value": missing_mask,
            "invalid_class": invalid_class_mask,
            "negative_or_invalid_amount": negative_amount_mask,
            "invalid_time": invalid_time_mask,
            "non_numeric_feature_value": non_numeric_feature_mask,
            "duplicate_row": duplicate_mask,
        },
        index=working.index,
    )
    reject_mask = flags.any(axis=1)

    rejected_df = working.loc[reject_mask].copy()
    rejected_flags = flags.loc[reject_mask]
    if rejected_flags.empty:
        # DataFrame.apply(axis=1) on a zero-row frame can't infer a Series
        # result and returns a DataFrame instead — short-circuit that case.
        rejected_df["rejection_reason"] = pd.Series(dtype=object)
    else:
        rejected_df["rejection_reason"] = rejected_flags.apply(
            lambda row: "; ".join(row.index[row.values]), axis=1
        )

    valid_df = working.loc[~reject_mask, REQUIRED_COLUMNS].copy()
    valid_df[NUMERIC_COLUMNS] = valid_df[NUMERIC_COLUMNS].apply(pd.to_numeric)
    valid_df["Class"] = valid_df["Class"].astype(int)

    summary = ValidationSummary(
        total_rows=total_rows,
        valid_rows=len(valid_df),
        rejected_rows=int(reject_mask.sum()),
        duplicate_rows=int(duplicate_mask.sum()),
        missing_value_rows=int(missing_mask.sum()),
        invalid_class_rows=int(invalid_class_mask.sum()),
        negative_amount_rows=int(negative_amount_mask.sum()),
        invalid_time_rows=int(invalid_time_mask.sum()),
        non_numeric_feature_rows=int(non_numeric_feature_mask.sum()),
        unexpected_columns=unexpected_columns,
    )

    return valid_df, rejected_df, summary


def run_validation(input_path: Path | None = None) -> ValidationSummary:
    """Read the raw file, validate it, and persist valid/rejected splits
    plus a summary under data/processed and data/rejected.
    """
    path = Path(input_path) if input_path is not None else settings.raw_data_path
    df = pd.read_csv(path)

    valid_df, rejected_df, summary = validate(df)

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.rejected_dir.mkdir(parents=True, exist_ok=True)

    valid_df.to_csv(settings.processed_dir / VALID_TRANSACTIONS_FILENAME, index=False)
    rejected_df.to_csv(settings.rejected_dir / REJECTED_TRANSACTIONS_FILENAME, index=False)
    with open(settings.processed_dir / VALIDATION_SUMMARY_FILENAME, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    log_event(
        logger,
        "validation_completed",
        total_rows=summary.total_rows,
        valid_rows=summary.valid_rows,
        rejected_rows=summary.rejected_rows,
    )
    if summary.rejected_rows:
        log_event(logger, "records_rejected", rejected_rows=summary.rejected_rows)

    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate an ingested transaction CSV.")
    parser.add_argument("--input", type=Path, default=None, help="Path to the CSV file to validate.")
    args = parser.parse_args()

    summary = run_validation(args.input)
    print(summary.as_text())


if __name__ == "__main__":
    main()
