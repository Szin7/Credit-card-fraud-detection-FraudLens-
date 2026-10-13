"""Row-level validation: split an ingested batch into rows that are safe to
feed into the model pipeline and rows that aren't, with a machine-readable
reason attached to every rejection.

Every check below produces a boolean mask aligned to the input's index, so a
row can be caught by more than one rule (e.g. missing Time *and* a negative
Amount) without the checks interfering with each other.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

from src.validation.identity import feature_digest
from src.utils.json_io import write_json

import pandas as pd

from src.utils.config import settings
from src.utils.logging import get_logger, log_event
from src.validation.schema import INFERENCE_COLUMNS, NUMERIC_COLUMNS, REQUIRED_COLUMNS, V_FEATURE_COLUMNS, VALID_CLASS_VALUES

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
    # True when duplicate_rows / total_rows exceeds
    # config/settings.yaml > validation > max_duplicate_ratio_warning — a
    # batch with an unusually high duplicate rate often signals an upstream
    # data issue (a retried export, a bug feeding the same file twice)
    # worth a human's attention rather than silently deduping and moving on.
    high_duplicate_ratio: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def as_text(self) -> str:
        lines = [
            f"Total rows:        {self.total_rows}",
            f"Valid rows:        {self.valid_rows}",
            f"Rejected rows:     {self.rejected_rows}",
            f"Duplicate rows:    {self.duplicate_rows}",
            f"Missing values:    {self.missing_value_rows}",
        ]
        if self.high_duplicate_ratio:
            lines.append(
                f"WARNING: duplicate rate exceeds "
                f"{settings.max_duplicate_ratio_warning:.0%} of the batch — check the upstream source."
            )
        return "\n".join(lines)


def validate(df: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, ValidationSummary]:
    """Split `df` into (valid_rows, rejected_rows, summary).

    `rejected_rows` carries every original column plus `rejection_reason`.
    `valid_rows` is restricted to REQUIRED_COLUMNS with clean numeric dtypes.
    """
    required = REQUIRED_COLUMNS if require_target else INFERENCE_COLUMNS
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValidationError(f"batch is missing required columns: {missing_columns}")
    if df.empty:
        raise ValidationError("batch contains no rows")
    total_rows = len(df)
    unexpected_columns = sorted(set(df.columns) - set(REQUIRED_COLUMNS) - {"transaction_id"})
    working = df.copy().reset_index(drop=True)
    supplied_ids = "transaction_id" in working.columns
    if "Class" not in working:
        working["Class"] = pd.NA

    numeric = working[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    missing_mask = working[required].isna().any(axis=1)
    class_numeric = pd.to_numeric(working["Class"], errors="coerce")
    invalid_class_mask = ~class_numeric.isin(VALID_CLASS_VALUES)
    if not require_target:
        invalid_class_mask &= working["Class"].notna()
    amount_numeric = numeric["Amount"]
    negative_amount_mask = ~np.isfinite(amount_numeric) | (amount_numeric < 0)
    time_numeric = numeric["Time"]
    invalid_time_mask = ~np.isfinite(time_numeric) | (time_numeric < 0)
    non_numeric_feature_mask = ~np.isfinite(numeric[V_FEATURE_COLUMNS]).all(axis=1)
    if supplied_ids:
        ids = working["transaction_id"].astype("string").str.strip()
        invalid_id_mask = ids.isna() | ids.eq("") | ids.str.len().gt(200)
        working["transaction_id"] = ids
        duplicate_mask = ids.duplicated(keep="first") & ~invalid_id_mask
    else:
        invalid_id_mask = pd.Series(False, index=working.index)
        duplicate_mask = numeric.duplicated(keep="first")
        working.insert(0, "transaction_id", pd.Series(index=working.index, dtype="string"))

    flags = pd.DataFrame(
        {
            "missing_required_value": missing_mask,
            "invalid_class": invalid_class_mask,
            "negative_or_invalid_amount": negative_amount_mask,
            "invalid_time": invalid_time_mask,
            "non_numeric_feature_value": non_numeric_feature_mask,
            "duplicate_row": duplicate_mask,
            "invalid_transaction_id": invalid_id_mask,
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

    valid_df = working.loc[~reject_mask, ["transaction_id", *REQUIRED_COLUMNS]].copy()
    valid_df[NUMERIC_COLUMNS] = valid_df[NUMERIC_COLUMNS].apply(pd.to_numeric)
    valid_df["Class"] = pd.to_numeric(valid_df["Class"]).astype("Int64")
    if not supplied_ids:
        valid_df["transaction_id"] = valid_df.apply(feature_digest, axis=1) if not valid_df.empty else pd.Series(dtype="string")

    duplicate_count = int(duplicate_mask.sum())
    duplicate_ratio = (duplicate_count / total_rows) if total_rows else 0.0

    summary = ValidationSummary(
        total_rows=total_rows,
        valid_rows=len(valid_df),
        rejected_rows=int(reject_mask.sum()),
        duplicate_rows=duplicate_count,
        missing_value_rows=int(missing_mask.sum()),
        invalid_class_rows=int(invalid_class_mask.sum()),
        negative_amount_rows=int(negative_amount_mask.sum()),
        invalid_time_rows=int(invalid_time_mask.sum()),
        non_numeric_feature_rows=int(non_numeric_feature_mask.sum()),
        unexpected_columns=unexpected_columns,
        high_duplicate_ratio=duplicate_ratio > settings.max_duplicate_ratio_warning,
    )

    return valid_df, rejected_df, summary


def run_validation(input_path: Path | None = None, *, require_target: bool = True, output_dir: Path | None = None) -> ValidationSummary:
    """Read the raw file, validate it, and persist valid/rejected splits
    plus a summary under data/processed and data/rejected.
    """
    path = Path(input_path) if input_path is not None else settings.raw_data_path
    df = pd.read_csv(path, dtype={"transaction_id": "string"})

    valid_df, rejected_df, summary = validate(df, require_target=require_target)

    processed = Path(output_dir) if output_dir is not None else settings.processed_dir
    rejected = processed if output_dir is not None else settings.rejected_dir
    processed.mkdir(parents=True, exist_ok=True)
    rejected.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(processed / VALID_TRANSACTIONS_FILENAME, index=False)
    rejected_df.to_csv(rejected / REJECTED_TRANSACTIONS_FILENAME, index=False)
    write_json(processed / VALIDATION_SUMMARY_FILENAME, summary.to_dict())

    log_event(
        logger,
        "validation_completed",
        total_rows=summary.total_rows,
        valid_rows=summary.valid_rows,
        rejected_rows=summary.rejected_rows,
    )
    if summary.rejected_rows:
        log_event(logger, "records_rejected", rejected_rows=summary.rejected_rows)
    if summary.high_duplicate_ratio:
        log_event(
            logger,
            "high_duplicate_ratio_detected",
            duplicate_rows=summary.duplicate_rows,
            total_rows=summary.total_rows,
            threshold=settings.max_duplicate_ratio_warning,
        )

    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate an ingested transaction CSV.")
    parser.add_argument("--input", type=Path, default=None, help="Path to the CSV file to validate.")
    parser.add_argument("--inference", action="store_true", help="Allow missing ground-truth labels.")
    args = parser.parse_args()

    summary = run_validation(args.input, require_target=not args.inference)
    print(summary.as_text())


if __name__ == "__main__":
    main()
