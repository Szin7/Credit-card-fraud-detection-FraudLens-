"""Generate a small, clearly-synthetic transaction dataset for local
development and dashboard demonstration.

This is NOT the real credit-card fraud dataset. It exists because the repo
ships without it (the real dataset has redistribution restrictions). The
schema (Time, V1-V28, Amount, Class) matches the real Kaggle dataset so the
rest of the pipeline is agnostic to which one it's reading, but the values
are entirely fabricated:

  - V1-V28 are independent Gaussian draws. For fraud rows a handful of them
    get an arbitrary mean/variance shift purely so a classifier has signal
    to find. This has no connection to what V1-V28 represent in the real
    (PCA-anonymized) dataset, and V1-V28 are never given semantic labels
    anywhere in this codebase.
  - Amount is drawn from a lognormal distribution with a modest shift for
    fraud rows.
  - Time is uniform over a 48-hour span, matching the real dataset's span.
  - A handful of exact duplicate rows are injected on purpose so the
    validation module's dedup logic has something real to detect and report.

Output: data/raw/creditcard_sample.csv (never named creditcard.csv, so it
can never be mistaken for the real dataset). See data/raw/README.md for how
to use the real dataset instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.logging import get_logger, log_event

logger = get_logger(__name__)

N_LEGIT = 24_000
N_FRAUD = 120
N_DUPLICATES = 25
TIME_SPAN_SECONDS = 172_792  # matches the real dataset's ~48-hour span
N_V_FEATURES = 28
SHIFTED_COMPONENTS = [1, 2, 3, 4, 5, 9, 14]  # arbitrary subset of V-indices, 1-based

FRAUD_MEAN_SHIFT = 2.4
FRAUD_STD_SCALE = 1.6

OUTPUT_PATH = settings.raw_dir / "creditcard_sample.csv"


def _generate_v_features(n_rows: int, rng: np.random.Generator, is_fraud: bool) -> np.ndarray:
    v = rng.normal(loc=0.0, scale=1.0, size=(n_rows, N_V_FEATURES))
    if is_fraud:
        for idx in SHIFTED_COMPONENTS:
            col = idx - 1
            v[:, col] = rng.normal(loc=FRAUD_MEAN_SHIFT, scale=FRAUD_STD_SCALE, size=n_rows)
    return v


def _generate_amount(n_rows: int, rng: np.random.Generator, is_fraud: bool) -> np.ndarray:
    if is_fraud:
        amount = rng.lognormal(mean=3.6, sigma=1.4, size=n_rows)
    else:
        amount = rng.lognormal(mean=3.0, sigma=1.1, size=n_rows)
    return np.round(amount, 2)


def generate_transactions(seed: int = settings.random_seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    legit_time = np.sort(rng.uniform(0, TIME_SPAN_SECONDS, size=N_LEGIT))
    fraud_time = np.sort(rng.uniform(0, TIME_SPAN_SECONDS, size=N_FRAUD))

    legit_v = _generate_v_features(N_LEGIT, rng, is_fraud=False)
    fraud_v = _generate_v_features(N_FRAUD, rng, is_fraud=True)

    legit_amount = _generate_amount(N_LEGIT, rng, is_fraud=False)
    fraud_amount = _generate_amount(N_FRAUD, rng, is_fraud=True)

    v_columns = [f"V{i}" for i in range(1, N_V_FEATURES + 1)]

    legit_df = pd.DataFrame(legit_v, columns=v_columns)
    legit_df.insert(0, "Time", np.round(legit_time, 0))
    legit_df["Amount"] = legit_amount
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud_v, columns=v_columns)
    fraud_df.insert(0, "Time", np.round(fraud_time, 0))
    fraud_df["Amount"] = fraud_amount
    fraud_df["Class"] = 1

    transactions = pd.concat([legit_df, fraud_df], ignore_index=True)
    transactions = transactions.sort_values("Time").reset_index(drop=True)
    transactions[v_columns] = transactions[v_columns].round(6)

    duplicate_rows = transactions.sample(n=N_DUPLICATES, random_state=seed)
    transactions = pd.concat([transactions, duplicate_rows], ignore_index=True)

    return transactions


def main() -> None:
    log_event(logger, "sample_data_generation_started", seed=settings.random_seed)

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    transactions = generate_transactions()
    transactions.to_csv(OUTPUT_PATH, index=False)

    fraud_count = int((transactions["Class"] == 1).sum())
    log_event(
        logger,
        "sample_data_generation_completed",
        output_path=str(OUTPUT_PATH),
        total_rows=len(transactions),
        fraud_rows=fraud_count,
        fraud_rate_pct=round(100 * fraud_count / len(transactions), 4),
        injected_duplicates=N_DUPLICATES,
    )


if __name__ == "__main__":
    main()
