"""Inspect the validation threshold sweep saved with the active model."""
from src.models.artifacts import load_bundle
import pandas as pd


def run_threshold_analysis():
    _, metadata = load_bundle()
    return pd.DataFrame(metadata["threshold_sweep"])


def main():
    print(run_threshold_analysis().to_string(index=False))


if __name__ == "__main__":
    main()
