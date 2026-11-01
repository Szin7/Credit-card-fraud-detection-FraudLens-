"""Run a synthetic-data demonstration with SQLite; optionally open the dashboard."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--skip-training", action="store_true", help="Reuse an existing active release")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    local = root / ".local"
    local.mkdir(exist_ok=True)
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{(local / 'demo.db').as_posix()}")

    import pandas as pd
    from scripts.generate_sample_data import generate_transactions
    from scripts.run_pipeline import run_daily_inference_pipeline, run_training_pipeline
    from src.utils.config import settings

    # Generate a separate, deterministic fixture: never overwrite a user's raw CSV.
    labeled = local / "synthetic-labeled.csv"
    generate_transactions().to_csv(labeled, index=False)
    if not args.skip_training or not (settings.model_dir / "active_model.json").exists():
        run_training_pipeline(labeled, dataset_kind="synthetic")
    unlabeled = local / "synthetic-unlabeled.csv"
    pd.read_csv(labeled).drop(columns=["Class"]).to_csv(unlabeled, index=False)
    first = run_daily_inference_pipeline(unlabeled)
    replay = run_daily_inference_pipeline(unlabeled)
    if replay["transactions_inserted"] or replay["predictions_inserted"]:
        raise RuntimeError("Replay verification failed")
    print(f"Scoring completed: {first['predictions_inserted']} new predictions. Replay inserted zero rows.")
    print("Synthetic demonstration only; dashboard volume is not a holdout evaluation.")
    if args.dashboard:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"], cwd=root, check=True)
    else:
        print("Open the dashboard with: python -m scripts.demo --skip-training --dashboard")


if __name__ == "__main__":
    main()
