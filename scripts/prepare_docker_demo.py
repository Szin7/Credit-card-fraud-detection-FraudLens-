"""Prepare an unlabeled synthetic inference batch and its labeled training source.

The files are small and intended only for Docker/Airflow smoke tests.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def prepare_demo(root: Path, *, replace: bool = False) -> tuple[Path, Path, int]:
    raw = root / "data/raw/creditcard_sample.csv"
    outdir = root / "data/incoming"
    train = outdir / "docker-demo-train.csv"
    inference = outdir / "docker-demo-inference.csv"
    if not raw.exists():
        raise FileNotFoundError(f"Bundled synthetic file not found: {raw}")
    if not replace and (train.exists() or inference.exists()):
        raise FileExistsError("Docker demo files already exist. Inspect them or use --replace.")

    with raw.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or "Class" not in fields:
            raise ValueError("Expected labeled synthetic sample with a Class column")
        unique = set()
        by_class: dict[str, list[dict[str, str]]] = {"0": [], "1": []}
        for row in reader:
            marker = tuple(row[field] for field in fields)
            if marker in unique:
                continue
            unique.add(marker)
            if row["Class"] not in by_class:
                raise ValueError("Unexpected class label in bundled synthetic sample")
            by_class[row["Class"]].append(row)

    rng = random.Random(42)
    if len(by_class["0"]) < 6000 or len(by_class["1"]) < 80:
        raise ValueError("Bundled sample is smaller than the expected test fixture")
    rows = rng.sample(by_class["0"], 6000) + rng.sample(by_class["1"], 80)
    rng.shuffle(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    for path, names in [(train, fields), (inference, [f for f in fields if f != "Class"])]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return train, inference, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true", help="Overwrite existing Docker demo files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    train, inference, count = prepare_demo(root, replace=args.replace)
    print(f"Prepared {count:,} synthetic transactions for local container checks")
    print(f"Training:  {train}")
    print(f"Inference: {inference}")
    print("These files are test fixtures, not the Kaggle real-data evaluation.")


if __name__ == "__main__":
    main()
