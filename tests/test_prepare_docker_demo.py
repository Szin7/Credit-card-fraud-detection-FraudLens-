"""Checks for the Docker input fixture generator."""

import csv

import pytest

from scripts.prepare_docker_demo import prepare_demo


def test_prepare_and_do_not_overwrite(tmp_path):
    raw = tmp_path / "data/raw/creditcard_sample.csv"
    raw.parent.mkdir(parents=True)
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Time", "Amount", "V1", "Class"])
        writer.writeheader()
        for i in range(6010):
            writer.writerow({"Time": str(i), "Amount": "10", "V1": str(i), "Class": "0"})
        for i in range(90):
            writer.writerow({"Time": str(7000+i), "Amount": "20", "V1": str(i), "Class": "1"})

    train, inference, size = prepare_demo(tmp_path)
    assert size == 6080
    with train.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6080
    assert sum(row["Class"] == "1" for row in rows) == 80
    with inference.open(newline="", encoding="utf-8") as handle:
        unlabeled = list(csv.DictReader(handle))
    assert len(unlabeled) == 6080
    assert "Class" not in unlabeled[0]
    with pytest.raises(FileExistsError):
        prepare_demo(tmp_path)
