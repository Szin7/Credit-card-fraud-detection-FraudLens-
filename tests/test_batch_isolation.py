from pathlib import Path

import pytest

from src.ingestion.ingest import ingest
from src.validation.validator import run_validation
from tests.test_reliability import frame


def test_each_batch_uses_its_own_input_snapshot(tmp_path):
    source = tmp_path / "incoming.csv"
    frame(3).to_csv(source, index=False)
    a = ingest(source, require_target=False, output_dir=tmp_path / "batch-a")
    frame(7).to_csv(source, index=False)
    b = ingest(source, require_target=False, output_dir=tmp_path / "batch-b")
    summary_a = run_validation(Path(a.source_path), require_target=False, output_dir=tmp_path / "batch-a")
    summary_b = run_validation(Path(b.source_path), require_target=False, output_dir=tmp_path / "batch-b")
    assert summary_a.valid_rows == 3 and summary_b.valid_rows == 7
    assert a.checksum_sha256 != b.checksum_sha256
    with pytest.raises(FileExistsError):
        ingest(source, require_target=False, output_dir=tmp_path / "batch-a")
