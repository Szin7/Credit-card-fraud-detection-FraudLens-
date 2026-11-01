"""S3-backed raw-data storage, used when STORAGE_MODE=s3.

Wraps boto3 behind the small interface ingestion needs, so it doesn't care
which storage backend it's talking to. Credentials are never read from
config here — boto3's standard credential chain (env vars,
~/.aws/credentials, or an IAM role) handles that; this module only ever
reads the bucket name and region.

Every function takes an optional `bucket` override (defaulting to
settings.aws_s3_bucket) purely for testability against a moto-mocked
bucket — settings is a frozen dataclass, so tests can't monkeypatch it
directly. See tests/test_storage.py.
"""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from src.utils.config import settings
from src.utils.logging import get_logger, log_event

logger = get_logger(__name__)


def _client():
    return boto3.client("s3", region_name=settings.aws_region)


def _require_bucket(bucket: str | None) -> str:
    bucket = bucket or settings.aws_s3_bucket
    if not bucket:
        raise ValueError("AWS_S3_BUCKET is not set - required when STORAGE_MODE=s3")
    return bucket


def upload_file(local_path: Path, key: str, bucket: str | None = None) -> str:
    bucket = _require_bucket(bucket)
    _client().upload_file(str(local_path), bucket, key)
    log_event(logger, "s3_upload_completed", bucket=bucket, key=key)
    return f"s3://{bucket}/{key}"


def download_file(key: str, local_path: Path, bucket: str | None = None) -> Path:
    bucket = _require_bucket(bucket)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket, key, str(local_path))
    log_event(logger, "s3_download_completed", bucket=bucket, key=key)
    return local_path


def object_exists(key: str, bucket: str | None = None) -> bool:
    bucket = _require_bucket(bucket)
    try:
        _client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise
