import boto3
import pytest
from moto import mock_aws

from src.storage import s3

BUCKET = "fraudlens-test-bucket"


@pytest.fixture
def bucket():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield BUCKET


def test_upload_and_download_round_trip(bucket, tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("Time,Amount,Class\n0,10.0,0\n")

    s3_uri = s3.upload_file(source, "raw/test.csv", bucket=bucket)
    assert s3_uri == f"s3://{bucket}/raw/test.csv"

    destination = tmp_path / "downloaded" / "test.csv"
    s3.download_file("raw/test.csv", destination, bucket=bucket)

    assert destination.read_text() == source.read_text()


def test_object_exists(bucket, tmp_path):
    assert s3.object_exists("raw/missing.csv", bucket=bucket) is False

    source = tmp_path / "source.csv"
    source.write_text("data")
    s3.upload_file(source, "raw/present.csv", bucket=bucket)

    assert s3.object_exists("raw/present.csv", bucket=bucket) is True


def test_upload_without_bucket_configured_raises(tmp_path):
    # No bucket override and no AWS_S3_BUCKET set in the test environment
    # (.env.example ships it blank) - both fall through to the same error.
    source = tmp_path / "source.csv"
    source.write_text("data")

    with pytest.raises(ValueError):
        s3.upload_file(source, "raw/test.csv", bucket="")
