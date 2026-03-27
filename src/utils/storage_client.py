import functools

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import storage_config
from src.utils.logger import get_logger

log = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def get_s3_client() -> BaseClient:
    """
    Return a cached boto3 S3 client configured for MinIO (or AWS S3).

    The client is created once and reused — boto3 clients are thread-safe.
    Credentials come from the centralized config module.
    """
    cfg = storage_config()
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        region_name="us-east-1",  # required by boto3, ignored by MinIO
    )


def object_exists(client: BaseClient, bucket: str, key: str) -> bool:
    """Return True if the object already exists in the bucket."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            return False
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def upload_bytes(
    client: BaseClient,
    data: bytes,
    bucket: str,
    key: str,
    content_type: str = "application/octet-stream",
    overwrite: bool = True,
) -> int | None:
    """
    Upload raw bytes to S3/MinIO with automatic retry on transient errors.

    Returns the number of bytes uploaded, or None if skipped because the
    object already exists and overwrite=False.
    """
    if not overwrite and object_exists(client, bucket, key):
        log.info("object_skipped_already_exists", bucket=bucket, key=key)
        return None

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    log.info("object_uploaded", bucket=bucket, key=key, size_bytes=len(data))
    return len(data)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    reraise=True,
)
def list_objects(client: BaseClient, bucket: str, prefix: str) -> list[str]:
    """Return a list of all object keys under the given prefix (paginated)."""
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys
