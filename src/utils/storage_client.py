import os
from typing import Optional

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from src.utils.logger import get_logger

log = get_logger(__name__)


def get_s3_client() -> BaseClient:
    """
    Return a boto3 S3 client configured for MinIO (or AWS S3 in production).

    All connection parameters come from environment variables so no credentials
    are ever hardcoded.
    """
    endpoint = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",  # required by boto3, ignored by MinIO
    )


def object_exists(client: BaseClient, bucket: str, key: str) -> bool:
    """Return True if the object already exists in the bucket."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "404":
            return False
        raise


def upload_bytes(
    client: BaseClient,
    data: bytes,
    bucket: str,
    key: str,
    content_type: str = "application/octet-stream",
    overwrite: bool = True,
) -> Optional[int]:
    """
    Upload raw bytes to S3/MinIO.

    Returns the number of bytes uploaded, or None if skipped (already exists
    and overwrite=False).
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


def list_objects(client: BaseClient, bucket: str, prefix: str) -> list[str]:
    """Return a list of object keys under the given prefix."""
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys
