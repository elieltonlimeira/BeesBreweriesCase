"""
Bronze layer writer.

Persists raw API responses (JSON) to the bronze bucket in MinIO/S3.
Files are partitioned by execution date and page number for idempotent reruns.

Storage layout:
    s3://brewery-bronze/raw/dt=YYYY-MM-DD/page=NNN.json
"""

import json
from typing import Any

from botocore.client import BaseClient

from src.utils.config import storage_config
from src.utils.logger import get_logger
from src.utils.storage_client import upload_bytes

log = get_logger(__name__)


def build_bronze_key(execution_date: str, page: int) -> str:
    """
    Return the S3 key for a bronze page file.

    Format: raw/dt=YYYY-MM-DD/page=NNN.json  (zero-padded page number)
    Zero-padding ensures lexicographic sort order matches page order.
    """
    return f"raw/dt={execution_date}/page={page:03d}.json"


def write_page(
    records: list[dict[str, Any]],
    page: int,
    execution_date: str,
    s3_client: BaseClient,
    overwrite: bool = True,
) -> dict[str, Any]:
    """
    Serialize a list of brewery records and upload to the bronze bucket.

    Args:
        records: Raw brewery dicts returned by the API.
        page: Page number (1-indexed).
        execution_date: ISO date string (YYYY-MM-DD) used for partitioning.
        s3_client: Configured boto3 S3 client.
        overwrite: If False, skip the upload when the file already exists.
                   Set to False to make pipeline reruns idempotent.

    Returns:
        dict with "key", "records_written", "bytes_written", and "skipped" for XCom.
    """
    cfg = storage_config()
    key = build_bronze_key(execution_date, page)
    data = json.dumps(records, ensure_ascii=False).encode("utf-8")

    bytes_written = upload_bytes(
        client=s3_client,
        data=data,
        bucket=cfg.bronze_bucket,
        key=key,
        content_type="application/json",
        overwrite=overwrite,
    )

    result = {
        "key": key,
        "records_written": len(records) if bytes_written is not None else 0,
        "bytes_written": bytes_written or 0,
        "skipped": bytes_written is None,
    }

    log.info("bronze_page_written", page=page, execution_date=execution_date, **result)
    return result
