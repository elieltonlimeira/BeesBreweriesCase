import math
from typing import Any

import requests
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.config import api_config
from src.utils.logger import get_logger

log = get_logger(__name__)

CONNECT_TIMEOUT = 10  # seconds
READ_TIMEOUT = 30  # seconds

_RETRYABLE_NETWORK_ERRORS = (
    requests.Timeout,
    requests.ConnectionError,
)


def _log_before_sleep(retry_state: RetryCallState) -> None:
    """Log a warning before each retry attempt (structlog-compatible)."""
    log.warning(
        "retrying_after_error",
        attempt=retry_state.attempt_number,
        wait_seconds=retry_state.next_action.sleep if retry_state.next_action else None,
        error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


def fetch_brewery_meta() -> dict[str, Any]:
    """
    Fetch metadata from the Open Brewery DB API.

    Returns a dict containing at minimum {"total": <int>}.
    """
    cfg = api_config()
    url = f"{cfg.base_url}/meta"
    log.info("fetching_brewery_meta", url=url)

    response = requests.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    response.raise_for_status()

    meta = response.json()
    log.info("brewery_meta_received", total=meta.get("total"))
    return meta


def _make_fetch_page(cfg_max_retries: int):
    """Build the fetch_brewery_page function with runtime retry config."""

    @retry(
        stop=stop_after_attempt(cfg_max_retries),
        wait=wait_exponential(multiplier=1, min=5, max=30),
        retry=(
            retry_if_exception_type(_RETRYABLE_NETWORK_ERRORS)
            | retry_if_exception_type(requests.HTTPError)
        ),
        before_sleep=_log_before_sleep,
        reraise=True,
    )
    def _fetch(page: int, per_page: int) -> list[dict[str, Any]]:
        cfg = api_config()
        params = {"page": page, "per_page": per_page}
        log.info("fetching_brewery_page", page=page, per_page=per_page)

        response = requests.get(
            cfg.base_url,
            params=params,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )

        if response.status_code >= 500:
            response.raise_for_status()  # triggers tenacity retry

        response.raise_for_status()  # propagates 4xx immediately (no retry)

        records = response.json()
        log.info("brewery_page_fetched", page=page, records=len(records))
        return records

    return _fetch


def fetch_brewery_page(page: int, per_page: int | None = None) -> list[dict[str, Any]]:
    """
    Fetch a single page of brewery records from the API.

    Retries on network errors and HTTP 5xx with exponential backoff.
    Raises immediately on HTTP 4xx.
    """
    cfg = api_config()
    effective_per_page = per_page if per_page is not None else cfg.page_size
    fetcher = _make_fetch_page(cfg.max_retries)
    return fetcher(page, effective_per_page)


def calculate_total_pages(total: int, per_page: int | None = None) -> int:
    """Return the number of pages required to retrieve all records."""
    cfg = api_config()
    effective_per_page = per_page if per_page is not None else cfg.page_size
    return math.ceil(total / effective_per_page)
