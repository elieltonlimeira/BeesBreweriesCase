"""
Shared pytest fixtures for all test suites.

Fixtures here are available to every test file without explicit imports.
The session-scoped SparkSession is the most critical — one JVM is shared
across the entire test run to avoid the ~30s startup cost per test.
"""

import json
import os
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Environment — set BEFORE any src imports so config.py reads correct values
# ---------------------------------------------------------------------------

os.environ.setdefault("MINIO_ACCESS_KEY", "test-access-key")
os.environ.setdefault("MINIO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("MINIO_ENDPOINT", "http://localhost:9000")
os.environ.setdefault("BRONZE_BUCKET", "test-bronze")
os.environ.setdefault("SILVER_BUCKET", "test-silver")
os.environ.setdefault("GOLD_BUCKET", "test-gold")
os.environ.setdefault("API_BASE_URL", "https://api.openbrewerydb.org/v1/breweries")
os.environ.setdefault("API_PAGE_SIZE", "5")
os.environ.setdefault("API_MAX_RETRIES", "2")


# ---------------------------------------------------------------------------
# SparkSession — session-scoped: created once for the entire test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """
    Session-scoped SparkSession running in local mode via Spark Connect.

    Uses Spark Connect (remote("local")) which communicates over gRPC instead
    of py4j sockets — required for Python 3.12+ compatibility where the classic
    PySpark worker socket protocol is broken.

    - shuffle.partitions=2: reduces overhead for tiny test datasets
    """
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.remote("local")
        .config("spark.sql.shuffle.partitions", "2")
        .appName("brewery-tests")
        .getOrCreate()
    )
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Sample brewery data — covers all field types and key edge cases
# ---------------------------------------------------------------------------

SAMPLE_BREWERIES: list[dict] = [
    {
        "id": "b54b16e1-ac3b-4bff-a11f-f7ae9ddc27e0",
        "name": "Ales for ALS",
        "brewery_type": "micro",
        "address_1": "2nd Street",
        "address_2": None,
        "address_3": None,
        "city": "San Diego",
        "state_province": "California",
        "state": "California",
        "postal_code": "92101",
        "country": "United States",
        "longitude": -117.1611,
        "latitude": 32.7157,
        "phone": "6195551234",
        "website_url": "http://www.alesforals.com",
        "street": "2nd Street",
    },
    {
        "id": "5128df48-79fc-4f0f-8b52-d06be54d0cec",
        "name": "(405) Brewing Co",
        "brewery_type": "micro",
        "address_1": "1716 Topeka St",
        "address_2": None,
        "address_3": None,
        "city": "Norman",
        "state_province": "Oklahoma",
        "state": "Oklahoma",
        "postal_code": "73069-8224",
        "country": "United States",
        "longitude": -97.46818222,
        "latitude": 35.25738891,
        "phone": "4058160490",
        "website_url": "http://www.405brewing.com",
        "street": "1716 Topeka St",
    },
    {
        # Edge case: null coordinates
        "id": "abc123-no-geo",
        "name": "No Geo Brewery",
        "brewery_type": "brewpub",
        "address_1": "123 Main St",
        "address_2": None,
        "address_3": None,
        "city": "Austin",
        "state_province": None,   # edge: null state_province, has state
        "state": "Texas",
        "postal_code": "78701",
        "country": "United States",
        "longitude": None,
        "latitude": None,
        "phone": None,
        "website_url": None,
        "street": "123 Main St",
    },
    {
        # Edge case: duplicate id (should be deduplicated)
        "id": "b54b16e1-ac3b-4bff-a11f-f7ae9ddc27e0",
        "name": "Ales for ALS (duplicate)",
        "brewery_type": "MICRO",   # uppercase — should be normalized
        "address_1": "2nd Street",
        "address_2": None,
        "address_3": None,
        "city": "San Diego",
        "state_province": "California",
        "state": "California",
        "postal_code": "92101",
        "country": "united states",  # lowercase — should be normalized
        "longitude": -117.1611,
        "latitude": 32.7157,
        "phone": None,
        "website_url": None,
        "street": "2nd Street",
    },
    {
        # Edge case: null id — should be quarantined
        "id": None,
        "name": "Ghost Brewery",
        "brewery_type": "planning",
        "address_1": None,
        "address_2": None,
        "address_3": None,
        "city": "Portland",
        "state_province": "Oregon",
        "state": "Oregon",
        "postal_code": "97201",
        "country": "United States",
        "longitude": None,
        "latitude": None,
        "phone": None,
        "website_url": None,
        "street": None,
    },
]


@pytest.fixture
def sample_breweries() -> list[dict]:
    """Return the canonical test dataset with all edge cases."""
    return [b.copy() for b in SAMPLE_BREWERIES]


@pytest.fixture
def sample_breweries_json(sample_breweries) -> bytes:
    """Return sample breweries serialized as a JSON byte string."""
    return json.dumps(sample_breweries).encode("utf-8")


# ---------------------------------------------------------------------------
# Mock S3 client
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_s3_client() -> MagicMock:
    """Return a MagicMock pre-configured to simulate an S3 client."""
    client = MagicMock()
    # head_object raises ClientError with 404 by default (object does not exist)
    from botocore.exceptions import ClientError

    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    return client


# ---------------------------------------------------------------------------
# Config reset — ensures tests don't bleed config state into each other
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_pipeline_config():
    """Reset the config singletons before each test."""
    from src.utils.config import reset_config

    reset_config()
    yield
    reset_config()
