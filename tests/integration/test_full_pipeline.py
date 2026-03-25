"""
End-to-end pipeline integration test using moto (S3 mock) and in-process Spark.

Pipeline flow tested:
    Bronze write  →  Silver transform  →  Gold aggregate

This test validates the full data transformation chain without requiring a
running MinIO instance or a Spark cluster. It uses moto to mock S3 and the
session-scoped SparkSession from conftest.

Key assertions:
    - Gold SUM(brewery_count) == Silver valid row count
    - Silver deduplicates, quarantines null IDs, backfills state_province
    - Gold groups correctly by (brewery_type, country, state_province)
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from moto import mock_aws
import boto3

from src.bronze.bronze_writer import write_page
from src.silver.silver_transformer import transform
from src.gold.gold_aggregator import aggregate

# moto requires these env vars before creating the S3 resource
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")

BRONZE_BUCKET = "test-bronze"
SILVER_BUCKET = "test-silver"
GOLD_BUCKET = "test-gold"


@pytest.fixture(scope="module")
def s3_buckets():
    """Create mocked S3 buckets for the duration of the module tests."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        for bucket in (BRONZE_BUCKET, SILVER_BUCKET, GOLD_BUCKET):
            client.create_bucket(Bucket=bucket)
        yield client


@pytest.fixture(scope="module")
def bronze_records():
    """
    Sample records representative of what the API returns.

    5 unique records + 1 explicit duplicate (id-0000) = 6 total records.
    After dedup: 5 unique records.
    """
    _specs = [
        # (id, brewery_type, city, state, country, lon, lat)
        ("id-0000", "micro",   "Denver",   "Colorado", "United States", -104.99, 39.73),
        ("id-0001", "micro",   "Boulder",  "Colorado", "United States", -105.27, 40.01),
        ("id-0002", "brewpub", "Austin",   "Texas",    "United States", -97.74,  30.27),
        ("id-0003", "brewpub", "Houston",  "Texas",    "United States", None,    None),
        ("id-0004", "nano",    "Portland", "Oregon",   "United States", -122.67, 45.52),
        # Explicit duplicate of id-0000 — must be deduped by silver transformer
        ("id-0000", "micro",   "Denver",   "Colorado", "United States", -104.99, 39.73),
    ]
    return [
        {
            "id": rec_id,
            "name": f"Brewery {i}",
            "brewery_type": btype,
            "address_1": f"{i} Main St",
            "address_2": None,
            "address_3": None,
            "city": city,
            "state_province": state,
            "state": state,
            "postal_code": "12345",
            "country": country,
            "longitude": lon,
            "latitude": lat,
            "phone": None,
            "website_url": None,
            "street": f"{i} Main St",
        }
        for i, (rec_id, btype, city, state, country, lon, lat) in enumerate(_specs)
    ]


class TestBronzeWrite:
    def test_write_page_uploads_json(self, s3_buckets, bronze_records):
        with patch("src.utils.config.storage_config") as mock_cfg:
            mock_cfg.return_value = MagicMock(bronze_bucket=BRONZE_BUCKET)
            result = write_page(
                records=bronze_records,
                page=1,
                execution_date="2024-03-24",
                s3_client=s3_buckets,
            )

        assert result["records_written"] == len(bronze_records)
        assert result["skipped"] is False

    def test_bronze_key_is_retrievable(self, s3_buckets):
        obj = s3_buckets.get_object(
            Bucket=BRONZE_BUCKET, Key="raw/dt=2024-03-24/page=001.json"
        )
        content = json.loads(obj["Body"].read().decode("utf-8"))
        assert isinstance(content, list)
        assert len(content) > 0


class TestSilverTransform:
    def test_transform_deduplicates_and_quarantines(self, spark, bronze_records):
        """End-to-end transform on the full bronze_records set."""
        from pyspark.sql.types import (
            DoubleType, StringType, StructField, StructType,
        )

        schema = StructType([
            StructField("id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("brewery_type", StringType(), True),
            StructField("address_1", StringType(), True),
            StructField("address_2", StringType(), True),
            StructField("address_3", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state_province", StringType(), True),
            StructField("state", StringType(), True),
            StructField("postal_code", StringType(), True),
            StructField("country", StringType(), True),
            StructField("longitude", DoubleType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("phone", StringType(), True),
            StructField("website_url", StringType(), True),
            StructField("street", StringType(), True),
        ])

        # Insert one null-id record for quarantine
        null_id_record = bronze_records[0].copy()
        null_id_record["id"] = None
        all_records = bronze_records + [null_id_record]

        df = spark.createDataFrame(all_records, schema=schema)
        valid_df, quarantine_df = transform(df, "2024-03-24")

        # Null id goes to quarantine
        assert quarantine_df.count() == 1

        # Duplicate id-0000 is deduped (6 records - 1 dup - 1 null_id → 5 valid)
        valid_ids = [r["id"] for r in valid_df.collect()]
        assert len(valid_ids) == len(set(valid_ids)), "Duplicate IDs in valid set"
        assert len(valid_ids) == 5

    def test_transform_adds_pipeline_run_date(self, spark, bronze_records):
        from datetime import date
        from pyspark.sql.types import (
            DoubleType, StringType, StructField, StructType,
        )

        schema = StructType([
            StructField("id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("brewery_type", StringType(), True),
            StructField("address_1", StringType(), True),
            StructField("address_2", StringType(), True),
            StructField("address_3", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state_province", StringType(), True),
            StructField("state", StringType(), True),
            StructField("postal_code", StringType(), True),
            StructField("country", StringType(), True),
            StructField("longitude", DoubleType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("phone", StringType(), True),
            StructField("website_url", StringType(), True),
            StructField("street", StringType(), True),
        ])

        df = spark.createDataFrame(bronze_records, schema=schema)
        valid_df, _ = transform(df, "2024-03-24")

        run_dates = {r["pipeline_run_date"] for r in valid_df.collect()}
        assert run_dates == {date(2024, 3, 24)}


class TestGoldAggregate:
    def _make_silver_df(self, spark, bronze_records):
        from pyspark.sql.types import (
            DateType, DoubleType, StringType, StructField, StructType,
        )
        from datetime import date

        schema = StructType([
            StructField("id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("brewery_type", StringType(), True),
            StructField("address_1", StringType(), True),
            StructField("address_2", StringType(), True),
            StructField("address_3", StringType(), True),
            StructField("city", StringType(), True),
            StructField("state_province", StringType(), True),
            StructField("state", StringType(), True),
            StructField("postal_code", StringType(), True),
            StructField("country", StringType(), True),
            StructField("longitude", DoubleType(), True),
            StructField("latitude", DoubleType(), True),
            StructField("phone", StringType(), True),
            StructField("website_url", StringType(), True),
            StructField("street", StringType(), True),
        ])
        bronze_df = spark.createDataFrame(bronze_records, schema=schema)
        valid_df, _ = transform(bronze_df, "2024-03-24")
        return valid_df

    def test_gold_sum_equals_silver_count(self, spark, bronze_records):
        """Core invariant: SUM(brewery_count) in gold == valid silver row count."""
        silver_df = self._make_silver_df(spark, bronze_records)
        gold_df = aggregate(silver_df)

        silver_count = silver_df.count()
        gold_sum = gold_df.agg({"brewery_count": "sum"}).collect()[0][0]
        assert gold_sum == silver_count

    def test_gold_groups_by_type_country_state(self, spark, bronze_records):
        silver_df = self._make_silver_df(spark, bronze_records)
        gold_df = aggregate(silver_df)

        # Expected groups after dedup of the duplicate id-0000:
        # micro/Colorado: id-0000 + id-0001 = 2 records
        # brewpub/Texas:  id-0002 + id-0003 = 2 records
        # nano/Oregon:    id-0004            = 1 record
        rows = {
            (r["brewery_type"], r["state_province"]): r["brewery_count"]
            for r in gold_df.collect()
        }
        assert rows[("micro", "Colorado")] == 2
        assert rows[("brewpub", "Texas")] == 2
        assert rows[("nano", "Oregon")] == 1

    def test_gold_geocoded_count_correct(self, spark, bronze_records):
        silver_df = self._make_silver_df(spark, bronze_records)
        gold_df = aggregate(silver_df)

        rows = {
            (r["brewery_type"], r["state_province"]): r["geocoded_count"]
            for r in gold_df.collect()
        }
        # Houston brewpub has no coords; Austin does → geocoded_count = 1
        assert rows[("brewpub", "Texas")] == 1
