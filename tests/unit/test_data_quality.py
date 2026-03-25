"""Unit tests for data_quality.py."""

from datetime import date
from unittest.mock import MagicMock

import pytest
from pyspark.sql.types import (
    DateType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from src.quality.data_quality import (
    BronzeQualityChecker,
    DataQualityException,
    GoldQualityChecker,
    SilverQualityChecker,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

SILVER_SCHEMA = StructType([
    StructField("id", StringType(), True),
    StructField("brewery_type", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state_province", StringType(), True),
    StructField("country", StringType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("pipeline_run_date", DateType(), True),
])

GOLD_SCHEMA = StructType([
    StructField("brewery_type", StringType(), True),
    StructField("country", StringType(), True),
    StructField("state_province", StringType(), True),
    StructField("brewery_count", LongType(), True),
    StructField("distinct_city_count", LongType(), True),
    StructField("geocoded_count", LongType(), True),
    StructField("last_updated", DateType(), True),
])


def _silver_row(
    id_val="id-1",
    brewery_type="micro",
    city="Denver",
    state_province="Colorado",
    country="United States",
    longitude=-104.99,
    latitude=39.73,
    run_date=date(2024, 3, 24),
):
    return {
        "id": id_val,
        "brewery_type": brewery_type,
        "city": city,
        "state_province": state_province,
        "country": country,
        "longitude": longitude,
        "latitude": latitude,
        "pipeline_run_date": run_date,
    }


def _gold_row(
    brewery_type="micro",
    country="United States",
    state_province="Colorado",
    brewery_count=3,
    distinct_city_count=2,
    geocoded_count=3,
    last_updated=date(2024, 3, 24),
):
    return {
        "brewery_type": brewery_type,
        "country": country,
        "state_province": state_province,
        "brewery_count": brewery_count,
        "distinct_city_count": distinct_city_count,
        "geocoded_count": geocoded_count,
        "last_updated": last_updated,
    }


# ---------------------------------------------------------------------------
# BronzeQualityChecker
# ---------------------------------------------------------------------------

class TestBronzePageFileCount:
    def _make_checker(self, page_count: int) -> BronzeQualityChecker:
        client = MagicMock()
        client.list_objects_v2.return_value = {
            "Contents": [{"Key": f"page={i:03d}.json"} for i in range(page_count)]
        }
        return BronzeQualityChecker(s3_client=client, bronze_bucket="test-bronze")

    def test_passes_when_count_matches(self):
        checker = self._make_checker(page_count=47)
        result = checker.check_page_file_count("2024-03-24", expected_pages=47)
        assert result == 47

    def test_passes_when_count_exceeds_expected(self):
        checker = self._make_checker(page_count=50)
        result = checker.check_page_file_count("2024-03-24", expected_pages=47)
        assert result == 50

    def test_raises_when_count_below_expected(self):
        checker = self._make_checker(page_count=10)
        with pytest.raises(DataQualityException, match="page file count mismatch"):
            checker.check_page_file_count("2024-03-24", expected_pages=47)

    def test_empty_prefix_raises(self):
        client = MagicMock()
        client.list_objects_v2.return_value = {}
        checker = BronzeQualityChecker(s3_client=client, bronze_bucket="test-bronze")
        with pytest.raises(DataQualityException):
            checker.check_page_file_count("2024-03-24", expected_pages=1)


class TestBronzeRecordCount:
    def _checker(self):
        return BronzeQualityChecker(s3_client=MagicMock(), bronze_bucket="test-bronze")

    def test_passes_when_above_threshold(self):
        checker = self._checker()
        checker.check_record_count(actual_records=9000, expected_total=9383)  # ~96%

    def test_passes_at_exact_threshold(self):
        checker = self._checker()
        checker.check_record_count(actual_records=8914, expected_total=9383)  # ~95%

    def test_raises_when_below_threshold(self):
        checker = self._checker()
        with pytest.raises(DataQualityException, match="record count too low"):
            checker.check_record_count(actual_records=5000, expected_total=9383)

    def test_zero_expected_does_not_raise(self):
        checker = self._checker()
        checker.check_record_count(actual_records=0, expected_total=0)


# ---------------------------------------------------------------------------
# SilverQualityChecker
# ---------------------------------------------------------------------------

class TestSilverNullIds:
    def test_passes_with_no_null_ids(self, spark):
        rows = [_silver_row("id-1"), _silver_row("id-2")]
        df = spark.createDataFrame(rows, schema=SILVER_SCHEMA)
        checker = SilverQualityChecker()
        checker.check_no_null_ids(df)  # should not raise

    def test_raises_with_null_id(self, spark):
        rows = [_silver_row(id_val=None)]
        df = spark.createDataFrame(rows, schema=SILVER_SCHEMA)
        checker = SilverQualityChecker()
        with pytest.raises(DataQualityException, match="null id"):
            checker.check_no_null_ids(df)


class TestSilverBreweryTypeNullRate:
    def test_no_warning_below_threshold(self, spark):
        rows = [_silver_row(brewery_type="micro") for _ in range(10)]
        df = spark.createDataFrame(rows, schema=SILVER_SCHEMA)
        SilverQualityChecker().check_brewery_type_null_rate(df)  # should not raise

    def test_handles_empty_dataframe(self, spark):
        df = spark.createDataFrame([], schema=SILVER_SCHEMA)
        SilverQualityChecker().check_brewery_type_null_rate(df)  # no division by zero


class TestSilverQuarantineRate:
    def test_passes_when_quarantine_below_threshold(self, spark):
        valid_rows = [_silver_row(id_val=f"id-{i}") for i in range(95)]
        quarantine_rows = [_silver_row(id_val=None) for _ in range(5)]
        valid_df = spark.createDataFrame(valid_rows, schema=SILVER_SCHEMA)
        quarantine_df = spark.createDataFrame(quarantine_rows, schema=SILVER_SCHEMA)
        SilverQualityChecker().check_quarantine_rate(valid_df, quarantine_df)

    def test_raises_when_quarantine_above_threshold(self, spark):
        valid_rows = [_silver_row(id_val=f"id-{i}") for i in range(50)]
        quarantine_rows = [_silver_row(id_val=None) for _ in range(60)]
        valid_df = spark.createDataFrame(valid_rows, schema=SILVER_SCHEMA)
        quarantine_df = spark.createDataFrame(quarantine_rows, schema=SILVER_SCHEMA)
        with pytest.raises(DataQualityException, match="quarantine rate"):
            SilverQualityChecker().check_quarantine_rate(valid_df, quarantine_df)

    def test_empty_dataframes_do_not_raise(self, spark):
        empty = spark.createDataFrame([], schema=SILVER_SCHEMA)
        SilverQualityChecker().check_quarantine_rate(empty, empty)


class TestSilverCoordinateBounds:
    def test_valid_coordinates_pass(self, spark):
        rows = [_silver_row(latitude=45.0, longitude=-93.0)]
        df = spark.createDataFrame(rows, schema=SILVER_SCHEMA)
        SilverQualityChecker().check_coordinate_bounds(df)  # should not raise

    def test_null_coordinates_ignored(self, spark):
        rows = [_silver_row(latitude=None, longitude=None)]
        df = spark.createDataFrame(rows, schema=SILVER_SCHEMA)
        SilverQualityChecker().check_coordinate_bounds(df)  # should not raise

    def test_out_of_bounds_latitude_triggers_warning(self, spark, caplog):
        rows = [_silver_row(latitude=95.0, longitude=0.0)]
        df = spark.createDataFrame(rows, schema=SILVER_SCHEMA)
        # Should log a warning, not raise
        SilverQualityChecker().check_coordinate_bounds(df)

    def test_out_of_bounds_longitude_triggers_warning(self, spark, caplog):
        rows = [_silver_row(latitude=0.0, longitude=200.0)]
        df = spark.createDataFrame(rows, schema=SILVER_SCHEMA)
        SilverQualityChecker().check_coordinate_bounds(df)


# ---------------------------------------------------------------------------
# GoldQualityChecker
# ---------------------------------------------------------------------------

class TestGoldSumIntegrity:
    def test_passes_when_sum_matches_silver_count(self, spark):
        rows = [_gold_row(brewery_count=5), _gold_row(brewery_type="brewpub", brewery_count=3)]
        gold_df = spark.createDataFrame(rows, schema=GOLD_SCHEMA)
        GoldQualityChecker().check_sum_integrity(gold_df, silver_count=8)

    def test_raises_when_sum_does_not_match(self, spark):
        rows = [_gold_row(brewery_count=5)]
        gold_df = spark.createDataFrame(rows, schema=GOLD_SCHEMA)
        with pytest.raises(DataQualityException, match="sum integrity"):
            GoldQualityChecker().check_sum_integrity(gold_df, silver_count=99)


class TestGoldNullBreweryType:
    def test_passes_with_no_null_types(self, spark):
        rows = [_gold_row(brewery_type="micro")]
        gold_df = spark.createDataFrame(rows, schema=GOLD_SCHEMA)
        GoldQualityChecker().check_no_null_brewery_type(gold_df)  # should not raise

    def test_null_type_logs_warning_but_does_not_raise(self, spark):
        rows = [_gold_row(brewery_type=None)]
        gold_df = spark.createDataFrame(rows, schema=GOLD_SCHEMA)
        GoldQualityChecker().check_no_null_brewery_type(gold_df)  # soft check, no raise


class TestGoldRunAll:
    def test_run_all_passes_clean_data(self, spark):
        rows = [_gold_row(brewery_count=3)]
        gold_df = spark.createDataFrame(rows, schema=GOLD_SCHEMA)
        GoldQualityChecker().run_all(gold_df, silver_count=3)

    def test_run_all_raises_on_sum_mismatch(self, spark):
        rows = [_gold_row(brewery_count=3)]
        gold_df = spark.createDataFrame(rows, schema=GOLD_SCHEMA)
        with pytest.raises(DataQualityException):
            GoldQualityChecker().run_all(gold_df, silver_count=10)
