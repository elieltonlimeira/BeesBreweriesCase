"""Unit tests for gold_aggregator.py — all tests use in-memory Spark (no S3)."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from pyspark.sql.types import (
    DateType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from src.gold.gold_aggregator import aggregate

# Silver schema (redundant cols already dropped by silver transformer)
SILVER_SCHEMA = StructType([
    StructField("id", StringType(), True),
    StructField("name", StringType(), True),
    StructField("brewery_type", StringType(), True),
    StructField("address_1", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state_province", StringType(), True),
    StructField("postal_code", StringType(), True),
    StructField("country", StringType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("phone", StringType(), True),
    StructField("website_url", StringType(), True),
    StructField("pipeline_run_date", DateType(), True),
])


def make_silver_df(spark, records):
    return spark.createDataFrame(records, schema=SILVER_SCHEMA)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

def _base_records():
    """Three breweries in two groups."""
    run_date = date(2024, 3, 24)
    return [
        # Group 1: micro / United States / California — 2 breweries, 2 cities, both geocoded
        {
            "id": "id-1",
            "name": "Brewery One",
            "brewery_type": "micro",
            "address_1": "100 A St",
            "city": "San Diego",
            "state_province": "California",
            "postal_code": "92101",
            "country": "United States",
            "longitude": -117.16,
            "latitude": 32.71,
            "phone": None,
            "website_url": None,
            "pipeline_run_date": run_date,
        },
        {
            "id": "id-2",
            "name": "Brewery Two",
            "brewery_type": "micro",
            "address_1": "200 B St",
            "city": "Los Angeles",
            "state_province": "California",
            "postal_code": "90001",
            "country": "United States",
            "longitude": -118.24,
            "latitude": 34.05,
            "phone": None,
            "website_url": None,
            "pipeline_run_date": run_date,
        },
        # Group 2: brewpub / United States / Texas — 1 brewery, no coordinates
        {
            "id": "id-3",
            "name": "Brewery Three",
            "brewery_type": "brewpub",
            "address_1": "300 C Ave",
            "city": "Austin",
            "state_province": "Texas",
            "postal_code": "78701",
            "country": "United States",
            "longitude": None,
            "latitude": None,
            "phone": None,
            "website_url": None,
            "pipeline_run_date": run_date,
        },
    ]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_expected_columns_present(self, spark):
        df = make_silver_df(spark, _base_records())
        gold = aggregate(df)
        expected = {
            "brewery_type", "country", "state_province",
            "brewery_count", "distinct_city_count", "geocoded_count", "last_updated",
        }
        assert expected == set(gold.columns)

    def test_brewery_count_is_long(self, spark):
        df = make_silver_df(spark, _base_records())
        gold = aggregate(df)
        dtype = dict(gold.dtypes)["brewery_count"]
        assert dtype == "bigint"

    def test_last_updated_is_date(self, spark):
        df = make_silver_df(spark, _base_records())
        gold = aggregate(df)
        dtype = dict(gold.dtypes)["last_updated"]
        assert dtype == "date"


# ---------------------------------------------------------------------------
# Grouping correctness
# ---------------------------------------------------------------------------

class TestGrouping:
    def test_row_count_matches_distinct_groups(self, spark):
        df = make_silver_df(spark, _base_records())
        gold = aggregate(df)
        # 2 distinct (brewery_type, country, state_province) groups
        assert gold.count() == 2

    def test_brewery_count_per_group(self, spark):
        df = make_silver_df(spark, _base_records())
        gold = aggregate(df)

        rows = {
            (r["brewery_type"], r["state_province"]): r["brewery_count"]
            for r in gold.collect()
        }
        assert rows[("micro", "California")] == 2
        assert rows[("brewpub", "Texas")] == 1

    def test_distinct_city_count(self, spark):
        df = make_silver_df(spark, _base_records())
        gold = aggregate(df)

        rows = {
            (r["brewery_type"], r["state_province"]): r["distinct_city_count"]
            for r in gold.collect()
        }
        assert rows[("micro", "California")] == 2  # San Diego + Los Angeles
        assert rows[("brewpub", "Texas")] == 1      # Austin only


# ---------------------------------------------------------------------------
# Geocoded count
# ---------------------------------------------------------------------------

class TestGeocodedCount:
    def test_geocoded_count_only_counts_non_null_latitude(self, spark):
        df = make_silver_df(spark, _base_records())
        gold = aggregate(df)

        rows = {
            (r["brewery_type"], r["state_province"]): r["geocoded_count"]
            for r in gold.collect()
        }
        assert rows[("micro", "California")] == 2   # both have lat
        assert rows[("brewpub", "Texas")] == 0       # no coords

    def test_partial_geocoding_counted_correctly(self, spark):
        """One geocoded, one not — geocoded_count should be 1."""
        run_date = date(2024, 3, 24)
        records = [
            {
                "id": "geo-1",
                "name": "Has Coords",
                "brewery_type": "micro",
                "address_1": None,
                "city": "Denver",
                "state_province": "Colorado",
                "postal_code": None,
                "country": "United States",
                "longitude": -104.99,
                "latitude": 39.73,
                "phone": None,
                "website_url": None,
                "pipeline_run_date": run_date,
            },
            {
                "id": "no-geo-1",
                "name": "No Coords",
                "brewery_type": "micro",
                "address_1": None,
                "city": "Boulder",
                "state_province": "Colorado",
                "postal_code": None,
                "country": "United States",
                "longitude": None,
                "latitude": None,
                "phone": None,
                "website_url": None,
                "pipeline_run_date": run_date,
            },
        ]
        df = make_silver_df(spark, records)
        gold = aggregate(df)
        row = gold.collect()[0]
        assert row["geocoded_count"] == 1


# ---------------------------------------------------------------------------
# last_updated (max pipeline_run_date)
# ---------------------------------------------------------------------------

class TestLastUpdated:
    def test_last_updated_is_max_run_date(self, spark):
        """When two run dates exist, last_updated should be the latest."""
        records = [
            {
                "id": "id-a",
                "name": "A",
                "brewery_type": "micro",
                "address_1": None,
                "city": "Tulsa",
                "state_province": "Oklahoma",
                "postal_code": None,
                "country": "United States",
                "longitude": None,
                "latitude": None,
                "phone": None,
                "website_url": None,
                "pipeline_run_date": date(2024, 3, 20),
            },
            {
                "id": "id-b",
                "name": "B",
                "brewery_type": "micro",
                "address_1": None,
                "city": "Tulsa",
                "state_province": "Oklahoma",
                "postal_code": None,
                "country": "United States",
                "longitude": None,
                "latitude": None,
                "phone": None,
                "website_url": None,
                "pipeline_run_date": date(2024, 3, 24),
            },
        ]
        df = make_silver_df(spark, records)
        gold = aggregate(df)
        row = gold.collect()[0]
        assert row["last_updated"] == date(2024, 3, 24)


# ---------------------------------------------------------------------------
# Duplicate city names within same group
# ---------------------------------------------------------------------------

class TestDuplicateCities:
    def test_same_city_counted_once(self, spark):
        """Two breweries in the same city → distinct_city_count = 1."""
        run_date = date(2024, 3, 24)
        records = [
            {
                "id": "x-1",
                "name": "Norman 1",
                "brewery_type": "micro",
                "address_1": None,
                "city": "Norman",
                "state_province": "Oklahoma",
                "postal_code": None,
                "country": "United States",
                "longitude": -97.46,
                "latitude": 35.25,
                "phone": None,
                "website_url": None,
                "pipeline_run_date": run_date,
            },
            {
                "id": "x-2",
                "name": "Norman 2",
                "brewery_type": "micro",
                "address_1": None,
                "city": "Norman",
                "state_province": "Oklahoma",
                "postal_code": None,
                "country": "United States",
                "longitude": -97.47,
                "latitude": 35.26,
                "phone": None,
                "website_url": None,
                "pipeline_run_date": run_date,
            },
        ]
        df = make_silver_df(spark, records)
        gold = aggregate(df)
        row = gold.collect()[0]
        assert row["distinct_city_count"] == 1
        assert row["brewery_count"] == 2


# ---------------------------------------------------------------------------
# Sum integrity — gold total must equal silver row count
# ---------------------------------------------------------------------------

class TestSumIntegrity:
    def test_sum_of_brewery_counts_equals_silver_row_count(self, spark):
        df = make_silver_df(spark, _base_records())
        gold = aggregate(df)

        silver_count = df.count()
        gold_sum = gold.agg({"brewery_count": "sum"}).collect()[0][0]
        assert gold_sum == silver_count


# ---------------------------------------------------------------------------
# I/O functions (S3 interactions tested via mocks)
# ---------------------------------------------------------------------------

class TestReadSilver:
    @patch("src.gold.gold_aggregator.storage_config")
    def test_reads_parquet_from_correct_s3_path(self, mock_config):
        mock_config.return_value = MagicMock(silver_bucket="test-silver")
        mock_spark = MagicMock()

        from src.gold.gold_aggregator import read_silver

        read_silver(mock_spark)

        mock_spark.read.parquet.assert_called_once_with("s3a://test-silver/breweries/")


class TestWriteGold:
    @patch("src.gold.gold_aggregator.storage_config")
    def test_writes_parquet_to_correct_path_and_returns_count(self, mock_config):
        mock_config.return_value = MagicMock(gold_bucket="test-gold")
        mock_df = MagicMock()
        mock_cached_df = MagicMock()
        mock_df.cache.return_value = mock_cached_df
        mock_cached_df.count.return_value = 5
        mock_writer = MagicMock()
        mock_cached_df.write.mode.return_value = mock_writer

        from src.gold.gold_aggregator import write_gold

        count = write_gold(mock_df, "2024-03-24")

        mock_df.cache.assert_called_once()
        mock_cached_df.write.mode.assert_called_once_with("overwrite")
        mock_writer.parquet.assert_called_once_with(
            "s3a://test-gold/brewery_counts/dt=2024-03-24/"
        )
        mock_cached_df.unpersist.assert_called_once()
        assert count == 5


class TestGoldRun:
    @patch("src.gold.gold_aggregator.write_gold", return_value=5)
    @patch("src.gold.gold_aggregator.aggregate")
    @patch("src.gold.gold_aggregator.read_silver")
    @patch("src.gold.gold_aggregator.get_spark_session")
    def test_orchestrates_all_steps(
        self, mock_get_spark, mock_read, mock_agg, mock_write
    ):
        mock_spark = MagicMock()
        mock_get_spark.return_value = mock_spark
        mock_silver_df = MagicMock()
        mock_read.return_value = mock_silver_df
        mock_gold_df = MagicMock()
        mock_agg.return_value = mock_gold_df

        from src.gold.gold_aggregator import run

        result = run("2024-03-24")

        mock_get_spark.assert_called_once_with("BreweryPipeline-Gold")
        mock_read.assert_called_once_with(mock_spark)
        mock_agg.assert_called_once_with(mock_silver_df)
        mock_write.assert_called_once_with(mock_gold_df, "2024-03-24")
        assert result == {"gold_rows": 5, "execution_date": "2024-03-24"}


# ---------------------------------------------------------------------------
# __main__ entrypoint
# ---------------------------------------------------------------------------

class TestGoldMain:
    @patch("src.gold.gold_aggregator.run")
    def test_main_calls_run_with_date_arg(self, mock_run):
        mock_run.return_value = {}
        with patch("sys.argv", ["gold_aggregator", "2024-03-24"]):
            from src.gold.gold_aggregator import main
            main()
        mock_run.assert_called_once_with("2024-03-24")

    def test_main_exits_without_args(self):
        with patch("sys.argv", ["gold_aggregator"]):
            with pytest.raises(SystemExit) as exc_info:
                from src.gold.gold_aggregator import main
                main()
            assert exc_info.value.code == 1
