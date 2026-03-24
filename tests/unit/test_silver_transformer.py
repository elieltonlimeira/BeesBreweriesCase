"""Unit tests for silver_transformer.py — all tests use in-memory Spark (no S3)."""

import pytest
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from src.silver.silver_transformer import REDUNDANT_COLUMNS, transform

# Explicit schema avoids CANNOT_DETERMINE_TYPE when all values in a column are None
BREWERY_SCHEMA = StructType([
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(spark, records):
    """Create a DataFrame with explicit schema (handles all-None columns)."""
    return spark.createDataFrame(records, schema=BREWERY_SCHEMA)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_ids_are_removed(self, spark, sample_breweries):
        # sample_breweries has one duplicate id (index 0 and 3 share same id)
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        ids = [r["id"] for r in valid_df.collect()]
        assert len(ids) == len(set(ids)), "Duplicate ids remain after dedup"

    def test_keeps_first_occurrence(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        # The original name (not the duplicate) should be present
        names = [r["name"] for r in valid_df.collect()]
        assert "Ales for ALS" in names
        assert "Ales for ALS (duplicate)" not in names


# ---------------------------------------------------------------------------
# Null routing / quarantine
# ---------------------------------------------------------------------------

class TestNullRouting:
    def test_null_id_goes_to_quarantine(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        valid_df, quarantine_df = transform(df, "2024-03-24")

        quarantine_ids = [r["id"] for r in quarantine_df.collect()]
        assert None in quarantine_ids

    def test_valid_rows_have_no_null_id(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        null_count = valid_df.filter(valid_df["id"].isNull()).count()
        assert null_count == 0

    def test_quarantine_plus_valid_equals_deduplicated_total(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        deduped_count = df.dropDuplicates(["id"]).count()
        valid_df, quarantine_df = transform(df, "2024-03-24")

        assert valid_df.count() + quarantine_df.count() == deduped_count


# ---------------------------------------------------------------------------
# state_province backfill
# ---------------------------------------------------------------------------

class TestStateProvinceBackfill:
    def test_null_state_province_backfilled_from_state(self, spark, sample_breweries):
        # sample_breweries[2] has state_province=None, state="Texas"
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        austin_row = valid_df.filter(valid_df["city"] == "Austin").collect()
        assert len(austin_row) == 1
        assert austin_row[0]["state_province"] == "Texas"

    def test_existing_state_province_not_overwritten(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        norman_row = valid_df.filter(valid_df["city"] == "Norman").collect()
        assert norman_row[0]["state_province"] == "Oklahoma"


# ---------------------------------------------------------------------------
# Redundant column removal
# ---------------------------------------------------------------------------

class TestRedundantColumns:
    def test_redundant_columns_are_dropped(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        remaining = set(valid_df.columns)
        for col in REDUNDANT_COLUMNS:
            assert col not in remaining, f"Column '{col}' should have been dropped"

    def test_address_1_is_kept(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        assert "address_1" in valid_df.columns


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_brewery_type_lowercased(self, spark, sample_breweries):
        # sample_breweries[3] has brewery_type="MICRO" (uppercase duplicate, gets deduped)
        # Let's create a fresh record with uppercase type
        records = [sample_breweries[1].copy()]
        records[0]["brewery_type"] = "MICRO"
        records[0]["id"] = "unique-uppercase-id"

        df = make_df(spark, records)
        valid_df, _ = transform(df, "2024-03-24")

        types = {r["brewery_type"] for r in valid_df.collect()}
        assert all(t == t.lower() for t in types if t is not None)

    def test_country_title_cased(self, spark, sample_breweries):
        records = [sample_breweries[0].copy()]
        records[0]["country"] = "united states"

        df = make_df(spark, records)
        valid_df, _ = transform(df, "2024-03-24")

        row = valid_df.collect()[0]
        assert row["country"] == "United States"

    def test_brewery_type_whitespace_stripped(self, spark):
        records = [{
            "id": "strip-test",
            "name": "Whitespace Brewery",
            "brewery_type": "  micro  ",
            "city": "Test City",
            "state_province": "Texas",
            "state": "Texas",
            "country": "United States",
            "address_1": None, "address_2": None, "address_3": None,
            "postal_code": None, "longitude": None, "latitude": None,
            "phone": None, "website_url": None, "street": None,
        }]
        df = make_df(spark, records)
        valid_df, _ = transform(df, "2024-03-24")

        row = valid_df.collect()[0]
        assert row["brewery_type"] == "micro"


# ---------------------------------------------------------------------------
# Coordinate casting
# ---------------------------------------------------------------------------

class TestCoordinateCasting:
    def test_longitude_is_double_type(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        lon_type = dict(valid_df.dtypes)["longitude"]
        assert lon_type == "double"

    def test_latitude_is_double_type(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        lat_type = dict(valid_df.dtypes)["latitude"]
        assert lat_type == "double"

    def test_null_coordinates_preserved(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        # Austin row has null coords
        austin = valid_df.filter(valid_df["city"] == "Austin").collect()[0]
        assert austin["longitude"] is None
        assert austin["latitude"] is None


# ---------------------------------------------------------------------------
# Metadata column
# ---------------------------------------------------------------------------

class TestMetadataColumn:
    def test_pipeline_run_date_added(self, spark, sample_breweries):
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        assert "pipeline_run_date" in valid_df.columns

    def test_pipeline_run_date_value(self, spark, sample_breweries):
        from datetime import date
        df = make_df(spark, sample_breweries)
        valid_df, _ = transform(df, "2024-03-24")

        dates = {r["pipeline_run_date"] for r in valid_df.collect()}
        assert dates == {date(2024, 3, 24)}
