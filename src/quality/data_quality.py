"""
Data quality checks for each pipeline layer.

Three checker classes, one per layer:
    - BronzeQualityChecker  : page file count + total record count vs API meta
    - SilverQualityChecker  : null rates, quarantine rate, coordinate bounds
    - GoldQualityChecker    : sum integrity, no-null brewery_type

Failures are split into two categories:
    - Hard (raises DataQualityException) — pipeline should not proceed
    - Soft (logs a warning)              — acceptable drift, pipeline continues

Usage example (called from Airflow PythonOperator):
    checker = SilverQualityChecker(spark)
    checker.run_all(valid_df, quarantine_df)
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.utils.config import storage_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class DataQualityException(Exception):
    """Raised when a hard data-quality check fails."""


# ---------------------------------------------------------------------------
# Bronze
# ---------------------------------------------------------------------------

class BronzeQualityChecker:
    """Quality checks for the bronze layer."""

    # Tolerate up to 5% record loss vs. the API-reported total
    MIN_RECORD_RATIO = 0.95

    def __init__(self, s3_client, bronze_bucket: str | None = None) -> None:
        self.s3_client = s3_client
        self.bronze_bucket = bronze_bucket or storage_config().bronze_bucket

    def check_page_file_count(self, execution_date: str, expected_pages: int) -> int:
        """
        Count JSON page files for the given date; raise if fewer than expected.

        Returns the actual page count.
        """
        prefix = f"raw/dt={execution_date}/"
        response = self.s3_client.list_objects_v2(
            Bucket=self.bronze_bucket, Prefix=prefix
        )
        actual = len(response.get("Contents", []))
        if actual < expected_pages:
            raise DataQualityException(
                f"Bronze page file count mismatch: expected {expected_pages}, got {actual}"
            )
        log.info("bronze_page_count_ok", expected=expected_pages, actual=actual)
        return actual

    def check_record_count(self, actual_records: int, expected_total: int) -> None:
        """
        Raise if actual records fall below MIN_RECORD_RATIO of expected_total.
        """
        if expected_total == 0:
            log.info("bronze_record_count_skipped", reason="expected_total=0")
            return
        ratio = actual_records / expected_total
        if ratio < self.MIN_RECORD_RATIO:
            raise DataQualityException(
                f"Bronze record count too low: {actual_records}/{expected_total} "
                f"({ratio:.1%} < {self.MIN_RECORD_RATIO:.0%})"
            )
        log.info(
            "bronze_record_count_ok",
            actual=actual_records,
            expected=expected_total,
            ratio=f"{ratio:.1%}",
        )

    def run_all(
        self, execution_date: str, expected_pages: int, actual_records: int, expected_total: int
    ) -> None:
        self.check_page_file_count(execution_date, expected_pages)
        self.check_record_count(actual_records, expected_total)


# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------

class SilverQualityChecker:
    """Quality checks for the silver layer."""

    MAX_NULL_ID_RATE = 0.0          # Hard: any null id in valid set is a bug
    MAX_NULL_TYPE_RATE = 0.05       # Soft: > 5% unknown brewery type is a warning
    MAX_QUARANTINE_RATE = 0.10      # Hard: > 10% quarantine signals an upstream problem
    LAT_BOUNDS = (-90.0, 90.0)
    LON_BOUNDS = (-180.0, 180.0)

    def check_no_null_ids(self, valid_df: DataFrame) -> None:
        """Hard check: valid DataFrame must have zero null ids."""
        null_count = valid_df.filter(F.col("id").isNull()).count()
        if null_count > 0:
            raise DataQualityException(
                f"Silver valid set contains {null_count} rows with null id"
            )
        log.info("silver_null_id_check_ok")

    def check_brewery_type_null_rate(self, valid_df: DataFrame) -> None:
        """Soft warning when brewery_type null rate exceeds threshold."""
        total = valid_df.count()
        if total == 0:
            return
        null_count = valid_df.filter(F.col("brewery_type").isNull()).count()
        rate = null_count / total
        if rate > self.MAX_NULL_TYPE_RATE:
            log.warning(
                "silver_high_null_brewery_type",
                rate=f"{rate:.1%}",
                threshold=f"{self.MAX_NULL_TYPE_RATE:.0%}",
            )
        else:
            log.info("silver_brewery_type_null_rate_ok", rate=f"{rate:.1%}")

    def check_quarantine_rate(self, valid_df: DataFrame, quarantine_df: DataFrame) -> None:
        """Hard check: quarantine must not exceed MAX_QUARANTINE_RATE of total ingested."""
        valid_count = valid_df.count()
        quarantine_count = quarantine_df.count()
        total = valid_count + quarantine_count
        if total == 0:
            return
        rate = quarantine_count / total
        if rate > self.MAX_QUARANTINE_RATE:
            raise DataQualityException(
                f"Silver quarantine rate {rate:.1%} exceeds threshold "
                f"{self.MAX_QUARANTINE_RATE:.0%} ({quarantine_count}/{total} rows)"
            )
        log.info("silver_quarantine_rate_ok", rate=f"{rate:.1%}")

    def check_coordinate_bounds(self, valid_df: DataFrame) -> None:
        """Soft warning when geocoded coordinates fall outside valid ranges."""
        lat_min, lat_max = self.LAT_BOUNDS
        lon_min, lon_max = self.LON_BOUNDS
        bad = valid_df.filter(
            (F.col("latitude").isNotNull() & (
                (F.col("latitude") < lat_min) | (F.col("latitude") > lat_max)
            )) |
            (F.col("longitude").isNotNull() & (
                (F.col("longitude") < lon_min) | (F.col("longitude") > lon_max)
            ))
        ).count()
        if bad > 0:
            log.warning("silver_out_of_bounds_coordinates", count=bad)
        else:
            log.info("silver_coordinate_bounds_ok")

    def run_all(self, valid_df: DataFrame, quarantine_df: DataFrame) -> None:
        self.check_no_null_ids(valid_df)
        self.check_brewery_type_null_rate(valid_df)
        self.check_quarantine_rate(valid_df, quarantine_df)
        self.check_coordinate_bounds(valid_df)


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------

class GoldQualityChecker:
    """Quality checks for the gold layer."""

    def check_sum_integrity(self, gold_df: DataFrame, silver_count: int) -> None:
        """
        Hard check: SUM(brewery_count) in gold must equal silver row count.
        """
        gold_sum = gold_df.agg(F.sum("brewery_count")).collect()[0][0] or 0
        if gold_sum != silver_count:
            raise DataQualityException(
                f"Gold sum integrity failure: SUM(brewery_count)={gold_sum} "
                f"!= silver COUNT(*)={silver_count}"
            )
        log.info("gold_sum_integrity_ok", gold_sum=gold_sum)

    def check_no_null_brewery_type(self, gold_df: DataFrame) -> None:
        """Soft warning when null brewery_type rows exist in gold."""
        null_count = gold_df.filter(F.col("brewery_type").isNull()).count()
        if null_count > 0:
            log.warning("gold_null_brewery_type_rows", count=null_count)
        else:
            log.info("gold_brewery_type_not_null_ok")

    def run_all(self, gold_df: DataFrame, silver_count: int) -> None:
        self.check_sum_integrity(gold_df, silver_count)
        self.check_no_null_brewery_type(gold_df)
