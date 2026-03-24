"""
Silver layer transformer.

Reads raw JSON from the bronze bucket, applies transformations, and writes
Parquet partitioned by (country, state_province) to the silver bucket.

Transformations applied (in order):
    1. Deduplication by `id` (keeps first occurrence)
    2. Null routing — rows with null `id` go to quarantine
    3. state_province backfill from `state` when null
    4. Drop redundant columns: state, street, address_2, address_3
    5. Normalize brewery_type → lowercase + stripped
    6. Normalize country → title case + stripped
    7. Cast longitude / latitude to DoubleType
    8. Add pipeline_run_date metadata column

Usage (called via spark-submit from Airflow BashOperator):
    python -m src.silver.silver_transformer 2024-03-24
"""

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.utils.config import storage_config
from src.utils.logger import get_logger
from src.utils.spark_session import get_spark_session

log = get_logger(__name__)

# Columns present in the API response that are redundant given other fields.
# `state` duplicates `state_province`; `street` duplicates `address_1`.
REDUNDANT_COLUMNS = ["state", "street", "address_2", "address_3"]


def read_bronze(spark: SparkSession, execution_date: str) -> DataFrame:
    """Read all JSON page files for the given execution date from bronze."""
    cfg = storage_config()
    path = f"s3a://{cfg.bronze_bucket}/raw/dt={execution_date}/"
    log.info("reading_bronze", path=path)
    return spark.read.option("multiline", "true").json(path)


def transform(df: DataFrame, execution_date: str) -> tuple[DataFrame, DataFrame]:
    """
    Apply all silver transformations and split into valid + quarantine sets.

    Returns:
        (valid_df, quarantine_df)
    """
    # 1. Deduplication — keep first occurrence per id
    df = df.dropDuplicates(["id"])

    # 2. Null routing — rows without an id cannot be linked to anything
    quarantine_df = df.filter(F.col("id").isNull())
    df = df.filter(F.col("id").isNotNull())

    # 3. Backfill state_province from state when null
    df = df.withColumn(
        "state_province",
        F.coalesce(F.col("state_province"), F.col("state")),
    )

    # 4. Drop redundant columns (only those present in the DataFrame)
    cols_to_drop = [c for c in REDUNDANT_COLUMNS if c in df.columns]
    df = df.drop(*cols_to_drop)

    # 5. Normalize brewery_type to lowercase
    df = df.withColumn(
        "brewery_type",
        F.lower(F.trim(F.col("brewery_type"))),
    )

    # 6. Normalize country to title case
    df = df.withColumn(
        "country",
        F.initcap(F.trim(F.col("country"))),
    )

    # 7. Cast coordinates — JSON inference may read as strings in edge cases
    df = df.withColumn("longitude", F.col("longitude").cast(DoubleType()))
    df = df.withColumn("latitude", F.col("latitude").cast(DoubleType()))

    # 8. Add pipeline metadata column
    df = df.withColumn(
        "pipeline_run_date",
        F.lit(execution_date).cast("date"),
    )

    return df, quarantine_df


def write_silver(df: DataFrame) -> int:
    """Write valid records partitioned by country and state_province."""
    cfg = storage_config()
    path = f"s3a://{cfg.silver_bucket}/breweries/"
    log.info("writing_silver", path=path)
    df.write.mode("overwrite").partitionBy("country", "state_province").parquet(path)
    count = df.count()
    log.info("silver_written", records=count)
    return count


def write_quarantine(df: DataFrame, execution_date: str) -> int:
    """Write quarantined records to a separate path for investigation."""
    cfg = storage_config()
    path = f"s3a://{cfg.silver_bucket}/quarantine/dt={execution_date}/"
    count = df.count()
    if count > 0:
        df.write.mode("overwrite").parquet(path)
        log.warning("quarantine_records_written", count=count, path=path)
    return count


def run(execution_date: str) -> dict:
    """Entry point for the silver transformation job."""
    spark = get_spark_session("BreweryPipeline-Silver")
    bronze_df = read_bronze(spark, execution_date)
    valid_df, quarantine_df = transform(bronze_df, execution_date)
    valid_count = write_silver(valid_df)
    quarantine_count = write_quarantine(quarantine_df, execution_date)
    result = {
        "valid_records": valid_count,
        "quarantine_records": quarantine_count,
        "execution_date": execution_date,
    }
    log.info("silver_job_complete", **result)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.silver.silver_transformer YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])
