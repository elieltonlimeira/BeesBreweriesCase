"""
Gold layer aggregator.

Reads the silver Parquet dataset and produces a compact aggregation table
that answers: "how many breweries of each type exist, per country and state?"

Aggregations computed:
    - brewery_count       : total breweries in the group
    - distinct_city_count : number of distinct cities in the group
    - geocoded_count      : breweries with non-null latitude (have coordinates)
    - last_updated        : latest pipeline_run_date seen in the group

Usage (called via spark-submit from Airflow BashOperator):
    python -m src.gold.gold_aggregator 2024-03-24
"""

import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.utils.config import storage_config
from src.utils.logger import get_logger
from src.utils.spark_session import get_spark_session

log = get_logger(__name__)


def read_silver(spark: SparkSession) -> DataFrame:
    """Read the full silver breweries dataset."""
    cfg = storage_config()
    path = f"s3a://{cfg.silver_bucket}/breweries/"
    log.info("reading_silver", path=path)
    return spark.read.parquet(path)


def aggregate(df: DataFrame) -> DataFrame:
    """
    Group by (brewery_type, country, state_province) and compute metrics.

    Returns a DataFrame with columns:
        brewery_type, country, state_province,
        brewery_count, distinct_city_count, geocoded_count, last_updated
    """
    return df.groupBy("brewery_type", "country", "state_province").agg(
        F.count("id").alias("brewery_count"),
        F.countDistinct("city").alias("distinct_city_count"),
        F.sum(F.when(F.col("latitude").isNotNull(), 1).otherwise(0)).alias("geocoded_count"),
        F.max("pipeline_run_date").alias("last_updated"),
    )


def write_gold(df: DataFrame, execution_date: str) -> int:
    """Write the aggregated table to the gold bucket."""
    cfg = storage_config()
    path = f"s3a://{cfg.gold_bucket}/brewery_counts/dt={execution_date}/"
    log.info("writing_gold", path=path)
    df.write.mode("overwrite").parquet(path)
    count = df.count()
    log.info("gold_written", rows=count)
    return count


def run(execution_date: str) -> dict:
    """Entry point for the gold aggregation job."""
    spark = get_spark_session("BreweryPipeline-Gold")
    silver_df = read_silver(spark)
    gold_df = aggregate(silver_df)
    row_count = write_gold(gold_df, execution_date)
    result = {
        "gold_rows": row_count,
        "execution_date": execution_date,
    }
    log.info("gold_job_complete", **result)
    return result


def main() -> None:
    """CLI entrypoint — parses argv and delegates to run()."""
    if len(sys.argv) != 2:
        print("Usage: python -m src.gold.gold_aggregator YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])


if __name__ == "__main__":
    main()
