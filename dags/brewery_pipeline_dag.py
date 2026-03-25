"""
Airflow DAG: brewery_data_pipeline

Orchestrates the full brewery data pipeline:
    1. fetch_meta          — query the API for total brewery count → XCom
    2. fetch_bronze_pages  — paginated API fetch, one task per page (dynamic mapping)
    3. validate_bronze     — page file count + record count quality checks
    4. transform_silver    — PySpark transformation via spark-submit (BashOperator)
    5. validate_silver     — null rates + quarantine rate + coordinate bounds
    6. aggregate_gold      — PySpark aggregation via spark-submit (BashOperator)
    7. validate_gold       — sum integrity + null brewery_type check

Schedule: daily at 06:00 UTC.  No catchup (API returns current snapshot only).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

from src.ingestion.brewery_api_client import (
    calculate_total_pages,
    fetch_brewery_meta,
    fetch_brewery_page,
)
from src.bronze.bronze_writer import write_page
from src.quality.data_quality import (
    BronzeQualityChecker,
    DataQualityException,
    GoldQualityChecker,
    SilverQualityChecker,
)
from src.utils.config import api_config, storage_config
from src.utils.storage_client import get_s3_client

# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------

_DEFAULT_ARGS = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

# spark-submit command template — executed inside the Airflow container which
# has Java + Hadoop AWS JARs installed (see docker/airflow/Dockerfile)
_SPARK_SUBMIT = (
    "spark-submit"
    " --master local[2]"
    " --conf spark.sql.shuffle.partitions=8"
    " --conf spark.hadoop.fs.s3a.endpoint=$MINIO_ENDPOINT"
    " --conf spark.hadoop.fs.s3a.access.key=$MINIO_ACCESS_KEY"
    " --conf spark.hadoop.fs.s3a.secret.key=$MINIO_SECRET_KEY"
    " --conf spark.hadoop.fs.s3a.path.style.access=true"
    " --conf spark.hadoop.fs.s3a.impl=org.apache.hadoop.fs.s3a.S3AFileSystem"
    " --conf 'spark.hadoop.fs.s3a.aws.credentials.provider="
    "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider'"
    " --conf spark.hadoop.fs.s3a.connection.ssl.enabled=false"
    " --driver-class-path /opt/spark/jars/*"
    " --conf spark.executor.extraClassPath=/opt/spark/jars/*"
    " -m {module} {execution_date}"
)


# ---------------------------------------------------------------------------
# DAG
# ---------------------------------------------------------------------------

@dag(
    dag_id="brewery_data_pipeline",
    description="Fetch Open Brewery DB data and build bronze → silver → gold layers.",
    schedule="0 6 * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    default_args=_DEFAULT_ARGS,
    tags=["brewery", "etl"],
)
def brewery_pipeline():

    # -----------------------------------------------------------------------
    # Task 1 — fetch API meta (total count + pages)
    # -----------------------------------------------------------------------

    @task
    def fetch_meta(execution_date: str | None = None) -> dict:
        """
        Query the /breweries/meta endpoint and compute the total page count.

        Returns a dict pushed to XCom:
            {"total": 9383, "total_pages": 47, "per_page": 200}
        """
        cfg = api_config()
        meta = fetch_brewery_meta(base_url=cfg.base_url)
        total = meta["total"]
        total_pages = calculate_total_pages(total, cfg.page_size)
        return {
            "total": total,
            "total_pages": total_pages,
            "per_page": cfg.page_size,
        }

    # -----------------------------------------------------------------------
    # Task 2 — fetch bronze pages (dynamically mapped)
    # -----------------------------------------------------------------------

    @task(max_active_tis_per_dag=10)
    def fetch_bronze_page(page: int, execution_date: str | None = None) -> dict:
        """
        Fetch one page from the API and persist it as JSON in the bronze bucket.

        Returns a write-result dict for XCom (key, records_written, bytes_written).
        """
        cfg = api_config()
        records = fetch_brewery_page(page=page, per_page=cfg.page_size, base_url=cfg.base_url)
        s3 = get_s3_client()
        return write_page(
            records=records,
            page=page,
            execution_date=execution_date,
            s3_client=s3,
            overwrite=False,  # idempotent re-runs skip existing pages
        )

    # -----------------------------------------------------------------------
    # Task 3 — bronze quality check
    # -----------------------------------------------------------------------

    @task
    def validate_bronze(meta: dict, page_results: list[dict], execution_date: str | None = None) -> None:
        """
        Verify:
          - All expected page files are present in S3.
          - Total records written ≥ 95% of API-reported total.
        Raises DataQualityException (hard failure) on violation.
        """
        checker = BronzeQualityChecker(s3_client=get_s3_client())
        total_records = sum(r.get("records_written", 0) for r in page_results)
        checker.run_all(
            execution_date=execution_date,
            expected_pages=meta["total_pages"],
            actual_records=total_records,
            expected_total=meta["total"],
        )

    # -----------------------------------------------------------------------
    # Task 4 — transform silver (spark-submit via BashOperator)
    # -----------------------------------------------------------------------

    transform_silver = BashOperator(
        task_id="transform_silver",
        bash_command=_SPARK_SUBMIT.format(
            module="src.silver.silver_transformer",
            execution_date="{{ ds }}",
        ),
        env={**os.environ},
    )

    # -----------------------------------------------------------------------
    # Task 5 — silver quality check
    # -----------------------------------------------------------------------

    @task
    def validate_silver(execution_date: str | None = None) -> None:
        """
        Read the silver valid + quarantine DataFrames and run quality checks.
        Requires an active SparkSession — spawns one via get_spark_session().
        """
        from pyspark.sql import SparkSession

        from src.utils.spark_session import get_spark_session

        cfg = storage_config()
        spark = get_spark_session("BreweryPipeline-QC-Silver")

        valid_path = f"s3a://{cfg.silver_bucket}/breweries/"
        quarantine_path = f"s3a://{cfg.silver_bucket}/quarantine/dt={execution_date}/"

        valid_df = spark.read.parquet(valid_path)

        try:
            quarantine_df = spark.read.parquet(quarantine_path)
        except Exception:
            # No quarantine file is acceptable (zero bad rows)
            quarantine_df = spark.createDataFrame([], valid_df.schema)

        SilverQualityChecker().run_all(valid_df, quarantine_df)

    # -----------------------------------------------------------------------
    # Task 6 — aggregate gold (spark-submit via BashOperator)
    # -----------------------------------------------------------------------

    aggregate_gold = BashOperator(
        task_id="aggregate_gold",
        bash_command=_SPARK_SUBMIT.format(
            module="src.gold.gold_aggregator",
            execution_date="{{ ds }}",
        ),
        env={**os.environ},
    )

    # -----------------------------------------------------------------------
    # Task 7 — gold quality check
    # -----------------------------------------------------------------------

    @task
    def validate_gold(execution_date: str | None = None) -> None:
        """
        Read the gold and silver DataFrames and run sum-integrity check.
        """
        from src.utils.spark_session import get_spark_session

        cfg = storage_config()
        spark = get_spark_session("BreweryPipeline-QC-Gold")

        silver_df = spark.read.parquet(f"s3a://{cfg.silver_bucket}/breweries/")
        gold_df = spark.read.parquet(
            f"s3a://{cfg.gold_bucket}/brewery_counts/dt={execution_date}/"
        )

        GoldQualityChecker().run_all(gold_df, silver_count=silver_df.count())

    # -----------------------------------------------------------------------
    # Wire up the dependency chain
    # -----------------------------------------------------------------------

    meta = fetch_meta()

    # Dynamic mapping: one task instance per page number
    pages = fetch_bronze_page.expand(page=meta["total_pages"])

    bronze_ok = validate_bronze(meta=meta, page_results=pages)
    bronze_ok >> transform_silver
    transform_silver >> validate_silver()
    validate_silver() >> aggregate_gold
    aggregate_gold >> validate_gold()


# Instantiate the DAG
brewery_pipeline()
