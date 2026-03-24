import os

from pyspark.sql import SparkSession


def get_spark_session(app_name: str = "BreweryPipeline") -> SparkSession:
    """
    Build and return a SparkSession configured for S3A (MinIO/S3) access.

    S3A credentials and endpoint are read from environment variables so the
    same code runs locally (MinIO) and in production (AWS S3) without changes.
    """
    endpoint = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")

    spark = (
        SparkSession.builder.appName(app_name)
        # S3A connector — use path-style access required for MinIO
        .config("spark.hadoop.fs.s3a.endpoint", endpoint)
        .config("spark.hadoop.fs.s3a.access.key", access_key)
        .config("spark.hadoop.fs.s3a.secret.key", secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        # Avoid SSL errors with local MinIO
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # Reduce shuffle partitions for small datasets
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
