from pyspark.sql import SparkSession

from src.utils.config import storage_config


def get_spark_session(app_name: str = "BreweryPipeline") -> SparkSession:
    """
    Build and return a SparkSession configured for S3A (MinIO/S3) access.

    Credentials are read via the centralized config module, which validates
    that all required env vars are set before the session is created.

    The session uses getOrCreate(), so calling this multiple times in the
    same JVM process always returns the same session (Spark singleton).
    App name is set only on first creation.
    """
    cfg = storage_config()

    spark = (
        SparkSession.builder.appName(app_name)
        # S3A connector — path-style access is required for MinIO
        .config("spark.hadoop.fs.s3a.endpoint", cfg.endpoint)
        .config("spark.hadoop.fs.s3a.access.key", cfg.access_key)
        .config("spark.hadoop.fs.s3a.secret.key", cfg.secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        # Point Spark to the pre-downloaded Hadoop AWS JARs in the Docker image
        .config("spark.driver.extraClassPath", "/opt/spark/jars/*")
        .config("spark.executor.extraClassPath", "/opt/spark/jars/*")
        # Small shuffle partition count — dataset is ~9k rows, not petabytes
        .config("spark.sql.shuffle.partitions", "8")
        # Explicit integer — prevents "60s" format from being passed to getLong()
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
