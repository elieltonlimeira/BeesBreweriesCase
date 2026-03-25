"""Unit tests for spark_session.py."""

from unittest.mock import MagicMock, patch


class TestGetSparkSession:
    @patch("src.utils.spark_session.storage_config")
    @patch("src.utils.spark_session.SparkSession")
    def test_builds_session_with_s3a_config(self, mock_spark_cls, mock_config):
        mock_config.return_value = MagicMock(
            endpoint="http://minio:9000",
            access_key="ak",
            secret_key="sk",
        )
        mock_builder = MagicMock()
        mock_spark_cls.builder = mock_builder
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_session = MagicMock()
        mock_builder.getOrCreate.return_value = mock_session

        from src.utils.spark_session import get_spark_session

        result = get_spark_session("TestApp")

        mock_builder.appName.assert_called_once_with("TestApp")
        mock_builder.getOrCreate.assert_called_once()
        mock_session.sparkContext.setLogLevel.assert_called_once_with("WARN")
        assert result is mock_session

        config_calls = {
            call.args[0]: call.args[1]
            for call in mock_builder.config.call_args_list
        }
        assert config_calls["spark.hadoop.fs.s3a.endpoint"] == "http://minio:9000"
        assert config_calls["spark.hadoop.fs.s3a.access.key"] == "ak"
        assert config_calls["spark.hadoop.fs.s3a.secret.key"] == "sk"
        assert config_calls["spark.hadoop.fs.s3a.path.style.access"] == "true"

    @patch("src.utils.spark_session.storage_config")
    @patch("src.utils.spark_session.SparkSession")
    def test_uses_default_app_name(self, mock_spark_cls, mock_config):
        mock_config.return_value = MagicMock(
            endpoint="http://minio:9000", access_key="ak", secret_key="sk"
        )
        mock_builder = MagicMock()
        mock_spark_cls.builder = mock_builder
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = MagicMock()

        from src.utils.spark_session import get_spark_session

        get_spark_session()

        mock_builder.appName.assert_called_once_with("BreweryPipeline")
