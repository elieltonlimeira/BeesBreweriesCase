"""Unit tests for storage_client.py."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.utils import storage_client as sc


class TestGetS3Client:
    def setup_method(self):
        sc.get_s3_client.cache_clear()

    def teardown_method(self):
        sc.get_s3_client.cache_clear()

    @patch("src.utils.storage_client.boto3.client")
    @patch("src.utils.storage_client.storage_config")
    def test_creates_client_with_config_values(self, mock_config, mock_boto_client):
        mock_config.return_value = MagicMock(
            endpoint="http://localhost:9000",
            access_key="test-ak",
            secret_key="test-sk",
        )
        mock_boto_client.return_value = MagicMock()

        result = sc.get_s3_client()

        mock_boto_client.assert_called_once_with(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="test-ak",
            aws_secret_access_key="test-sk",
            region_name="us-east-1",
        )
        assert result is mock_boto_client.return_value

    @patch("src.utils.storage_client.boto3.client")
    @patch("src.utils.storage_client.storage_config")
    def test_returns_cached_client_on_second_call(self, mock_config, mock_boto_client):
        mock_config.return_value = MagicMock(
            endpoint="http://localhost:9000", access_key="ak", secret_key="sk"
        )
        mock_boto_client.return_value = MagicMock()

        c1 = sc.get_s3_client()
        c2 = sc.get_s3_client()

        assert c1 is c2
        assert mock_boto_client.call_count == 1


class TestObjectExists:
    def test_returns_true_when_object_exists(self):
        client = MagicMock()
        client.head_object.return_value = {"ContentLength": 100}
        assert sc.object_exists(client, "my-bucket", "my-key") is True

    def test_returns_false_on_404(self):
        client = MagicMock()
        client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        assert sc.object_exists(client, "my-bucket", "my-key") is False

    def test_returns_false_on_no_such_key(self):
        client = MagicMock()
        client.head_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "HeadObject"
        )
        assert sc.object_exists(client, "my-bucket", "my-key") is False

    def test_reraises_non_404_client_error(self):
        client = MagicMock()
        client.head_object.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
        )
        with pytest.raises(ClientError) as exc_info:
            sc.object_exists(client, "my-bucket", "my-key")

        assert exc_info.value.response["Error"]["Code"] == "403"


class TestListObjects:
    def test_returns_keys_from_single_page(self):
        client = MagicMock()
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "prefix/a.json"}, {"Key": "prefix/b.json"}]}
        ]

        result = sc.list_objects(client, "bucket", "prefix/")

        client.get_paginator.assert_called_once_with("list_objects_v2")
        paginator.paginate.assert_called_once_with(Bucket="bucket", Prefix="prefix/")
        assert result == ["prefix/a.json", "prefix/b.json"]

    def test_returns_keys_across_multiple_pages(self):
        client = MagicMock()
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "p/1.json"}]},
            {"Contents": [{"Key": "p/2.json"}, {"Key": "p/3.json"}]},
        ]

        result = sc.list_objects(client, "bucket", "p/")

        assert result == ["p/1.json", "p/2.json", "p/3.json"]

    def test_handles_empty_page(self):
        client = MagicMock()
        paginator = MagicMock()
        client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{}]

        result = sc.list_objects(client, "bucket", "empty/")

        assert result == []
