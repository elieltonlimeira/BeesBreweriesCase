"""Unit tests for bronze_writer.py."""

import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from src.bronze.bronze_writer import build_bronze_key, write_page


class TestBuildBronzeKey:
    def test_correct_format(self):
        key = build_bronze_key("2024-03-24", 1)
        assert key == "raw/dt=2024-03-24/page=001.json"

    def test_zero_padding_three_digits(self):
        assert build_bronze_key("2024-03-24", 5) == "raw/dt=2024-03-24/page=005.json"
        assert build_bronze_key("2024-03-24", 47) == "raw/dt=2024-03-24/page=047.json"
        assert build_bronze_key("2024-03-24", 100) == "raw/dt=2024-03-24/page=100.json"

    def test_lexicographic_order_preserved(self):
        """Zero-padded keys sort correctly in S3 listings."""
        keys = [build_bronze_key("2024-03-24", p) for p in [1, 2, 10, 47]]
        assert keys == sorted(keys)


class TestWritePage:
    def test_uploads_json_to_correct_key(self, mock_s3_client, sample_breweries):
        mock_s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        mock_s3_client.put_object.return_value = {}

        result = write_page(
            records=sample_breweries[:2],
            page=1,
            execution_date="2024-03-24",
            s3_client=mock_s3_client,
        )

        assert result["key"] == "raw/dt=2024-03-24/page=001.json"
        assert result["records_written"] == 2
        assert result["bytes_written"] > 0
        assert result["skipped"] is False

        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "test-bronze"
        assert call_kwargs["Key"] == "raw/dt=2024-03-24/page=001.json"
        assert call_kwargs["ContentType"] == "application/json"

    def test_uploaded_content_is_valid_json(self, mock_s3_client, sample_breweries):
        mock_s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        mock_s3_client.put_object.return_value = {}

        write_page(
            records=sample_breweries[:1],
            page=1,
            execution_date="2024-03-24",
            s3_client=mock_s3_client,
        )

        body = mock_s3_client.put_object.call_args.kwargs["Body"]
        parsed = json.loads(body.decode("utf-8"))
        assert isinstance(parsed, list)
        assert parsed[0]["id"] == sample_breweries[0]["id"]

    def test_skips_upload_when_overwrite_false_and_exists(
        self, mock_s3_client, sample_breweries
    ):
        # Clear the 404 side_effect set in conftest so head_object succeeds
        mock_s3_client.head_object.side_effect = None
        mock_s3_client.head_object.return_value = {"ContentLength": 100}

        result = write_page(
            records=sample_breweries[:1],
            page=1,
            execution_date="2024-03-24",
            s3_client=mock_s3_client,
            overwrite=False,
        )

        assert result["skipped"] is True
        assert result["records_written"] == 0
        assert result["bytes_written"] == 0
        mock_s3_client.put_object.assert_not_called()

    def test_overwrites_when_overwrite_true(self, mock_s3_client, sample_breweries):
        """Even if file exists, overwrite=True should always upload."""
        mock_s3_client.head_object.side_effect = None
        mock_s3_client.head_object.return_value = {"ContentLength": 100}
        mock_s3_client.put_object.return_value = {}

        result = write_page(
            records=sample_breweries[:1],
            page=1,
            execution_date="2024-03-24",
            s3_client=mock_s3_client,
            overwrite=True,
        )

        assert result["skipped"] is False
        mock_s3_client.put_object.assert_called_once()

    def test_empty_records_list(self, mock_s3_client):
        mock_s3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        mock_s3_client.put_object.return_value = {}

        result = write_page(
            records=[],
            page=1,
            execution_date="2024-03-24",
            s3_client=mock_s3_client,
        )

        assert result["records_written"] == 0
        mock_s3_client.put_object.assert_called_once()
