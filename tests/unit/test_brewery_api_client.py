"""Unit tests for brewery_api_client.py."""

import pytest
import requests

from src.ingestion.brewery_api_client import (
    calculate_total_pages,
    fetch_brewery_meta,
    fetch_brewery_page,
)

META_URL = "https://api.openbrewerydb.org/v1/breweries/meta"
PAGE_URL = "https://api.openbrewerydb.org/v1/breweries"


# ---------------------------------------------------------------------------
# fetch_brewery_meta
# ---------------------------------------------------------------------------

class TestFetchBreweryMeta:
    def test_returns_total_count(self, requests_mock):
        requests_mock.get(META_URL, json={"total": 9383, "page": 1, "per_page": 50})

        result = fetch_brewery_meta()

        assert result["total"] == 9383

    def test_raises_on_http_error(self, requests_mock):
        requests_mock.get(META_URL, status_code=500)

        with pytest.raises(requests.HTTPError):
            fetch_brewery_meta()

    def test_raises_on_404(self, requests_mock):
        requests_mock.get(META_URL, status_code=404)

        with pytest.raises(requests.HTTPError):
            fetch_brewery_meta()


# ---------------------------------------------------------------------------
# fetch_brewery_page
# ---------------------------------------------------------------------------

class TestFetchBreweryPage:
    def test_returns_list_of_records(self, requests_mock, sample_breweries):
        requests_mock.get(PAGE_URL, json=sample_breweries[:2])

        result = fetch_brewery_page(page=1, per_page=2)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == sample_breweries[0]["id"]

    def test_passes_pagination_params(self, requests_mock):
        requests_mock.get(PAGE_URL, json=[])

        fetch_brewery_page(page=3, per_page=10)

        assert requests_mock.last_request.qs == {"page": ["3"], "per_page": ["10"]}

    def test_raises_on_4xx_without_retry(self, requests_mock):
        """4xx errors should propagate immediately without retrying."""
        requests_mock.get(PAGE_URL, status_code=403)

        with pytest.raises(requests.HTTPError) as exc_info:
            fetch_brewery_page(page=1)

        assert exc_info.value.response.status_code == 403

    def test_retries_on_500_then_succeeds(self, requests_mock, sample_breweries):
        """First call returns 500, second returns 200 — should succeed after retry."""
        requests_mock.get(
            PAGE_URL,
            [
                {"status_code": 500},
                {"json": sample_breweries[:1], "status_code": 200},
            ],
        )

        # With max_retries=2 (set in conftest env), this should succeed
        result = fetch_brewery_page(page=1)

        assert len(result) == 1

    def test_empty_page_returns_empty_list(self, requests_mock):
        requests_mock.get(PAGE_URL, json=[])

        result = fetch_brewery_page(page=999)

        assert result == []


# ---------------------------------------------------------------------------
# calculate_total_pages
# ---------------------------------------------------------------------------

class TestCalculateTotalPages:
    @pytest.mark.parametrize(
        "total, per_page, expected",
        [
            (9383, 200, 47),   # real-world case: ceil(9383/200) = 47
            (200, 200, 1),     # exactly one page
            (201, 200, 2),     # one record on second page
            (1, 200, 1),       # single record
            (0, 200, 0),       # empty dataset
        ],
    )
    def test_calculates_correctly(self, total, per_page, expected):
        assert calculate_total_pages(total, per_page) == expected

    def test_uses_config_page_size_by_default(self, monkeypatch):
        """When per_page is not passed, uses API_PAGE_SIZE from config."""
        monkeypatch.setenv("API_PAGE_SIZE", "100")

        from src.utils.config import reset_config
        reset_config()

        result = calculate_total_pages(total=350)

        assert result == 4  # ceil(350/100)
