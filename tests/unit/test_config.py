"""Unit tests for src/utils/config.py."""

import pytest

from src.utils.config import (
    ApiConfig,
    StorageConfig,
    api_config,
    reset_config,
    storage_config,
)


# ---------------------------------------------------------------------------
# _require
# ---------------------------------------------------------------------------

class TestRequire:
    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "hello")
        from src.utils.config import _require
        assert _require("TEST_VAR") == "hello"

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        from src.utils.config import _require
        with pytest.raises(EnvironmentError, match="MISSING_VAR"):
            _require("MISSING_VAR")

    def test_raises_when_empty_string(self, monkeypatch):
        monkeypatch.setenv("EMPTY_VAR", "")
        from src.utils.config import _require
        with pytest.raises(EnvironmentError, match="EMPTY_VAR"):
            _require("EMPTY_VAR")


# ---------------------------------------------------------------------------
# _optional
# ---------------------------------------------------------------------------

class TestOptional:
    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("OPT_VAR", "custom")
        from src.utils.config import _optional
        assert _optional("OPT_VAR", "default") == "custom"

    def test_returns_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("OPT_VAR", raising=False)
        from src.utils.config import _optional
        assert _optional("OPT_VAR", "default") == "default"


# ---------------------------------------------------------------------------
# StorageConfig
# ---------------------------------------------------------------------------

class TestStorageConfig:
    def test_loads_all_values_from_env(self, monkeypatch):
        monkeypatch.setenv("MINIO_ACCESS_KEY", "mykey")
        monkeypatch.setenv("MINIO_SECRET_KEY", "mysecret")
        monkeypatch.setenv("MINIO_ENDPOINT", "http://custom:9000")
        monkeypatch.setenv("BRONZE_BUCKET", "my-bronze")
        monkeypatch.setenv("SILVER_BUCKET", "my-silver")
        monkeypatch.setenv("GOLD_BUCKET", "my-gold")
        reset_config()

        cfg = storage_config()

        assert cfg.access_key == "mykey"
        assert cfg.secret_key == "mysecret"
        assert cfg.endpoint == "http://custom:9000"
        assert cfg.bronze_bucket == "my-bronze"
        assert cfg.silver_bucket == "my-silver"
        assert cfg.gold_bucket == "my-gold"

    def test_default_endpoint(self, monkeypatch):
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
        reset_config()

        cfg = storage_config()

        assert cfg.endpoint == "http://minio:9000"

    def test_default_bucket_names(self, monkeypatch):
        monkeypatch.delenv("BRONZE_BUCKET", raising=False)
        monkeypatch.delenv("SILVER_BUCKET", raising=False)
        monkeypatch.delenv("GOLD_BUCKET", raising=False)
        reset_config()

        cfg = storage_config()

        assert cfg.bronze_bucket == "brewery-bronze"
        assert cfg.silver_bucket == "brewery-silver"
        assert cfg.gold_bucket == "brewery-gold"

    def test_raises_when_access_key_missing(self, monkeypatch):
        monkeypatch.delenv("MINIO_ACCESS_KEY", raising=False)
        reset_config()

        with pytest.raises(EnvironmentError, match="MINIO_ACCESS_KEY"):
            storage_config()

    def test_raises_when_secret_key_missing(self, monkeypatch):
        monkeypatch.delenv("MINIO_SECRET_KEY", raising=False)
        reset_config()

        with pytest.raises(EnvironmentError, match="MINIO_SECRET_KEY"):
            storage_config()

    def test_is_immutable(self):
        cfg = storage_config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.access_key = "hacked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ApiConfig
# ---------------------------------------------------------------------------

class TestApiConfig:
    def test_loads_all_values_from_env(self, monkeypatch):
        monkeypatch.setenv("API_BASE_URL", "https://custom.api/v1/breweries")
        monkeypatch.setenv("API_PAGE_SIZE", "100")
        monkeypatch.setenv("API_MAX_RETRIES", "5")
        reset_config()

        cfg = api_config()

        assert cfg.base_url == "https://custom.api/v1/breweries"
        assert cfg.page_size == 100
        assert cfg.max_retries == 5

    def test_default_values(self, monkeypatch):
        monkeypatch.delenv("API_BASE_URL", raising=False)
        monkeypatch.delenv("API_PAGE_SIZE", raising=False)
        monkeypatch.delenv("API_MAX_RETRIES", raising=False)
        reset_config()

        cfg = api_config()

        assert cfg.base_url == "https://api.openbrewerydb.org/v1/breweries"
        assert cfg.page_size == 200
        assert cfg.max_retries == 3

    def test_page_size_is_int(self, monkeypatch):
        monkeypatch.setenv("API_PAGE_SIZE", "50")
        reset_config()

        cfg = api_config()

        assert isinstance(cfg.page_size, int)

    def test_invalid_page_size_raises(self, monkeypatch):
        monkeypatch.setenv("API_PAGE_SIZE", "not-a-number")
        reset_config()

        with pytest.raises(ValueError):
            api_config()


# ---------------------------------------------------------------------------
# Singleton / reset_config
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_returns_same_instance(self):
        cfg1 = storage_config()
        cfg2 = storage_config()
        assert cfg1 is cfg2

    def test_reset_allows_new_config(self, monkeypatch):
        cfg_before = storage_config()
        monkeypatch.setenv("BRONZE_BUCKET", "new-bronze")
        reset_config()

        cfg_after = storage_config()

        assert cfg_after is not cfg_before
        assert cfg_after.bronze_bucket == "new-bronze"
