"""
Centralized pipeline configuration.

All environment variables are loaded and validated here.
Other modules import from this file — never call os.environ directly in
pipeline code. This ensures the app fails fast with a clear message when a
required setting is missing, rather than silently using wrong credentials.
"""

import os
from dataclasses import dataclass


def _require(name: str) -> str:
    """Return env var value or raise a clear error if not set."""
    value = os.environ.get(name)
    if not value:
        raise OSError(
            f"Required environment variable '{name}' is not set. "
            "Copy .env.example to .env and fill in all values."
        )
    return value


def _optional(name: str, default: str) -> str:
    """Return env var value or a documented default."""
    return os.environ.get(name, default)


@dataclass(frozen=True)
class StorageConfig:
    endpoint: str
    access_key: str
    secret_key: str
    bronze_bucket: str
    silver_bucket: str
    gold_bucket: str

    @classmethod
    def from_env(cls) -> "StorageConfig":
        return cls(
            endpoint=_optional("MINIO_ENDPOINT", "http://minio:9000"),
            access_key=_require("MINIO_ACCESS_KEY"),
            secret_key=_require("MINIO_SECRET_KEY"),
            bronze_bucket=_optional("BRONZE_BUCKET", "brewery-bronze"),
            silver_bucket=_optional("SILVER_BUCKET", "brewery-silver"),
            gold_bucket=_optional("GOLD_BUCKET", "brewery-gold"),
        )


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    page_size: int
    max_retries: int

    @classmethod
    def from_env(cls) -> "ApiConfig":
        return cls(
            base_url=_optional(
                "API_BASE_URL", "https://api.openbrewerydb.org/v1/breweries"
            ),
            page_size=int(_optional("API_PAGE_SIZE", "200")),
            max_retries=int(_optional("API_MAX_RETRIES", "3")),
        )


# Module-level singletons — instantiated lazily so tests can monkeypatch
# env vars before the first call.
_storage: StorageConfig | None = None
_api: ApiConfig | None = None


def storage_config() -> StorageConfig:
    global _storage
    if _storage is None:
        _storage = StorageConfig.from_env()
    return _storage


def api_config() -> ApiConfig:
    global _api
    if _api is None:
        _api = ApiConfig.from_env()
    return _api


def reset_config() -> None:
    """Reset cached singletons. Call this in tests after monkeypatching env vars."""
    global _storage, _api
    _storage = None
    _api = None
