import logging
import sys
from typing import Any

import structlog


def _add_logger_name(
    logger: Any, method_name: str, event_dict: dict
) -> dict:
    """Add logger name to the event dict (compatible with PrintLogger)."""
    event_dict["logger"] = getattr(logger, "name", str(logger))
    return event_dict


# Configure structlog once at module import time (global operation).
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        _add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    cache_logger_on_first_use=True,
)


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structured logger bound to the given name."""
    return structlog.get_logger(name)
