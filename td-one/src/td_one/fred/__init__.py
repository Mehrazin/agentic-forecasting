"""FRED data access: deterministic mock series, live API, plus a file cache.

Tests and development default to the mock fixtures; live FRED goes
through the ``custom.fred`` vault connector (key never in process).
"""

from td_one.fred.cache import DEFAULT_CACHE_ROOT, FileCache
from td_one.fred.live import FredUnavailableError, get_series_live
from td_one.fred.mock import (
    FIXTURE_END,
    FIXTURE_START,
    SUPPORTED_SERIES,
    get_series,
)

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "FredUnavailableError",
    "FIXTURE_END",
    "FIXTURE_START",
    "SUPPORTED_SERIES",
    "FileCache",
    "get_series",
    "get_series_live",
]
