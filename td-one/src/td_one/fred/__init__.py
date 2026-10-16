"""FRED data access: deterministic mock series plus a file cache.

Live FRED is out of scope (raises NotImplementedError); development and
tests use the mock fixtures.
"""

from td_one.fred.cache import DEFAULT_CACHE_ROOT, FileCache
from td_one.fred.mock import (
    FIXTURE_END,
    FIXTURE_START,
    SUPPORTED_SERIES,
    get_series,
    get_series_live,
)

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "FIXTURE_END",
    "FIXTURE_START",
    "SUPPORTED_SERIES",
    "FileCache",
    "get_series",
    "get_series_live",
]
