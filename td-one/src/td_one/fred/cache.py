"""File-backed parquet cache for FRED (and other macro) series.

Keys are sanitized to ``[A-Za-z0-9_.-]`` so a cache key can never escape the
cache root (no path traversal).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

#: Default cache root (gitignored).
DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[3] / "data" / "fred_cache"

_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class FileCache:
    """Parquet file cache keyed by sanitized string keys."""

    def __init__(self, root: Path | str = DEFAULT_CACHE_ROOT) -> None:
        """Initialize the cache.

        Args:
            root: Directory holding ``<key>.parquet`` files (created lazily).

        """
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        """Resolve key to a file path; reject traversal attempts."""
        if not _KEY_RE.match(key):
            raise ValueError(
                f"invalid cache key {key!r}: only letters, digits, '_', '.', '-' allowed"
            )
        return self.root / f"{key}.parquet"

    def get(self, key: str) -> pd.DataFrame | None:
        """Return the cached DataFrame, or None on a cache miss.

        Args:
            key: Cache key.

        """
        path = self._path(key)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def put(self, key: str, df: pd.DataFrame) -> None:
        """Write a DataFrame to the cache.

        Args:
            key: Cache key.
            df: DataFrame to persist.

        """
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
