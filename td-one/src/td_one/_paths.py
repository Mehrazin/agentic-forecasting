"""Shared filesystem helpers for the ``td_one`` package.

Centralizes the "where is the project root?" arithmetic so modules don't
each repeat ``Path(__file__).resolve().parents[N]`` with a hand-counted
``N`` — the exact pattern behind the earlier silent-zero-evidence bug,
where one module used ``parents[2]`` instead of ``parents[3]`` and read
from the wrong directory without any error.
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    """Return the td-one project root (the directory holding ``pyproject.toml``).

    This module lives at ``src/td_one/_paths.py``, so the root is two
    levels up. The path is resolved (no symlinks) so callers can compare
    paths reliably.
    """
    return Path(__file__).resolve().parents[2]
