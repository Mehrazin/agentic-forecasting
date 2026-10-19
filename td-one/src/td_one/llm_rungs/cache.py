"""Resume-safe JSON cache helpers for LLM grid outputs (exp55–exp57).

Cache schema (per the ladder plan):
``{"variant","dataset","model","levels":[0.05,...,0.95],"per_origin":[...]}``
with one record per origin:
``{"origin","as_of","actual","grid":[11 floats],"crps","ok"}``.
CRPS is scored on the 9 levels 0.10–0.90 (exp52/exp53 convention) while the
full 11-level grid is retained for the router.

Two APIs share the schema:

- The *worker* API (``new_cache`` / ``save_cache`` / ``append_origin_record``
  / ``done_origins`` / ``load_cache``): resume-safe, incremental, never
  raises on a missing/corrupt file — the inference worker writes one origin
  at a time and must survive restarts.
- The *evaluation* API (``save_grid_cache`` / ``load_grid_cache`` /
  ``grid_to_crps``): one-shot, validated, raises on any schema violation —
  evaluation must fail loudly rather than silently score a corrupt cache.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def new_cache(variant: str, dataset: str, model: str, levels: list[float]) -> dict[str, Any]:
    """Create an empty cache header."""
    return {
        "variant": variant,
        "dataset": dataset,
        "model": model,
        "levels": list(levels),
        "per_origin": [],
    }


def load_cache(path: Path) -> dict[str, Any] | None:
    """Load an existing cache, or None if missing/corrupt (resume-safe)."""
    if not path.exists():
        return None
    try:
        cache = json.loads(path.read_text())
        if not isinstance(cache, dict) or "per_origin" not in cache:
            return None
        return cache
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    """Write the cache atomically (tmp file + rename) after every origin."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=1))
    tmp.replace(path)


def append_origin_record(path: Path, cache: dict[str, Any], record: dict[str, Any]) -> None:
    """Append one origin record and persist the cache immediately."""
    cache.setdefault("per_origin", []).append(record)
    save_cache(path, cache)


def done_origins(cache: dict[str, Any]) -> set[int]:
    """Origin indices already recorded in the cache (for resume)."""
    return {int(r["origin"]) for r in cache.get("per_origin", []) if "origin" in r}


# ---------------------------------------------------------------------------
# Evaluation API: one-shot, validated grid caches
# ---------------------------------------------------------------------------

REQUIRED_CACHE_KEYS = ("variant", "dataset", "model", "levels", "per_origin")
REQUIRED_ORIGIN_KEYS = ("origin", "as_of", "actual", "grid", "crps", "ok")
_AS_OF_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def grid_to_crps(grid: Any, levels: Any, actual: float) -> float:
    """CRPS via 2 * mean pinball loss over quantile levels.

    Formula copied from ``experiments/exp53_vector_llm.py::crps_quantile``
    (not imported: ``experiments/`` is not a package dependency of ``src/``).
    Convention is to call this with our 9 levels (0.1..0.9); the function
    itself works for any level grid.
    """
    q = np.asarray(grid, dtype=float)
    lv = np.asarray(levels, dtype=float)
    if q.shape != lv.shape:
        raise ValueError(f"grid shape {q.shape} != levels shape {lv.shape}")
    pinball = np.where(
        float(actual) >= q,
        (float(actual) - q) * lv,
        (q - float(actual)) * (1.0 - lv),
    )
    return float(2.0 * np.mean(pinball))


def _checked_origin(entry: dict[str, Any], n_levels: int, i: int) -> dict[str, Any]:
    """Validate one per-origin record; return a JSON-safe normalized copy."""
    missing = [k for k in REQUIRED_ORIGIN_KEYS if k not in entry]
    if missing:
        raise ValueError(f"per_origin[{i}]: missing keys {missing}")
    as_of = entry["as_of"]
    if not isinstance(as_of, str) or not _AS_OF_RE.match(as_of):
        raise ValueError(f"per_origin[{i}]: 'as_of' must be 'YYYY-MM-DD', got {as_of!r}")
    grid = entry["grid"]
    if not isinstance(grid, (list, tuple)) or len(grid) != n_levels:
        raise ValueError(
            f"per_origin[{i}]: 'grid' must have {n_levels} entries "
            f"(one per level), got {len(grid) if isinstance(grid, (list, tuple)) else type(grid)}"
        )
    if not isinstance(entry["ok"], bool):
        raise ValueError(f"per_origin[{i}]: 'ok' must be bool, got {entry['ok']!r}")
    return {
        "origin": int(entry["origin"]),
        "as_of": as_of,
        "actual": float(entry["actual"]),
        "grid": [float(v) for v in grid],
        "crps": float(entry["crps"]),
        "ok": entry["ok"],
    }


def _checked_cache(cache: dict[str, Any]) -> dict[str, Any]:
    """Validate a cache dict; return a JSON-safe normalized copy."""
    if not isinstance(cache, dict):
        raise ValueError(f"cache must be a dict, got {type(cache)}")
    missing = [k for k in REQUIRED_CACHE_KEYS if k not in cache]
    if missing:
        raise ValueError(f"cache missing keys {missing}")
    for k in ("variant", "dataset", "model"):
        if not isinstance(cache[k], str):
            raise ValueError(f"cache[{k!r}] must be str, got {cache[k]!r}")
    levels = cache["levels"]
    levels_ok = (
        isinstance(levels, (list, tuple))
        and bool(levels)
        and all(isinstance(v, (int, float)) and 0.0 < float(v) < 1.0 for v in levels)
        and all(b > a for a, b in zip(levels, levels[1:], strict=False))
    )
    if not levels_ok:
        raise ValueError(
            f"cache['levels'] must be strictly increasing levels in (0,1), got {levels!r}"
        )
    levels_f = [float(v) for v in levels]
    per_origin = cache["per_origin"]
    if not isinstance(per_origin, (list, tuple)):
        raise ValueError(f"cache['per_origin'] must be a list, got {type(per_origin)}")
    return {
        "variant": cache["variant"],
        "dataset": cache["dataset"],
        "model": cache["model"],
        "levels": levels_f,
        "per_origin": [
            _checked_origin(dict(e), len(levels_f), i) for i, e in enumerate(per_origin)
        ],
    }


def save_grid_cache(
    path: str | Path,
    variant: str,
    dataset: str,
    model: str,
    levels: list[float],
    per_origin: list[dict[str, Any]],
) -> None:
    """Write a complete, validated grid cache in one shot.

    Unlike the worker API above (incremental, resume-safe), this validates
    the full schema and raises :class:`ValueError` on any violation — a
    cache that fails validation must never reach evaluation silently.
    """
    cache = _checked_cache(
        {
            "variant": variant,
            "dataset": dataset,
            "model": model,
            "levels": levels,
            "per_origin": per_origin,
        }
    )
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=1))
    tmp.replace(p)


def load_grid_cache(path: str | Path) -> dict[str, Any]:
    """Load a grid cache, raising on missing file or schema violation.

    (Contrast with :func:`load_cache`, which returns ``None`` for the
    resume-safe worker path.)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"grid cache not found: {p}")
    try:
        cache = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        raise ValueError(f"grid cache {p} is not valid JSON: {e}") from e
    try:
        return _checked_cache(cache)
    except ValueError as e:
        raise ValueError(f"grid cache {p} failed schema validation: {e}") from e
