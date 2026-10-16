"""Evidence Tool: cutoff-safe keyword retrieval over a frozen evidence corpus.

The corpus is ``td-one/experiments/evidence_corpus/corpus.json``: a JSON list
of entries, each with keys ``id``, ``date`` (ISO ``YYYY-MM-DD``), ``title``,
``text``, and ``source``. The corpus is frozen at experiment time; this tool
only reads it and enforces the point-in-time cutoff.

Ranking is deliberately naive (token overlap on title+text, title weighted
2x) so retrieval behaviour is transparent and reproducible — it is a lookup
aid, not a semantic search engine.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from td_one._paths import _repo_root

#: Corpus written by the evidence-corpus build step (frozen at experiment time).
CORPUS_PATH = _repo_root() / "experiments" / "evidence_corpus" / "corpus.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens."""
    return set(_TOKEN_RE.findall(text.lower()))


def _load_corpus() -> list[dict[str, Any]]:
    """Load the frozen corpus; [] when it does not exist yet."""
    if not CORPUS_PATH.exists():
        return []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def _score(query_tokens: set[str], entry: dict[str, Any]) -> int:
    """Token-overlap score: title matches count double."""
    title = _tokens(str(entry.get("title", "")))
    text = _tokens(str(entry.get("text", "")))
    return 2 * len(query_tokens & title) + len(query_tokens & text)


def evidence_tool(query: str, run_date: date, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve evidence entries visible at point-in-time, ranked by keyword.

    Entries dated after ``run_date`` are filtered out (TemporalGuard).

    Args:
        query: Free-text query; matched by token overlap on title+text.
        run_date: Point-in-time cutoff (inclusive).
        top_k: Maximum entries to return.

    Returns:
        Ranked list of entry dicts (keys: id, date, title, text, source).
        Empty when the corpus is missing or nothing matches.

    """
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    cutoff = run_date.isoformat()
    ranked: list[tuple[int, dict[str, Any]]] = []
    for entry in _load_corpus():
        entry_date = str(entry.get("date", ""))
        if not entry_date or entry_date > cutoff:
            continue  # not yet published at run_date
        score = _score(query_tokens, entry)
        if score > 0:
            ranked.append((score, entry))
    ranked.sort(key=lambda pair: (-pair[0], str(pair[1].get("date", ""))))
    return [entry for _, entry in ranked[: max(top_k, 0)]]
