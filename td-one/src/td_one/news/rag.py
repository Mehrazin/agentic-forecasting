"""News RAG tool for TD-One agents (offline-first).

Pipeline: BM25 retrieval (FTS5, free, reproducible) → Groq LLM rerank
(one call, JSON scores) → abstention when nothing is relevant enough.

The KB is built offline (scripts/fetch_news_*.py). The agent hot path
never touches a live API except Groq for reranking — and even that
degrades gracefully to BM25 order when Groq is unavailable.

PIT discipline: ``before`` is mandatory, future cutoffs are rejected,
and every article is re-checked against the cutoff client-side.
See docs/RAG_DESIGN.md for the full design.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from td_one.news import NewsStore

if TYPE_CHECKING:
    from td_one.agent.groq_client import GroqChatClient

logger = logging.getLogger(__name__)

#: BM25 candidate pool size (stage 1). Rerank sees at most this many.
BM25_CANDIDATES = 20

#: Default abstention threshold on reranked relevance (0..1).
DEFAULT_ABSTAIN_THRESHOLD = 0.4

#: Hard cap on articles per call (prevents context-window flooding).
MAX_LIMIT = 20

#: Snippets/titles truncated before reaching the LLM.
SNIPPET_CAP = 500
TITLE_CAP = 300

_RERANK_SYSTEM = """\
You are a relevance judge for mortgage-market news retrieval.
Score each article 0..1 for how directly it informs the analyst's question.
1.0 = directly answers it. 0.5 = useful background. 0.0 = unrelated.
Score junk, ads, and duplicates 0 — never inflate relevance.
Reply ONLY with a JSON array of {"idx": <int>, "relevance": <float>} objects, \
one per article, in the same order as given.\
"""


@dataclass(frozen=True)
class RankedArticle:
    """An article with its reranked relevance score."""

    pub_date: str
    title: str
    url: str
    snippet: str
    relevance: float  # 0..1 from the reranker (BM25 order if rerank failed)


@dataclass
class RAGResult:
    """Output of RAGTool.search."""

    articles: list[RankedArticle] = field(default_factory=list)
    abstained: bool = False
    backend: str = "bm25"  # "bm25+rerank" when the LLM rerank succeeded


def _validate_cutoff(before: str | date) -> str:
    """Validate ``before`` as YYYY-MM-DD; reject future dates.

    Raises
    ------
    ValueError
        Malformed date or cutoff after today.
    TypeError
        Non-string/non-date input.

    """
    if isinstance(before, date):
        cutoff = before.isoformat()
    elif isinstance(before, str):
        cutoff = before
    else:
        raise TypeError(f"before must be a YYYY-MM-DD string or date, got {type(before).__name__}")
    # date.fromisoformat is lenient in 3.11+ (accepts "20260601"),
    # so enforce the dashed form explicitly.
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff):
        raise ValueError(f"before must be YYYY-MM-DD, got {cutoff!r}")
    try:
        parsed = date.fromisoformat(cutoff)
    except ValueError:
        raise ValueError(f"before must be YYYY-MM-DD, got {cutoff!r}") from None
    if parsed > datetime.now(UTC).date():
        raise ValueError(f"before={cutoff} is in the future — refusing to preserve PIT discipline")
    return cutoff


class RAGTool:
    """Offline-first news RAG for agents.

    Parameters
    ----------
    db_path:
        Passed to NewsStore (the frozen KB).
    groq:
        GroqChatClient for reranking. If None, runs BM25-only
        (no LLM calls; never abstains on relevance).
    abstain_threshold:
        Minimum top relevance to return results. Tune per
        docs/RAG_DESIGN.md; default 0.4.

    """

    def __init__(
        self,
        db_path: Path | None = None,
        groq: GroqChatClient | None = None,
        abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    ) -> None:
        if not 0.0 <= abstain_threshold <= 1.0:
            raise ValueError(f"abstain_threshold must be 0..1, got {abstain_threshold}")
        self._store = NewsStore(db_path=db_path)
        self._groq = groq
        self.abstain_threshold = abstain_threshold

    def search(
        self,
        query: str,
        before: str | date,
        limit: int = 5,
    ) -> RAGResult:
        """Retrieve and rerank news strictly before ``before``.

        Stage 1: BM25 over the frozen KB (top-20 candidates).
        Stage 2: one Groq rerank call (skipped when groq is None).
        Stage 3: abstain when the best relevance < threshold.

        Raises
        ------
        ValueError
            Empty query, malformed/future cutoff, or bad limit.

        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        cutoff = _validate_cutoff(before)
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError(f"limit must be an int, got {type(limit).__name__}")
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be 1..{MAX_LIMIT}, got {limit}")

        # --- stage 1: BM25 candidates, cutoff enforced in SQL ---
        raw = self._store.search(query.strip(), cutoff, BM25_CANDIDATES)
        candidates = [
            RankedArticle(
                pub_date=a.pub_date,
                title=a.title[:TITLE_CAP],
                url=a.url,
                snippet=a.snippet[:SNIPPET_CAP],
                relevance=0.0,  # filled by the reranker
            )
            for a in raw
            if a.pub_date < cutoff  # belt-and-braces PIT re-check
        ]
        if not candidates:
            return RAGResult(articles=[], abstained=True, backend="bm25")

        # --- stage 2: LLM rerank (one call) or BM25 fallback ---
        if self._groq is None:
            return RAGResult(articles=candidates[:limit], abstained=False, backend="bm25")
        ranked, rerank_ok = self._rerank(query.strip(), candidates)
        backend = "bm25+rerank" if rerank_ok else "bm25"

        # --- stage 3: abstention (only when the reranker actually scored) ---
        if rerank_ok:
            top = ranked[0].relevance if ranked else 0.0
            if top < self.abstain_threshold:
                logger.info(
                    "rag: abstaining (top relevance %.2f < threshold %.2f, query=%r)",
                    top,
                    self.abstain_threshold,
                    query.strip()[:80],
                )
                return RAGResult(articles=[], abstained=True, backend=backend)
        return RAGResult(articles=ranked[:limit], abstained=False, backend=backend)

    def _rerank(
        self, query: str, candidates: list[RankedArticle]
    ) -> tuple[list[RankedArticle], bool]:
        """Score candidates with one Groq call.

        Returns (ranked, ok): ok=False means BM25 order was kept
        (Groq failed or JSON unparseable) — degraded but functional.
        """
        assert self._groq is not None
        prompt = self._build_rerank_prompt(query, candidates)
        try:
            text = self._groq.complete(prompt, system=_RERANK_SYSTEM, json_mode=True)
        except Exception as exc:  # noqa: BLE001 — degraded mode must never raise
            logger.warning("rag: rerank Groq call failed (%s); using BM25 order", exc)
            return candidates, False
        scores = self._parse_rerank_json(text, len(candidates))
        if scores is None:
            logger.warning("rag: rerank JSON unparseable; using BM25 order")
            return candidates, False
        scored = [
            RankedArticle(
                pub_date=c.pub_date,
                title=c.title,
                url=c.url,
                snippet=c.snippet,
                relevance=min(1.0, max(0.0, s)),
            )
            for c, s in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda a: a.relevance, reverse=True)
        return scored, True

    @staticmethod
    def _build_rerank_prompt(query: str, candidates: list[RankedArticle]) -> str:
        """Build the relevance-judge prompt for the reranker."""
        lines = [
            f"Analyst question: {query}",
            "",
            "Candidate articles (score each 0..1):",
        ]
        for i, c in enumerate(candidates):
            lines.append(f"[{i}] ({c.pub_date}) {c.title}")
            if c.snippet:
                lines.append(f"     {c.snippet[:200]}")
        return "\n".join(lines)

    @staticmethod
    def _parse_rerank_json(text: str, n: int) -> list[float] | None:
        """Parse [{"idx": i, "relevance": r}] into a score list.

        Returns None when the payload is unusable (caller falls back
        to BM25 order). Scores are clamped to 0..1 by the caller.
        """
        t = text.strip()
        if t.startswith("```"):
            parts = t.split("\n", 1)
            t = parts[1].rsplit("```", 1)[0] if len(parts) > 1 else ""
        try:
            items = json.loads(t)
        except json.JSONDecodeError:
            return None
        if not isinstance(items, list) or len(items) != n:
            return None
        scores: list[float] = []
        for it in items:
            if not isinstance(it, dict):
                return None
            try:
                idx = int(it["idx"])
                rel = float(it["relevance"])
            except (KeyError, TypeError, ValueError):
                return None
            if idx < 0 or idx >= n:
                return None
            scores.append(rel)
        return scores
