"""News search tool for TD-One agents (PIT-safe by construction).

Wraps :class:`td_one.news.NewsStore` (local GDELT SQLite) and
:class:`td_one.news.perplexity.PerplexityClient` (Agent API) behind one
agent-facing interface. The temporal guard is enforced at THREE levels:

1. ``before`` is mandatory — there is no default, so a caller cannot
   accidentally omit the cutoff and see the whole corpus.
2. Future cutoffs are rejected — ``before`` must not be after today.
3. The backends themselves filter ``pub_date < before``.

Every call is logged (query, cutoff, backend, result count) so a PIT audit
can reconstruct exactly what the agent saw at each forecast origin.

Also ships :data:`NEWS_SYSTEM_PROMPT`, a system-prompt fragment that teaches
the LLM the news-use contract: state the cutoff, cite dates, say
"insufficient evidence" instead of hallucinating, never extrapolate the
future from past articles.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from td_one.news import Article, NewsStore
    from td_one.news.perplexity import PerplexityArticle, PerplexityClient

logger = logging.getLogger(__name__)

#: Hard cap on searches per tool instance (one instance = one agent run).
#: Prevents runaway Perplexity API costs from a looping agent.
MAX_SEARCHES_PER_RUN = 10

#: Hard cap on articles per call (prevents context-window flooding).
MAX_LIMIT = 20

#: Snippets are truncated to this many characters before reaching the LLM.
SNIPPET_CAP = 500

#: Titles are truncated to this many characters.
TITLE_CAP = 300


#: Article dict returned by NewsSearchTool.search.
ArticleDict = dict[str, str]


class NewsSearchTool:
    """PIT-safe news search for agents.

    Parameters
    ----------
    backend:
        "local" (NewsStore over the GDELT SQLite DB) or "perplexity"
        (Perplexity Agent API via the vault surrogate). Defaults to "local".
    db_path:
        Passed through to NewsStore when backend="local".
    max_searches:
        Per-run search budget. Defaults to :data:`MAX_SEARCHES_PER_RUN`.

    """

    def __init__(
        self,
        backend: str = "local",
        db_path: Path | None = None,
        max_searches: int = MAX_SEARCHES_PER_RUN,
    ) -> None:
        if backend not in ("local", "perplexity"):
            raise ValueError(f"backend must be 'local' or 'perplexity', got {backend!r}")
        if max_searches < 1:
            raise ValueError("max_searches must be >= 1")
        self.backend_name = backend
        self.max_searches = max_searches
        self._searches_used = 0
        if backend == "local":
            from td_one.news import NewsStore

            self._backend: NewsStore | PerplexityClient = NewsStore(db_path=db_path)
        else:
            from td_one.news.perplexity import PerplexityClient

            self._backend = PerplexityClient()

    @property
    def searches_used(self) -> int:
        """How many searches this instance has performed."""
        return self._searches_used

    @property
    def searches_remaining(self) -> int:
        """Remaining search budget for this run."""
        return self.max_searches - self._searches_used

    @staticmethod
    def _validate_cutoff(before: str | date) -> str:
        """Validate ``before`` and return it as YYYY-MM-DD.

        Raises
        ------
        ValueError
            On malformed dates or cutoffs in the future.
        TypeError
            On non-string/non-date input.

        """
        if isinstance(before, date):
            cutoff = before.isoformat()
        elif isinstance(before, str):
            cutoff = before
        else:
            raise TypeError(
                f"before must be a YYYY-MM-DD string or date, got {type(before).__name__}"
            )
        # Strict YYYY-MM-DD shape: date.fromisoformat is lenient in 3.11+
        # (accepts "20260601"), so enforce the dashed form explicitly.

        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cutoff):
            raise ValueError(f"before must be YYYY-MM-DD, got {cutoff!r}")
        try:
            parsed = date.fromisoformat(cutoff)
        except ValueError:
            raise ValueError(f"before must be YYYY-MM-DD, got {cutoff!r}") from None
        if parsed > datetime.now(UTC).date():
            raise ValueError(
                f"before={cutoff} is in the future — the news tool refuses "
                "future cutoffs to preserve PIT discipline"
            )
        return cutoff

    def search(self, query: str, before: str | date, limit: int = 5) -> list[ArticleDict]:
        """Search news published strictly before ``before``.

        Parameters
        ----------
        query:
            Keywords or quoted phrases. Must be non-empty.
        before:
            MANDATORY cutoff (YYYY-MM-DD or date). Articles with
            ``pub_date >= before`` are never returned. Future dates
            are rejected.
        limit:
            Max articles, 1..20. Defaults to 5.

        Returns
        -------
        List of {"date", "title", "url", "snippet"} dicts, newest first.
        Empty list when nothing matches — the caller must treat this as
        "insufficient evidence", not as license to hallucinate.

        Raises
        ------
        ValueError
            Empty query, bad cutoff, future cutoff, or bad limit.
        RuntimeError
            Search budget exhausted.

        """
        # --- input validation (fail loudly, never silently widen) ---
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        cutoff = self._validate_cutoff(before)
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise ValueError(f"limit must be an int, got {type(limit).__name__}")
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be 1..{MAX_LIMIT}, got {limit}")
        if self._searches_used >= self.max_searches:
            raise RuntimeError(
                f"news search budget exhausted ({self.max_searches} searches "
                "per run) — refine queries instead of issuing more"
            )

        self._searches_used += 1
        raw: list[Article] | list[PerplexityArticle] = self._backend.search(
            query.strip(), cutoff, limit
        )

        results: list[ArticleDict] = []
        for a in raw:
            # Belt-and-braces: re-check the PIT guard on every article,
            # even though both backends filter server-side.
            pub = str(getattr(a, "pub_date", ""))[:10]
            if not pub or pub >= cutoff:
                logger.warning(
                    "news_tool: dropped article with pub_date=%r >= cutoff=%s (backend=%s)",
                    pub,
                    cutoff,
                    self.backend_name,
                )
                continue
            results.append(
                {
                    "date": pub,
                    "title": str(getattr(a, "title", ""))[:TITLE_CAP],
                    "url": str(getattr(a, "url", "")),
                    "snippet": str(getattr(a, "snippet", ""))[:SNIPPET_CAP],
                }
            )

        # --- audit trail: every query is reconstructible ---
        logger.info(
            "news_tool: backend=%s query=%r before=%s limit=%d -> %d articles (search %d/%d)",
            self.backend_name,
            query.strip()[:120],
            cutoff,
            limit,
            len(results),
            self._searches_used,
            self.max_searches,
        )
        return results


NEWS_SYSTEM_PROMPT = """\
You have a news_search tool for researching economic context. It is \
PIT-safe: every call takes a MANDATORY `before` cutoff date (YYYY-MM-DD) \
and only returns articles published strictly before that date.

Rules for using news — follow them exactly:

1. ALWAYS pass the forecast origin as `before`. If you are forecasting \
from 2026-06-01, call news_search(query=..., before="2026-06-01"). \
Never use a later cutoff than your forecast origin.

2. STATE your cutoff in your reasoning: "Searching news before 2026-06-01 \
for Bank of Canada rate signals." The cutoff is part of your audit trail.

3. CITE article dates when making claims: "per Feb 2026 reporting, fixed \
mortgage rates were drifting lower" — never "rates were drifting lower" \
without the date. A claim without a date is unverifiable.

4. If search returns nothing useful, say INSUFFICIENT EVIDENCE and fall \
back to the statistical baseline. Do NOT hallucinate articles, invent \
headlines, or fill gaps with plausible-sounding narrative.

5. NEVER infer future events from past articles. A March article about \
"expected BoC cuts" does not tell you what the BoC did in April. Past \
reporting is context for regime classification, not a crystal ball.

6. You have a limited search budget per run. Make each query count: \
specific topics ("Bank of Canada rate decision March 2026"), not \
fishing expeditions.
"""
