"""Tests for the PIT-safe news search tool (td_one.agent.news_tool)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import pytest

from td_one.agent.news_tool import (
    MAX_LIMIT,
    MAX_SEARCHES_PER_RUN,
    NEWS_SYSTEM_PROMPT,
    NewsSearchTool,
)


@dataclass
class _FakeArticle:
    pub_date: str
    title: str
    url: str
    snippet: str = ""
    source: str = "test"


class _FakeBackend:
    """Controllable backend: returns whatever articles we plant."""

    def __init__(self, articles: list[_FakeArticle]) -> None:
        self.articles = articles
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query: str, before: str | date, limit: int) -> list[_FakeArticle]:
        cutoff = before.isoformat() if isinstance(before, date) else before
        self.calls.append((query, cutoff, limit))
        return [a for a in self.articles if a.pub_date < cutoff][:limit]


def _tool(articles: list[_FakeArticle], **kw) -> NewsSearchTool:
    tool = NewsSearchTool.__new__(NewsSearchTool)
    tool.backend_name = "fake"
    tool.max_searches = kw.get("max_searches", MAX_SEARCHES_PER_RUN)
    tool._searches_used = 0
    tool._backend = _FakeBackend(articles)
    return tool


# --- mandatory cutoff ---


def test_before_is_mandatory_no_default() -> None:
    """`before` must have no default — omitting it is a TypeError, not a leak."""
    sig = inspect.signature(NewsSearchTool.search)
    param = sig.parameters["before"]
    assert param.default is inspect.Parameter.empty


def test_missing_before_raises_typeerror() -> None:
    tool = _tool([])
    with pytest.raises(TypeError):
        tool.search("mortgage rates")  # type: ignore[call-arg]


# --- PIT guard ---


def test_no_future_articles_returned() -> None:
    arts = [
        _FakeArticle("2026-02-15", "Old news", "http://x/1"),
        _FakeArticle("2026-06-15", "Future news", "http://x/2"),
    ]
    tool = _tool(arts)
    out = tool.search("rates", before="2026-06-01")
    assert [r["date"] for r in out] == ["2026-02-15"]


def test_tool_drops_backend_leak() -> None:
    """Even if a backend misbehaves, the tool re-checks every article."""

    class Leaky(_FakeBackend):
        def search(self, query, before, limit):  # type: ignore[no-untyped-def]
            return self.articles  # ignores cutoff entirely

    tool = _tool([])
    tool._backend = Leaky([_FakeArticle("2026-09-01", "Leaked future", "http://x/9")])
    out = tool.search("rates", before="2026-06-01")
    assert out == []


def test_cutoff_boundary_exclusive() -> None:
    arts = [_FakeArticle("2026-06-01", "Same-day", "http://x/1")]
    tool = _tool(arts)
    assert tool.search("rates", before="2026-06-01") == []


def test_accepts_date_object() -> None:
    arts = [_FakeArticle("2026-02-15", "Old", "http://x/1")]
    tool = _tool(arts)
    out = tool.search("rates", before=date(2026, 6, 1))
    assert len(out) == 1


# --- future-date rejection ---


def test_future_cutoff_rejected() -> None:
    tool = _tool([])
    future = (datetime.now(UTC).date() + timedelta(days=30)).isoformat()
    with pytest.raises(ValueError, match="future"):
        tool.search("rates", before=future)


def test_today_cutoff_allowed() -> None:
    arts = [_FakeArticle("2026-01-15", "Old", "http://x/1")]
    tool = _tool(arts)
    out = tool.search("rates", before=datetime.now(UTC).date().isoformat())
    assert len(out) == 1


# --- input validation ---


def test_empty_query_rejected() -> None:
    tool = _tool([])
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="non-empty"):
            tool.search(bad, before="2026-06-01")


def test_non_string_query_rejected() -> None:
    tool = _tool([])
    with pytest.raises(ValueError, match="non-empty"):
        tool.search(None, before="2026-06-01")  # type: ignore[arg-type]


def test_malformed_date_rejected() -> None:
    tool = _tool([])
    for bad in ("2026-13-01", "06/01/2026", "yesterday", "20260601"):
        with pytest.raises(ValueError):
            tool.search("rates", before=bad)


def test_bad_limit_rejected() -> None:
    tool = _tool([])
    for bad in (0, -1, MAX_LIMIT + 1, "5"):  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="[Ll]imit"):
            tool.search("rates", before="2026-06-01", limit=bad)


# --- rate limiting ---


def test_rate_limit_enforced() -> None:
    tool = _tool([], max_searches=3)
    for _ in range(3):
        tool.search("rates", before="2026-06-01")
    assert tool.searches_used == 3
    assert tool.searches_remaining == 0
    with pytest.raises(RuntimeError, match="budget exhausted"):
        tool.search("rates", before="2026-06-01")


def test_default_budget_is_ten() -> None:
    assert MAX_SEARCHES_PER_RUN == 10
    tool = _tool([])
    assert tool.searches_remaining == 10


# --- result shape & caps ---


def test_result_dict_shape() -> None:
    arts = [
        _FakeArticle("2026-02-15", "T" * 500, "http://x/1", "S" * 2000),
    ]
    tool = _tool(arts)
    out = tool.search("rates", before="2026-06-01")
    assert len(out) == 1
    r = out[0]
    assert set(r) == {"date", "title", "url", "snippet"}
    assert len(r["title"]) <= 300
    assert len(r["snippet"]) <= 500


def test_empty_results_returned_as_empty_list() -> None:
    """No matches -> [] (caller must say 'insufficient evidence')."""
    tool = _tool([])
    assert tool.search("xyzzy nonexistent", before="2026-06-01") == []


def test_limit_respected() -> None:
    arts = [_FakeArticle(f"2026-01-{d:02d}", f"T{d}", f"http://x/{d}") for d in range(1, 15)]
    tool = _tool(arts)
    out = tool.search("rates", before="2026-06-01", limit=5)
    assert len(out) == 5


# --- audit logging ---


def test_search_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    tool = _tool([_FakeArticle("2026-02-15", "T", "http://x/1")])
    with caplog.at_level(logging.INFO, logger="td_one.agent.news_tool"):
        tool.search("mortgage rates", before="2026-06-01", limit=3)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("before=2026-06-01" in m and "mortgage rates" in m for m in msgs)


# --- prompt contract ---


def test_prompt_covers_pit_contract() -> None:
    p = NEWS_SYSTEM_PROMPT
    for needle in (
        "before",
        "INSUFFICIENT EVIDENCE",
        "hallucinate",
        "NEVER infer future",
        "CITE",
    ):
        assert needle in p, f"prompt missing {needle!r}"


def test_invalid_backend_rejected() -> None:
    with pytest.raises(ValueError, match="backend"):
        NewsSearchTool(backend="smoke-signals")
