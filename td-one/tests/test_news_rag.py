"""Tests for the news RAG tool (TDD — written before implementation)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from td_one.news import init_db
from td_one.news.rag import (
    DEFAULT_ABSTAIN_THRESHOLD,
    RAGTool,
    RankedArticle,
    _validate_cutoff,
)


@pytest.fixture()
def kb_path(tmp_path: Path) -> Path:
    """Build a small deterministic KB with known dates/titles."""
    db = tmp_path / "kb.db"
    init_db(db)
    rows = [
        # (pub_date, title, url) — two relevant, one junk, one future
        ("2026-02-10", "Bank of Canada holds policy rate at 2.25 percent", "http://x/1"),
        ("2026-02-12", "Canadian mortgage rates drift lower in February", "http://x/2"),
        ("2026-02-15", "Local hockey team wins championship game", "http://x/3"),
        ("2026-09-01", "Bank of Canada cuts rate in September surprise", "http://x/4"),
    ]
    con = sqlite3.connect(db)
    try:
        con.executemany(
            "INSERT INTO articles (pub_date, title, url, source) VALUES (?, ?, ?, 'test')",
            rows,
        )
        con.commit()
    finally:
        con.close()
    return db


def _fake_groq(scores: list[float]) -> object:
    """Return a Groq-shaped stub with canned relevance scores as JSON."""

    class _Fake:
        def complete(
            self,
            prompt: str,
            system: str | None = None,
            json_mode: bool = False,
            timeout_s: float = 120.0,
        ) -> str:
            import json as _json

            return _json.dumps([{"idx": i, "relevance": s} for i, s in enumerate(scores)])

    return _Fake()


# --- cutoff validation -------------------------------------------------


def test_cutoff_accepts_valid_past_date() -> None:
    assert _validate_cutoff("2026-03-01") == "2026-03-01"


def test_cutoff_accepts_date_object() -> None:
    assert _validate_cutoff(date(2026, 3, 1)) == "2026-03-01"


def test_cutoff_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        _validate_cutoff("2026/03/01")
    with pytest.raises(ValueError):
        _validate_cutoff("not-a-date")


def test_cutoff_rejects_future() -> None:
    future = (datetime.now(UTC).date() + timedelta(days=30)).isoformat()
    with pytest.raises(ValueError, match="future"):
        _validate_cutoff(future)


def test_cutoff_rejects_wrong_type() -> None:
    with pytest.raises((TypeError, ValueError)):
        _validate_cutoff(20260301)  # type: ignore[arg-type]


# --- input validation --------------------------------------------------


def test_search_rejects_empty_query(kb_path: Path) -> None:
    tool = RAGTool(db_path=kb_path)
    with pytest.raises(ValueError, match="non-empty"):
        tool.search("", before="2026-03-01")
    with pytest.raises(ValueError, match="non-empty"):
        tool.search("   ", before="2026-03-01")


def test_search_rejects_bad_limit(kb_path: Path) -> None:
    tool = RAGTool(db_path=kb_path)
    with pytest.raises(ValueError):
        tool.search("mortgage", before="2026-03-01", limit=0)
    with pytest.raises(ValueError):
        tool.search("mortgage", before="2026-03-01", limit=99)


# --- stage 1: BM25 + PIT ----------------------------------------------


def test_bm25_returns_candidates_before_cutoff(kb_path: Path) -> None:
    tool = RAGTool(db_path=kb_path, groq=None)  # BM25-only
    res = tool.search("Bank of Canada rate", before="2026-03-01", limit=5)
    assert not res.abstained  # BM25-only never abstains on relevance
    dates = [a.pub_date for a in res.articles]
    assert dates, "expected BM25 hits"
    assert all(d < "2026-03-01" for d in dates)
    # The September article must NOT leak through a March cutoff.
    assert all("September" not in a.title for a in res.articles)


def test_bm25_empty_when_nothing_matches(kb_path: Path) -> None:
    tool = RAGTool(db_path=kb_path, groq=None)
    res = tool.search("zebra migration patterns", before="2026-03-01")
    assert res.articles == []
    assert res.abstained is True


# --- stage 2: rerank ---------------------------------------------------


def test_rerank_orders_by_relevance(kb_path: Path) -> None:
    # "Canada" matches 2 articles before the cutoff; fake scores put
    # the second BM25 candidate on top after reranking.
    tool = RAGTool(db_path=kb_path, groq=_fake_groq([0.2, 0.9]))  # type: ignore[arg-type]
    res = tool.search("Canada OR Canadian", before="2026-03-01", limit=2)
    assert not res.abstained
    assert len(res.articles) == 2
    assert res.articles[0].relevance == pytest.approx(0.9)
    assert res.articles[1].relevance == pytest.approx(0.2)
    assert res.backend == "bm25+rerank"


def test_rerank_prompt_scores_junk_zero() -> None:
    prompt = RAGTool._build_rerank_prompt(
        "Bank of Canada rate decision",
        [
            RankedArticle("2026-02-10", "Bank of Canada holds rate", "http://x/1", "", 0.0),
            RankedArticle("2026-02-15", "Hockey team wins", "http://x/3", "", 0.0),
        ],
    )
    assert "Bank of Canada rate decision" in prompt
    assert "Hockey team wins" in prompt
    assert "0" in prompt and "1" in prompt  # scale documented


def test_rerank_malformed_json_falls_back_to_bm25(kb_path: Path) -> None:
    class _Bad:
        def complete(self, *a: object, **k: object) -> str:
            return "this is not json ["

    tool = RAGTool(db_path=kb_path, groq=_Bad())  # type: ignore[arg-type]
    res = tool.search("Bank of Canada", before="2026-03-01", limit=5)
    # Degraded but functional: BM25 order, backend flagged.
    assert res.backend == "bm25"
    assert len(res.articles) > 0


def test_rerank_groq_error_falls_back_to_bm25(kb_path: Path) -> None:
    class _Boom:
        def complete(self, *a: object, **k: object) -> str:
            raise RuntimeError("Groq 429 exploded")

    tool = RAGTool(db_path=kb_path, groq=_Boom())  # type: ignore[arg-type]
    res = tool.search("Bank of Canada", before="2026-03-01", limit=5)
    assert res.backend == "bm25"
    assert len(res.articles) > 0


# --- stage 3: abstention ------------------------------------------------


def test_abstains_when_nothing_relevant(kb_path: Path) -> None:
    tool = RAGTool(
        db_path=kb_path,
        groq=_fake_groq([0.1, 0.05]),  # type: ignore[arg-type]
        abstain_threshold=0.4,
    )
    res = tool.search("Canada OR Canadian", before="2026-03-01")
    assert res.abstained is True
    assert res.articles == []


def test_no_abstain_when_top_passes_threshold(kb_path: Path) -> None:
    tool = RAGTool(
        db_path=kb_path,
        groq=_fake_groq([0.1, 0.85]),  # type: ignore[arg-type]
        abstain_threshold=0.4,
    )
    res = tool.search("Canada OR Canadian", before="2026-03-01", limit=5)
    assert res.abstained is False
    assert res.articles[0].relevance == pytest.approx(0.85)


def test_abstain_threshold_configurable(kb_path: Path) -> None:
    tool = RAGTool(
        db_path=kb_path,
        groq=_fake_groq([0.5, 0.5]),  # type: ignore[arg-type]
        abstain_threshold=0.9,
    )
    res = tool.search("Canada OR Canadian", before="2026-03-01")
    assert res.abstained is True
    assert pytest.approx(0.4) == DEFAULT_ABSTAIN_THRESHOLD


def test_bm25_only_mode_never_llm_abstains(kb_path: Path) -> None:
    # Without a reranker there are no relevance scores; an empty BM25
    # result still abstains (nothing to show), but a hit never does.
    tool = RAGTool(db_path=kb_path, groq=None)
    res = tool.search("mortgage", before="2026-03-01")
    assert res.abstained is False
    assert len(res.articles) > 0
