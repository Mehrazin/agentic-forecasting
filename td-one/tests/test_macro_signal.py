"""Tests for the Macro Signal Service (src/td_one/news/macro_signal.py)."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.news.macro_signal import (  # noqa: E402
    MacroSignal,
    _parse,
    ensure_signal_table,
    extract_signal,
    get_category_articles,
    signals_asof,
)


class FakeClient:
    def __init__(self, raw: str) -> None:
        self.raw = raw

    def complete(self, prompt: str, json_mode: bool = False) -> str:
        assert json_mode
        return self.raw


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE articles (pub_date TEXT, title TEXT, url TEXT UNIQUE, source TEXT)")
    con.execute(
        "CREATE TABLE article_categories (url TEXT, category TEXT, PRIMARY KEY (url, category))"
    )
    arts = [
        ("2025-06-05", "Wages rise in Ontario", "u1", "s"),
        ("2025-06-20", "Wage growth slows", "u2", "s"),
        ("2025-07-02", "Next month story", "u3", "s"),
    ]
    for a in arts:
        con.execute("INSERT INTO articles VALUES (?, ?, ?, ?)", a)
    con.execute("INSERT INTO article_categories VALUES ('u1', 'salary_region')")
    con.execute("INSERT INTO article_categories VALUES ('u2', 'salary_region')")
    con.execute("INSERT INTO article_categories VALUES ('u3', 'salary_region')")
    return con


def test_get_category_articles_pit_month_filter() -> None:
    con = _db()
    arts = get_category_articles(con, "salary_region", "2025-06")
    assert [a["url"] for a in arts] == ["u1", "u2"]  # u3 is July


def test_signals_asof_excludes_future_months() -> None:
    con = sqlite3.connect(":memory:")
    ensure_signal_table(con)
    for ym in ("2025-05", "2025-06", "2025-07"):
        con.execute(
            "INSERT INTO macro_signals VALUES (?, 'inflation', 1, 'med', 0.7, 's', 5)",
            (ym,),
        )
    got = signals_asof(con, "2025-06")
    assert set(got) == {"inflation"}
    assert got["inflation"].month == "2025-06"  # latest <= origin wins
    assert signals_asof(con, "2025-04") == {}


def test_parse_rejects_bad_direction() -> None:
    assert _parse('{"direction": 5, "risk": "low", "certainty": 0.5, "summary": "x"}') is None
    assert _parse('{"direction": 1, "risk": "extreme", "certainty": 0.5, "summary": "x"}') is None
    assert _parse('{"direction": 1, "risk": "low", "certainty": 1.5, "summary": "x"}') is None
    ok = _parse('{"direction": -1, "risk": "high", "certainty": 0.8, "summary": "cites X"}')
    assert ok is not None and ok["direction"] == -1


def test_extract_signal_none_without_articles() -> None:
    client = FakeClient('{"direction": 1, "risk": "low", "certainty": 0.5, "summary": "x"}')
    assert extract_signal(client, "inflation", "2025-06", []) is None


def test_extract_signal_builds_macrosignal() -> None:
    client = FakeClient('{"direction": 2, "risk": "high", "certainty": 0.9, "summary": "Hot."}')
    sig = extract_signal(
        client, "inflation", "2025-06", [{"pub_date": "2025-06-01", "title": "t", "url": "u"}]
    )
    assert isinstance(sig, MacroSignal)
    assert (sig.month, sig.category, sig.direction, sig.risk) == ("2025-06", "inflation", 2, "high")
    assert sig.n_articles == 1
