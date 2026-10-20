"""Tests for the competitor-rates fetcher (mock-only, no Perplexity calls)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fetch_competitor_rates import _month_range, parse_rates


def test_parse_rates_clean_json() -> None:
    text = (
        '[{"lender": "TD Bank", "product": "5yr_fixed_posted", '
        '"rate_pct": 5.99, "source_url": "https://x"}]'
    )
    out = parse_rates(text)
    assert len(out) == 1
    assert out[0]["rate_pct"] == 5.99
    assert out[0]["lender"] == "TD Bank"


def test_parse_rates_fenced_and_prose() -> None:
    text = (
        "Here are the rates:\n```json\n"
        '[{"lender": "RBC", "product": "variable_posted", "rate": 6.10}]\n'
        "```\nThat's what I found."
    )
    out = parse_rates(text)
    assert len(out) == 1
    assert out[0]["rate_pct"] == 6.10


def test_parse_rates_rejects_garbage() -> None:
    text = (
        '[{"lender": "X", "product": "y", "rate_pct": 99.9},'
        '{"lender": "", "product": "z", "rate_pct": 5.0},'
        '{"lender": "A", "product": "b", "rate_pct": "n/a"}]'
    )
    assert parse_rates(text) == []
    assert parse_rates("no json here") == []


def test_month_range_excludes_current_month() -> None:
    from datetime import UTC, datetime

    months = _month_range(3)
    assert len(months) == 3
    assert months == sorted(months)
    # All strictly in the past (YYYY-MM of last complete month or earlier).
    today = datetime.now(UTC).date()
    for m in months:
        y, mo = int(m[:4]), int(m[5:7])
        assert (y, mo) < (today.year, today.month)
