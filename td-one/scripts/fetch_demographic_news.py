"""Fetch demographic-specific news via Perplexity; cache everything.

Queries target borrower demographics relevant to attrition grouping:
provinces, first-time buyers, seniors, newcomers, income segments.

**Every fetch enriches the SQLite cache** (``data/news/news_2026.db``) —
never re-fetch. Resume-safe: tracks completed (topic, month) pairs in
``fetch_log`` and skips them on restart.

Usage:
    .venv/bin/python scripts/fetch_demographic_news.py [--months N] [--dry-run]

Credits: modest by design — ~12 topics x N months, one Perplexity call
each. The user approved a credit increase; this script stays well under
$5 for the default 6-month window.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.news.perplexity import PerplexityClient

TOPICS = [
    "Ontario first-time home buyers mortgage",
    "British Columbia housing market borrowers",
    "Alberta mortgage borrowers oil economy",
    "Quebec home buyers mortgage rates",
    "Canadian seniors mortgage renewal retirement",
    "newcomer immigrant mortgage Canada",
    "low income household mortgage stress Canada",
    "high income borrower jumbo mortgage Canada",
    "young borrowers under 35 mortgage Canada",
    "self-employed borrower mortgage approval Canada",
    "rental investor mortgage Canada",
    "rural borrower mortgage Atlantic Canada",
]

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"


def _ensure_fetch_log(con: sqlite3.Connection) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS fetch_log (
            topic TEXT, month TEXT, n_articles INT,
            fetched_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (topic, month))"""
    )


def _month_range(n: int) -> list[tuple[str, str]]:
    """Last ``n`` complete months as (after, before) YYYY-MM-DD pairs."""
    today = date.today()
    out = []
    y, m = today.year, today.month - 1
    if m == 0:
        y, m = y - 1, 12
    for _ in range(n):
        before = f"{y}-{m:02d}-01"
        # after = first day of this month
        after = before
        out.append((after, before))
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return list(reversed(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH)
    _ensure_fetch_log(con)
    done = {
        (r[0], r[1]) for r in con.execute("SELECT topic, month FROM fetch_log")
    }

    client = PerplexityClient(preset="fast")
    total = 0
    for topic in TOPICS:
        for after, before in _month_range(args.months):
            key = (topic, after[:7])
            if key in done:
                continue
            if args.dry_run:
                print(f"WOULD fetch: {topic} [{after}..{before})")
                continue
            try:
                articles = client.search(topic, before=before, after=after, limit=5)
            except Exception as e:  # noqa: BLE001 — resume-safe
                print(f"FAILED {topic} {after}: {e}")
                continue
            n = 0
            for a in articles:
                try:
                    con.execute(
                        "INSERT OR IGNORE INTO articles "
                        "(pub_date, title, url, source) VALUES (?,?,?,?)",
                        (a.pub_date, a.title, a.url, "perplexity-demographic"),
                    )
                    n += 1
                except sqlite3.Error:
                    pass
            con.execute(
                "INSERT OR REPLACE INTO fetch_log (topic, month, n_articles)"
                " VALUES (?,?,?)",
                (topic, after[:7], n),
            )
            con.commit()
            total += n
            print(f"fetched {topic} {after[:7]}: {n} articles")
    con.close()
    print(f"done: {total} new articles cached")


if __name__ == "__main__":
    main()
