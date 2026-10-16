#!/usr/bin/env python3
"""Backfill the 2026 news DB from Perplexity Agent API (quality path).

Uses the vault surrogate credential (custom.perplexity) — the key never
touches this process. Writes into the SAME SQLite DB as the GDELT backfill
(scripts/fetch_news_gdelt.py); dedupe is by URL.

Usage:
    uv run python scripts/fetch_news_perplexity.py [--months 1 2 3]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_one.news import init_db  # noqa: E402
from td_one.news.perplexity import PerplexityClient  # noqa: E402

TOPICS = [
    "Canada mortgage rates",
    "Bank of Canada interest rate decision",
    "Canada housing market",
    "Canada government bond yields",
    "Canada inflation",
]


def month_cutoff(year: int, month: int) -> str:
    """First day of the NEXT month — the 'before' cutoff for this month."""
    if month == 12:
        return f"{year + 1}-01-01"
    return f"{year}-{month + 1:02d}-01"


def store(db_path: Path, articles: list) -> int:
    con = sqlite3.connect(db_path)
    n = 0
    try:
        for a in articles:
            try:
                cur = con.execute(
                    "INSERT OR IGNORE INTO articles (pub_date, title, url, source)"
                    " VALUES (?, ?, ?, ?)",
                    (a.pub_date, a.title, a.url, "perplexity"),
                )
                n += cur.rowcount
            except sqlite3.Error:
                pass
        con.commit()
    finally:
        con.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--db", default="data/news/news_2026.db")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--preset", default="fast")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[1] / db_path
    init_db(db_path)

    client = PerplexityClient(preset=args.preset)
    total = 0
    for month in args.months:
        cutoff = month_cutoff(args.year, month)
        want_prefix = f"{args.year}-{month:02d}"
        for topic in TOPICS:
            print(f"Perplexity [{args.preset}]: {topic} before {cutoff}...", flush=True)
            try:
                arts = client.search(topic, before=cutoff, limit=10)
            except Exception as e:  # noqa: BLE001 — keep going on transient errors
                print(f"  !! error: {e}")
                arts = []
            # Keep only articles actually from this month.
            arts = [a for a in arts if a.pub_date.startswith(want_prefix)]
            n = store(db_path, arts)
            total += n
            print(f"  {len(arts)} in-month, {n} new (total new: {total})", flush=True)
            time.sleep(3)
    print(f"Done. {total} new articles in {db_path}")


if __name__ == "__main__":
    main()
