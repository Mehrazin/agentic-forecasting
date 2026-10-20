#!/usr/bin/env python3
"""Run Macro Signal extraction over all fetched (month, category) cells.

Uses the Vector proxy (Gemini Flash) — no Perplexity credits. Resume-safe:
cells already in ``macro_signals`` are skipped.

Usage:
    .venv/bin/python scripts/run_macro_signal_extraction.py [--months 36]
        [--model flash-lite] [--db data/news/news_2026.db]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_one.llm_rungs.vector_client import VectorProxyClient  # noqa: E402
from td_one.news.macro_signal import extract_pending  # noqa: E402
from scripts.fetch_macro_signal_news import CATEGORIES, last_complete_months  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--model", default="flash-lite")
    ap.add_argument("--db", default="data/news/news_2026.db")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[1] / db_path
    months = last_complete_months(args.months)
    con = sqlite3.connect(db_path)
    have = con.execute("SELECT COUNT(*) FROM article_categories").fetchone()[0]
    con.close()
    print(f"article_categories rows: {have}; extracting {len(months)} months x "
          f"{len(CATEGORIES)} categories", flush=True)

    client = VectorProxyClient(model=args.model, call_budget=1000)
    counts = extract_pending(str(db_path), client, months, sorted(CATEGORIES))
    print(f"extracted: {counts}")
    con = sqlite3.connect(db_path)
    n = con.execute("SELECT COUNT(*) FROM macro_signals").fetchone()[0]
    con.close()
    print(f"macro_signals total: {n}")


if __name__ == "__main__":
    main()
