#!/usr/bin/env python3
"""Gap-fill GDELT backfill for specific (year, month, topic) combos.

Used when the main backfill hits throttling and skips combos. Runs with
conservative pacing (20s between requests, long backoff on 429).

Usage:
    uv run python scripts/fetch_news_gdelt_gapfill.py --jobs /tmp/gdelt_gaps.json
    # jobs file: [["2021","01","mortgage rates"], ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_news_gdelt import fetch_month, store  # noqa: E402

from td_one.news import init_db  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON file: [[year, month, topic], ...]")
    ap.add_argument("--db", default="data/news/news_2026.db")
    args = ap.parse_args()

    jobs = json.loads(Path(args.jobs).read_text())
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[1] / db_path
    init_db(db_path)

    total = 0
    failed = []
    for year_s, month_s, topic in jobs:
        year, month = int(year_s), int(month_s)
        print(f"Gap-fill: {topic} {year}-{month:02d}...", flush=True)
        try:
            arts = fetch_month(topic, year, month)
        except Exception as e:  # noqa: BLE001
            print(f"  !! error: {e}")
            failed.append([year_s, month_s, topic])
            time.sleep(180)
            continue
        n = store(db_path, arts)
        total += n
        print(f"  {len(arts)} articles, {n} new (total new: {total})", flush=True)
        if len(arts) == 0:
            # Suspicious for our high-coverage topics — likely soft-throttle.
            print("  !! 0 articles, will retry at end", flush=True)
            failed.append([year_s, month_s, topic])
        time.sleep(20)

    if failed:
        Path("/tmp/gdelt_gaps_remaining.json").write_text(json.dumps(failed, indent=2))
        print(f"{len(failed)} still failed, listed in /tmp/gdelt_gaps_remaining.json")
    print(f"Done. {total} new articles.")


if __name__ == "__main__":
    main()
