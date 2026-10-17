#!/usr/bin/env python3
"""Backfill the 2026 news DB from GDELT DOC API.

Patient by design: GDELT rate-limits to ~1 request per 5 seconds and each
request takes 10-20s. A full 2026 backfill (10 months x 5 topics) takes
~30-45 minutes. Run in background.

Usage:
    uv run python scripts/fetch_news_gdelt.py [--months 1 2 3] [--db PATH]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_one.news import init_db  # noqa: E402

# Topics that matter for mortgage attrition regime classification.
TOPICS = [
    '"mortgage rates"',
    '"Bank of Canada" interest rate',
    "housing market Canada",
    "bond yields Canada",
    "inflation Canada",
]

API = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_month(topic: str, year: int, month: int) -> list[dict]:
    """Fetch up to 250 articles for a topic/month. Returns article dicts."""
    start = f"{year}{month:02d}01000000"
    # Last day of month
    if month == 12:
        end = f"{year}1231235959"
    else:
        last_day = (date(year, month + 1, 1) - date(year, month, 1)).days
        # Simpler: use date arithmetic via replacement
        import calendar

        last_day = calendar.monthrange(year, month)[1]
        end = f"{year}{month:02d}{last_day:02d}235959"
    params = {
        "query": topic,
        "mode": "artlist",
        "format": "json",
        "startdatetime": start,
        "enddatetime": end,
        "maxrecords": 250,
        "sort": "datedesc",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "TD-One research"})
    # GDELT 429s are transient: back off and retry rather than skipping the
    # month entirely (a skipped month is a permanent coverage gap).
    raw = ""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                wait = 60 * (attempt + 1)
                print(f"  !! 429, backing off {wait}s (attempt {attempt + 1}/5)...", flush=True)
                time.sleep(wait)
                continue
            raise
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  !! non-JSON response for {topic} {year}-{month:02d}: {raw[:120]}")
        return []
    return data.get("articles", [])


def store(db_path: Path, articles: list[dict]) -> int:
    """Insert articles, skipping duplicates by URL. Returns new count."""
    from datetime import date as _date

    today = _date.today().isoformat()
    con = sqlite3.connect(db_path)
    n = 0
    try:
        for a in articles:
            url = a.get("url", "")
            if not url:
                continue
            # seendate format: "20260215T143000Z"
            seen = a.get("seendate", "")
            pub = seen[:10] if len(seen) >= 10 else ""
            if len(pub) == 8 and pub.isdigit():
                pub = f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}"
            elif "T" in seen:
                pub = seen.split("T")[0]
                pub = f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}"
            if not pub:
                continue
            # PIT: reject future-dated articles (GDELT can return them).
            if pub > today:
                continue
            # GDELT uses "domain", not "source".
            source = a.get("domain", "") or a.get("source", "")
            try:
                cur = con.execute(
                    "INSERT OR IGNORE INTO articles (pub_date, title, url, source)"
                    " VALUES (?, ?, ?, ?)",
                    (pub, a.get("title", "")[:500], url, source),
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
    ap.add_argument("--months", nargs="*", type=int, default=list(range(1, 11)),
                    help="Months of 2026 to fetch (default: 1-10)")
    ap.add_argument("--db", default="data/news/news_2026.db")
    ap.add_argument("--year", type=int, default=2026)
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[1] / db_path
    init_db(db_path)

    total = 0
    for month in args.months:
        for topic in TOPICS:
            print(f"Fetching {topic} {args.year}-{month:02d}...", flush=True)
            try:
                arts = fetch_month(topic, args.year, month)
            except Exception as e:  # noqa: BLE001 — keep going on transient errors
                print(f"  !! error: {e}")
                arts = []
            n = store(db_path, arts)
            total += n
            print(f"  {len(arts)} articles, {n} new (total new: {total})", flush=True)
            time.sleep(10)  # respect GDELT rate limit (429s observed at 6s)
    print(f"Done. {total} new articles in {db_path}")


if __name__ == "__main__":
    main()
