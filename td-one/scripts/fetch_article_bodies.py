"""Fetch article bodies from URLs into the news DB cache.

Fills the ``article_bodies`` table (url, body, fetched_at, fetch_status,
word_count). Uses trafilatura for readability-style extraction. Respectful:
browser User-Agent, per-host throttle, timeouts. Everything cached —
never re-fetch a URL with a terminal status.

Usage:
    .venv/bin/python scripts/fetch_article_bodies.py --limit 500
    .venv/bin/python scripts/fetch_article_bodies.py --limit 500 --offset 500
    .venv/bin/python scripts/fetch_article_bodies.py --all   # full run

Fetch statuses: ok | empty | blocked | error | skipped
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import trafilatura  # noqa: E402

logger = logging.getLogger(__name__)

DB = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
TIMEOUT = 20
HOST_DELAY = 1.0  # seconds between hits to the same host


def _host(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001  # network fetch: broad catch intentional
        return ""


def fetch_body(url: str) -> tuple[str, str]:
    """Return (status, body). Never raises."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            if resp.status != 200:
                return "blocked", ""
            ctype = resp.headers.get("Content-Type", "")
            if "html" not in ctype and "text" not in ctype:
                return "skipped", ""
            html = resp.read(2_000_000).decode("utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001  # network fetch: broad catch intentional
        logger.debug("fetch failed %s: %s", url, e)
        return "error", ""
    try:
        body = trafilatura.extract(html, include_comments=False) or ""
    except Exception as e:  # noqa: BLE001  # network fetch: broad catch intentional
        logger.debug("extract failed %s: %s", url, e)
        return "error", ""
    body = body.strip()
    if len(body.split()) < 30:
        return "empty", body
    return "ok", body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    con = sqlite3.connect(DB)
    cur = con.cursor()

    q = """
        SELECT a.url FROM articles a
        LEFT JOIN article_bodies b ON a.url = b.url
        WHERE b.url IS NULL OR b.fetch_status IN ('error', 'pending')
        ORDER BY a.pub_date DESC
    """
    urls = [r[0] for r in cur.execute(q).fetchall()]
    if not args.all:
        urls = urls[args.offset : args.offset + args.limit]
    logger.info("urls to fetch: %d", len(urls))

    last_hit: dict[str, float] = {}
    counts: dict[str, int] = {}
    now = datetime.datetime.now(datetime.UTC).isoformat()
    for i, url in enumerate(urls):
        host = _host(url)
        dt = time.time() - last_hit.get(host, 0)
        if dt < HOST_DELAY:
            time.sleep(HOST_DELAY - dt)
        last_hit[host] = time.time()
        status, body = fetch_body(url)
        counts[status] = counts.get(status, 0) + 1
        cur.execute(
            "INSERT OR REPLACE INTO article_bodies "
            "(url, body, fetched_at, fetch_status, word_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (url, body, now, status, len(body.split())),
        )
        if (i + 1) % 50 == 0:
            con.commit()
            logger.info("progress %d/%d %s", i + 1, len(urls), counts)
    con.commit()
    total_ok = cur.execute(
        "SELECT COUNT(*) FROM article_bodies WHERE fetch_status='ok'"
    ).fetchone()[0]
    logger.info("done. this run: %s. total ok in cache: %d", counts, total_ok)
    con.close()


if __name__ == "__main__":
    main()
