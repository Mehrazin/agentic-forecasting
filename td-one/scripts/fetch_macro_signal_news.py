#!/usr/bin/env python3
"""Feed the Macro Signal Service: per-category monthly news via Perplexity.

Fetches news for ~22 macro categories x monthly windows (last 36 complete
months) into the shared news DB (``data/news/news_2026.db``). No category
in isolation is the point — this corpus feeds a dedicated LLM that reads
ALL of it every month and emits calibrated categorical signals
(``td_one.news.macro_signal``).

Method (learned from the demographic-fetch failure, 2026-10-19): a strict
1-month ``after``/``before`` window makes niche topics return ``[]``.
Instead the month is named in the query text and only the ``before``
cutoff is set; returned articles are kept only when their real
``pub_date`` falls inside the target month (PIT-safe by construction).
Cells with < 3 in-month articles get ONE retry with a second phrasing
plus an explicit window (the backfill showed windows work for some
topics). Still-empty cells are logged as 0 — coverage is reported
honestly, never padded.

Budget: one call per (category, month) plus retries on poor cells,
hard-capped by ``--max-calls`` (default 850). At the observed
~$0.009/call this stays under ~$8. Resume-safe via ``fetch_log``
(topic = ``macro_signal:<category>``); re-runs skip completed cells.

Usage:
    .venv/bin/python scripts/fetch_macro_signal_news.py [--months 36]
        [--max-calls 850] [--db data/news/news_2026.db] [--dry-run]
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from td_one.news import init_db  # noqa: E402
from td_one.news.perplexity import PerplexityArticle, PerplexityClient  # noqa: E402

# Category -> query phrasings (no month in the text; appended at call time).
# Phrasings rotate by month index; the retry uses the next phrasing.
CATEGORIES: dict[str, list[str]] = {
    "immigration_law": [
        "Canada immigration law policy changes",
        "Canada immigration levels plan permanent residency newcomers",
        "Canada visa work permit study permit changes",
    ],
    "food_prices": [
        "Canada food prices groceries inflation",
        "Canada food price report grocery costs",
        "Canada grocery prices food inflation Statistics Canada",
    ],
    "basic_needs": [
        "Canada cost of living basic needs households",
        "Canadian household expenses utilities rent costs",
        "Canada affordability families essential expenses",
    ],
    "income_tax": [
        "Canada income tax changes brackets credits",
        "Canadian federal tax policy households benefits",
        "Canada tax refund benefit payments changes",
    ],
    "minimum_wage": [
        "Canada minimum wage increase provinces",
        "provincial minimum wage changes Canada workers",
        "Canada minimum wage policy announcement",
    ],
    "salary_region": [
        "Canada average salary wages by province region",
        "Canadian wage growth earnings Statistics Canada",
        "Canada average weekly earnings workers",
    ],
    "school_childcare": [
        "Canada childcare costs $10-a-day program",
        "Canada school education funding policy changes",
        "Canadian childcare subsidy fees parents",
    ],
    "services_costs": [
        "Canada services prices inflation telecom insurance",
        "Canadian internet phone plan prices CRTC",
        "Canada insurance premiums services costs rising",
    ],
    "inflation": [
        "Canada inflation CPI report Statistics Canada",
        "Canada consumer price index monthly",
        "Canadian inflation expectations price pressures",
    ],
    "elections_politics": [
        "Canada federal election polls political shift",
        "Canadian government policy announcement politics",
        "Canada provincial election results policy changes",
    ],
    "wars_geopolitics": [
        "war geopolitics global conflict news",
        "Russia Ukraine war Middle East conflict developments",
        "global geopolitical tensions military security",
    ],
    "pandemics_health": [
        "pandemic public health outbreak news",
        "Canada public health virus outbreak update",
        "global health emergency disease news",
    ],
    "flight_prices": [
        "airfare flight prices Canada airlines",
        "airline ticket prices travel costs news",
        "Canada air travel demand fares",
    ],
    "car_prices": [
        "Canada car prices auto market news",
        "vehicle prices electric vehicle incentives Canada",
        "Canada auto sales new car prices",
    ],
    "global_trade": [
        "global trade commerce tariffs news",
        "Canada US trade CUSMA tariff dispute",
        "world trade supply chain commerce shifts",
    ],
    "ai_startups": [
        "artificial intelligence startups industry news",
        "Canada tech startups funding AI",
        "AI artificial intelligence companies developments",
    ],
    "markets_sp500": [
        "S&P 500 stock market news outlook",
        "stock market equities rally selloff news",
        "TSX Toronto stock market news Canada",
    ],
    "central_bank_rates": [
        "Bank of Canada interest rate decision",
        "Federal Reserve interest rate decision markets",
        "central bank monetary policy rate outlook",
    ],
    "housing": [
        "Canada housing market prices home sales",
        "Canadian real estate market forecast",
        "Canada home prices mortgage market news",
    ],
    "labour": [
        "Canada unemployment jobs labour market report",
        "Canadian employment Statistics Canada jobs",
        "Canada layoffs hiring labour trends",
    ],
    "fx": [
        "Canadian dollar exchange rate outlook",
        "USD CAD currency news dollar",
        "Canadian dollar loonie forecast analysts",
    ],
    "small_business": [
        "Canada small business conditions insolvencies",
        "Canadian business bankruptcies outlook CFIB",
        "Canada small business confidence survey",
    ],
}

LOG_PREFIX = "macro_signal:"
MIN_GOOD = 3  # in-month articles below this trigger the single retry


def last_complete_months(n: int, today: dt.date | None = None) -> list[str]:
    """Return the last ``n`` complete months as YYYY-MM, newest first."""
    today = today or dt.date.today()
    y, m = today.year, today.month - 1  # current month is incomplete
    if m == 0:
        y, m = y - 1, 12
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out


def month_bounds(ym: str) -> tuple[str, str, str]:
    """(first_day, next_month_first_day, 'Month YYYY') for a YYYY-MM."""
    y, m = int(ym[:4]), int(ym[5:7])
    first = f"{ym}-01"
    nxt = f"{y + 1:04d}-01-01" if m == 12 else f"{y:04d}-{m + 1:02d}-01"
    label = f"{calendar.month_name[m]} {y}"
    return first, nxt, label


def init_fetch_log(con: sqlite3.Connection) -> None:
    con.execute(
        "CREATE TABLE IF NOT EXISTS fetch_log ("
        " ym TEXT, topic TEXT, n_new INTEGER, ts TEXT,"
        " PRIMARY KEY (ym, topic))"
    )
    con.commit()


def was_fetched(con: sqlite3.Connection, ym: str, category: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM fetch_log WHERE ym = ? AND topic = ?",
            (ym, LOG_PREFIX + category),
        ).fetchone()
        is not None
    )


def log_fetch(con: sqlite3.Connection, ym: str, category: str, n_new: int) -> None:
    con.execute(
        "INSERT OR REPLACE INTO fetch_log (ym, topic, n_new, ts) VALUES (?, ?, ?, ?)",
        (ym, LOG_PREFIX + category, n_new, dt.datetime.now(dt.UTC).isoformat()),
    )
    con.commit()


def init_article_categories(con: sqlite3.Connection) -> None:
    """Link table: which category a fetch was for (URL dedup is global,
    so one article can belong to several categories)."""
    con.execute(
        "CREATE TABLE IF NOT EXISTS article_categories ("
        " url TEXT, category TEXT, PRIMARY KEY (url, category))"
    )
    con.commit()


def store(
    con: sqlite3.Connection, articles: list[PerplexityArticle], category: str
) -> int:
    n = 0
    for a in articles:
        try:
            cur = con.execute(
                "INSERT OR IGNORE INTO articles (pub_date, title, url, source)"
                " VALUES (?, ?, ?, ?)",
                (a.pub_date, a.title, a.url, "perplexity-macro-signal"),
            )
            n += cur.rowcount
            con.execute(
                "INSERT OR IGNORE INTO article_categories (url, category)"
                " VALUES (?, ?)",
                (a.url, category),
            )
        except sqlite3.Error:
            pass
    con.commit()
    return n


def fetch_cell(
    client: PerplexityClient,
    category: str,
    ym: str,
    month_idx: int,
) -> list[PerplexityArticle]:
    """Fetch one (category, month) cell; returns in-month articles only."""
    first, nxt, label = month_bounds(ym)
    phrasings = CATEGORIES[category]
    query = f"{phrasings[month_idx % len(phrasings)]} — news from {label}"
    arts = client.search(query, before=nxt, limit=12)
    in_month = [a for a in arts if a.pub_date.startswith(ym)]
    if len(in_month) < MIN_GOOD:
        # One retry: next phrasing + explicit window (works for some topics).
        retry_q = f"{phrasings[(month_idx + 1) % len(phrasings)]} — news from {label}"
        try:
            arts2 = client.search(retry_q, before=nxt, after=first, limit=12)
        except Exception:  # noqa: BLE001 — retry is best-effort
            arts2 = []
        seen = {a.url for a in in_month}
        in_month += [a for a in arts2 if a.pub_date.startswith(ym) and a.url not in seen]
    return in_month


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=36)
    ap.add_argument("--max-calls", type=int, default=850)
    ap.add_argument("--db", default="data/news/news_2026.db")
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parents[1] / db_path
    init_db(db_path)
    con = sqlite3.connect(db_path)
    init_fetch_log(con)
    init_article_categories(con)

    months = last_complete_months(args.months)
    cells = [(cat, ym) for ym in months for cat in CATEGORIES]
    todo = [c for c in cells if not was_fetched(con, c[1], c[0])]
    print(
        f"macro-signal fetch: {len(cells)} cells, {len(todo)} todo "
        f"({len(cells) - len(todo)} already logged), cap {args.max_calls} calls",
        flush=True,
    )
    if args.dry_run:
        for cat, ym in todo[:10]:
            print(f"  WOULD fetch {cat} {ym}")
        return

    client = PerplexityClient(preset=args.preset)
    calls = 0
    total_new = 0
    coverage: dict[str, list[int]] = {}
    for idx, (cat, ym) in enumerate(todo):
        if calls >= args.max_calls:
            print(f"call cap reached ({calls}); stopping — rerun to resume", flush=True)
            break
        month_idx = months.index(ym)
        try:
            arts = fetch_cell(client, cat, ym, month_idx)
        except Exception as e:  # noqa: BLE001 — resume-safe: log and move on
            print(f"  !! {cat} {ym}: {e}", flush=True)
            arts = []
        calls += 1  # fetch_cell may spend a 2nd call on retry; counted in report
        n = store(con, arts, cat)
        log_fetch(con, ym, cat, n)
        total_new += n
        coverage.setdefault(cat, []).append(len(arts))
        if idx % 25 == 0:
            print(f"  [{idx}/{len(todo)}] {total_new} new articles, ~{calls} calls", flush=True)
        time.sleep(1)

    con.close()
    print(f"\nDone. {total_new} new articles from ~{calls} primary calls.")
    print("\nCoverage (in-month articles per category-month cell):")
    for cat in CATEGORIES:
        vals = coverage.get(cat, [])
        if vals:
            zeros = sum(1 for v in vals if v == 0)
            print(
                f"  {cat:20s} cells={len(vals):3d} mean={sum(vals) / len(vals):5.1f} "
                f"zero_cells={zeros}"
            )


if __name__ == "__main__":
    main()
