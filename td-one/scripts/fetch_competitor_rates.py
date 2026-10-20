"""Fetch competitor mortgage rates via Perplexity for refi-pressure analysis.

The refi-spread trigger (competitor rate - borrower's current rate) is a
core DGP driver in SimulatorV2 that we currently simulate. This pulls REAL
major-bank posted rates (5yr fixed, variable) and broker discount rates,
monthly, for the last 24 months.

Storage: competitor_rates table in data/news/news_2026.db
(schema: month, lender, product, rate_pct, source_url, fetched_at).
PIT-safe by construction: each row is as-of its month; experiments must
filter month <= origin.

Usage:
    .venv/bin/python scripts/fetch_competitor_rates.py [--months 24]

Credits: one Perplexity call per month (~24 calls). Resume-safe via a
fetch_log table (comp_rates_log) — never re-fetch a done month.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from td_one.news.perplexity import PerplexityClient

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "news" / "news_2026.db"

LENDERS = ["TD Bank", "RBC", "BMO", "Scotiabank", "CIBC"]
PRODUCTS = ["5yr_fixed_posted", "variable_posted", "broker_discount_5yr"]


def _ensure_tables(con: sqlite3.Connection) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS competitor_rates (
            month TEXT NOT NULL,
            lender TEXT NOT NULL,
            product TEXT NOT NULL,
            rate_pct REAL NOT NULL,
            source_url TEXT DEFAULT '',
            fetched_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (month, lender, product)
        )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS comp_rates_log (
            month TEXT PRIMARY KEY,
            n_rates INTEGER,
            fetched_at TEXT DEFAULT (datetime('now'))
        )"""
    )


def _month_range(n: int) -> list[str]:
    today = datetime.now(UTC).date()
    out = []
    y, m = today.year, today.month - 1  # last complete month
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return sorted(out)


def parse_rates(text: str) -> list[dict]:
    """Parse a JSON array of rate objects from an LLM response.

    Tolerant of fence wrapping and surrounding prose. Each object must
    have lender, product, and rate_pct (or rate) keys.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        lender = str(it.get("lender", "")).strip()
        product = str(it.get("product", "")).strip()
        rate = it.get("rate_pct", it.get("rate"))
        try:
            rate_f = float(rate)
        except (TypeError, ValueError):
            continue
        if lender and product and 0.5 <= rate_f <= 15.0:
            out.append(
                {"lender": lender, "product": product, "rate_pct": rate_f,
                 "source_url": str(it.get("source_url", ""))[:500]}
            )
    return out


def fetch_month(client: PerplexityClient, month: str) -> list[dict]:
    """One Perplexity call for one month's competitor rates."""
    next_m = (
        f"{int(month[:4]) + (1 if month[5:7] == '12' else 0):04d}-"
        f"{(int(month[5:7]) % 12) + 1:02d}"
    )
    prompt = (
        f"What were the posted mortgage rates in Canada in {month} "
        f"(published before {next_m}-01)? For each of these lenders: "
        f"{', '.join(LENDERS)} — plus the best broker discount rate "
        f"available that month — return a JSON array of objects with keys: "
        f"lender, product (one of: 5yr_fixed_posted, variable_posted, "
        f"broker_discount_5yr), rate_pct (number), source_url. "
        f"Use 'Broker' as lender for the broker discount rate. "
        f"Return ONLY the JSON array."
    )
    data = client._call(prompt)
    return parse_rates(PerplexityClient._extract_text(data))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    args = ap.parse_args()

    con = sqlite3.connect(DB_PATH)
    _ensure_tables(con)
    done = {r[0] for r in con.execute("SELECT month FROM comp_rates_log")}
    client = PerplexityClient(preset="fast")
    total = 0
    for month in _month_range(args.months):
        if month in done:
            continue
        try:
            rates = fetch_month(client, month)
        except Exception as e:  # noqa: BLE001 — resume-safe
            print(f"FAILED {month}: {e}")
            continue
        for r in rates:
            con.execute(
                "INSERT OR REPLACE INTO competitor_rates "
                "(month, lender, product, rate_pct, source_url) "
                "VALUES (?,?,?,?,?)",
                (month, r["lender"], r["product"], r["rate_pct"], r["source_url"]),
            )
        con.execute(
            "INSERT OR REPLACE INTO comp_rates_log (month, n_rates) VALUES (?,?)",
            (month, len(rates)),
        )
        con.commit()
        total += len(rates)
        print(f"{month}: {len(rates)} rates")
    con.close()
    print(f"done: {total} new rates cached")


if __name__ == "__main__":
    main()
