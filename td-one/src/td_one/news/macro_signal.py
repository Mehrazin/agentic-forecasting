"""Macro Signal Service: LLM-read monthly categorical macro signals.

A dedicated news-reading LLM reads one month's news across ~22 macro
categories and emits, per (month, category), a structured signal::

    {direction: -2..+2, risk: low|med|high, certainty: 0..1,
     summary: "1-2 sentences citing article titles"}

Signals are stored in the news DB's ``macro_signals`` table and consumed
PIT-safely via :func:`signals_asof` — forecasts at origin ``O`` may only
use signals from months ``<= O``. Raw news feeding (exp44/exp46) was an
honest negative; the hypothesis tested here (exp70) is that *precomputed,
calibrated* signals behave differently: the LLM synthesizes once per
month, the signals are validated against realized data on training
periods, and only categories passing the directional-accuracy gate go
live downstream.

Extraction LLM: the Vector proxy (Gemini Flash) — cheap, high-throughput.
Perplexity credits are spent only on fetching, never on extraction.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Per-category rubric: what direction -2..+2 means. Must agree with the
# calibration anchors in experiments/exp70 (sign conventions live here).
DIRECTION_GUIDE: dict[str, str] = {
    "immigration_law": "Policy direction: -2 = sharply loosening (more open), +2 = sharply tightening (restrictive).",
    "food_prices": "Price direction: -2 = falling fast, +2 = rising fast.",
    "basic_needs": "Cost-of-living direction: -2 = falling fast, +2 = rising fast.",
    "income_tax": "Household tax burden direction: -2 = falling fast (cuts/credits), +2 = rising fast.",
    "minimum_wage": "Wage-floor direction: -2 = cuts/freezes, +2 = large broad increases.",
    "salary_region": "Wage direction: -2 = falling fast, +2 = rising fast.",
    "school_childcare": "Cost/policy-burden direction: -2 = falling fast, +2 = rising fast.",
    "services_costs": "Services price direction: -2 = falling fast, +2 = rising fast.",
    "inflation": "Inflation direction: -2 = disinflating fast, +2 = accelerating fast.",
    "elections_politics": "Political disruption: -2 = strong stability/continuity, +2 = major disruption/shift.",
    "wars_geopolitics": "Conflict intensity: -2 = de-escalation, +2 = major escalation.",
    "pandemics_health": "Outbreak severity: -2 = subsiding, +2 = severe/worsening.",
    "flight_prices": "Fare direction: -2 = falling fast, +2 = rising fast.",
    "car_prices": "Price direction: -2 = falling fast, +2 = rising fast.",
    "global_trade": "Trade openness: -2 = opening up, +2 = barriers rising (tariffs/frictions).",
    "ai_startups": "Sector momentum: -2 = bust, +2 = boom.",
    "markets_sp500": "Market direction: -2 = sharp selloff, +2 = strong rally.",
    "central_bank_rates": "Policy stance: -2 = strong dovish tilt (cuts), +2 = strong hawkish tilt (hikes).",
    "housing": "Market heat: -2 = sharp cooling, +2 = strong heating.",
    "labour": "Unemployment direction: -2 = falling fast, +2 = rising fast.",
    "fx": "CAD direction: -2 = weakening fast, +2 = strengthening fast.",
    "small_business": "Business conditions: -2 = improving fast, +2 = deteriorating fast.",
}

RISKS = ("low", "med", "high")


@dataclass(frozen=True)
class MacroSignal:
    month: str  # YYYY-MM
    category: str
    direction: int  # -2..+2
    risk: str  # low|med|high
    certainty: float  # 0..1
    summary: str  # 1-2 sentences citing article titles
    n_articles: int


def ensure_signal_table(con: sqlite3.Connection) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS macro_signals (
            month TEXT, category TEXT,
            direction INTEGER, risk TEXT, certainty REAL,
            summary TEXT, n_articles INTEGER,
            PRIMARY KEY (month, category))"""
    )
    con.commit()


def get_category_articles(
    con: sqlite3.Connection, category: str, ym: str, max_articles: int = 15
) -> list[dict]:
    """Articles for one (category, month) cell. PIT: pub_date within month."""
    rows = con.execute(
        """SELECT a.pub_date, a.title, a.url
           FROM articles a JOIN article_categories c ON a.url = c.url
           WHERE c.category = ? AND a.pub_date LIKE ? ORDER BY a.pub_date""",
        (category, f"{ym}%"),
    ).fetchall()
    return [
        {"pub_date": r[0], "title": r[1], "url": r[2]} for r in rows[:max_articles]
    ]


def signals_asof(con: sqlite3.Connection, origin_ym: str) -> dict[str, MacroSignal]:
    """All signals from months <= ``origin_ym`` — the PIT-safe accessor.

    Keyed ``"{category}"`` (one signal per category per month; the latest
    available month <= origin wins per category).
    """
    rows = con.execute(
        """SELECT month, category, direction, risk, certainty, summary, n_articles
           FROM macro_signals WHERE month <= ? ORDER BY month""",
        (origin_ym,),
    ).fetchall()
    out: dict[str, MacroSignal] = {}
    for r in rows:
        out[r[1]] = MacroSignal(*r)
    return out


def _prompt(category: str, ym: str, articles: list[dict]) -> str:
    titles = "\n".join(f"- [{a['pub_date']}] {a['title']}" for a in articles)
    return f"""You are a macro analyst summarizing one month of news for a forecasting system.

Month: {ym}
Category: {category}
Direction rubric: {DIRECTION_GUIDE.get(category, 'Direction: -2 = strongly negative, +2 = strongly positive.')}

Headlines ({len(articles)} articles):
{titles}

Task: synthesize these headlines into ONE structured signal for this month and category.
- direction: integer from -2 to +2 per the rubric (0 = neutral/no signal).
- risk: "low", "med", or "high" — how risky the macro backdrop is for household finances in this category.
- certainty: 0.0 to 1.0 — your confidence. Few/contradictory headlines => low certainty.
- summary: 1-2 sentences citing specific headline titles in quotes.

Respond with ONLY a JSON object: {{"direction": int, "risk": "...", "certainty": float, "summary": "..."}}"""


def _parse(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    try:
        direction = int(obj["direction"])
        risk = str(obj["risk"])
        certainty = float(obj["certainty"])
        summary = str(obj["summary"])
    except (KeyError, TypeError, ValueError):
        return None
    if direction not in (-2, -1, 0, 1, 2):
        return None
    if risk not in RISKS:
        return None
    if not 0.0 <= certainty <= 1.0:
        return None
    if not summary:
        return None
    return {"direction": direction, "risk": risk, "certainty": certainty,
            "summary": summary[:1000]}


def extract_signal(
    client: object,  # VectorProxyClient-compatible: .complete(prompt, json_mode=True)
    category: str,
    ym: str,
    articles: list[dict],
    retries: int = 1,
) -> MacroSignal | None:
    """Run the LLM synthesis for one (month, category) cell.

    Returns None when there are no articles or the LLM output fails
    validation (both are honest outcomes, not errors).
    """
    if not articles:
        return None
    prompt = _prompt(category, ym, articles)
    for attempt in range(retries + 1):
        try:
            raw = client.complete(prompt, json_mode=True)  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 — transient LLM errors skip the cell
            logger.warning("extract %s %s failed: %s", ym, category, e)
            return None
        parsed = _parse(raw)
        if parsed:
            return MacroSignal(
                month=ym, category=category, n_articles=len(articles), **parsed
            )
        logger.warning("extract %s %s: invalid JSON (attempt %d)", ym, category, attempt)
    return None


def extract_pending(
    db_path: str, client: object, months: list[str], categories: list[str]
) -> dict[str, int]:
    """Extract signals for all (month, category) cells not yet in the table.

    Resume-safe: the macro_signals PK (month, category) is the log.
    Returns counts per category.
    """
    con = sqlite3.connect(db_path)
    try:
        ensure_signal_table(con)
        counts: dict[str, int] = {}
        for ym in months:
            for cat in categories:
                exists = con.execute(
                    "SELECT 1 FROM macro_signals WHERE month = ? AND category = ?",
                    (ym, cat),
                ).fetchone()
                if exists:
                    continue
                arts = get_category_articles(con, cat, ym)
                sig = extract_signal(client, cat, ym, arts)
                if sig is None:
                    continue
                con.execute(
                    "INSERT OR REPLACE INTO macro_signals"
                    " (month, category, direction, risk, certainty, summary, n_articles)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (sig.month, sig.category, sig.direction, sig.risk,
                     sig.certainty, sig.summary, sig.n_articles),
                )
                con.commit()
                counts[cat] = counts.get(cat, 0) + 1
        return counts
    finally:
        con.close()
