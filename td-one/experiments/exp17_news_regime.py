"""Experiment 17: News-augmented regime reader, end-to-end.

Wires three components that were built but never ran together:
  1. RAGTool (BM25 -> Groq rerank -> abstain) for news retrieval
  2. NewsSearchTool(backend="rag") as the agent-facing interface
  3. read_regime_with_news (v2 prompt: ABSTAIN, counterargument, news discipline)

For 4 historical months in 2026, retrieves news before the month's cutoff,
feeds it to the regime reader, and compares the predicted switching-rate
direction against the ACTUAL next-month 30Y mortgage rate move (FRED
MORTGAGE30US). Rising rates -> lock-in -> DOWN switching (expected);
falling rates -> UP switching.

This is a QUALITATIVE evaluation (n=4), not a backtest. It answers:
  - Does RAG retrieve relevant articles?
  - Does the regime reader produce sensible classifications?
  - Does it abstain when it should?
  - Any prompt failures or hallucinations?

Budget: 4 months x 2 searches = 8 RAG searches (each = 1 Groq rerank call),
plus 4 regime-reader calls = ~12 Groq calls total. Well within free tier.

Usage:
    cd td-one && uv run python experiments/exp17_news_regime.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.agent.groq_client import GroqChatClient  # noqa: E402
from td_one.agent.news_tool import NewsSearchTool  # noqa: E402
from td_one.agent.regime_reader import (  # noqa: E402
    RegimeReading,
    read_regime_with_news,
)

# FRED MORTGAGE30US month-end values (verified 2026-10-16).
# Used as ground truth for "what actually happened next month".
RATE_MONTH_END: dict[str, float] = {
    "2026-01": 6.10,
    "2026-02": 5.98,
    "2026-03": 6.38,
    "2026-04": 6.30,
    "2026-05": 6.53,
    "2026-06": 6.49,
    "2026-07": 6.66,
    "2026-08": 6.66,
    "2026-09": 7.03,
    "2026-10": 7.40,
}

# (month_key, cutoff, expected_switching_direction, why)
# Expected direction from the lock-in mechanism: rate rise -> DOWN switching.
MONTHS: list[tuple[str, str, str, str]] = [
    (
        "2026-02",
        "2026-03-01",
        "DOWN",
        "Mar rate jumped +0.40pp (5.98->6.38): sharp rise should suppress switching",
    ),
    (
        "2026-03",
        "2026-04-01",
        "UP",
        "Apr rate eased -0.08pp (6.38->6.30): slight relief should nudge switching up",
    ),
    (
        "2026-06",
        "2026-07-01",
        "DOWN",
        "Jul rate rose +0.17pp (6.49->6.66): continued climb should suppress switching",
    ),
    (
        "2026-08",
        "2026-09-01",
        "DOWN",
        "Sep rate jumped +0.37pp (6.66->7.03): sharp rise should suppress switching",
    ),
]

QUERIES = [
    # Keyword-style queries: FTS5 ANDs terms, so keep them short.
    # (Quoted multi-word phrases like '"Bank of Canada" rate decision'
    # return zero hits — see RAG_DESIGN.md evaluation section.)
    "Bank of Canada",
    "mortgage rates",
]


def _scenario(month_key: str) -> tuple[str, float]:
    """Build a realistic scenario description for the regime reader."""
    rate = RATE_MONTH_END[month_key]
    desc = (
        f"Canadian mortgage market, end of {month_key}. The 30-year fixed "
        f"mortgage rate stands at {rate:.2f}%. Borrowers with upcoming renewals "
        f"are deciding whether to switch lenders for a better rate or stay "
        f"with their current lender. Assess the direction of the switching-rate "
        f"shift for the coming month."
    )
    # Base switching rate: plausible 2026 level (~8%).
    return desc, 0.080


def run_month(
    month_key: str,
    cutoff: str,
    groq: GroqChatClient,
) -> dict[str, object]:
    """Run one month: RAG search -> regime reading. Returns a result dict."""
    print(f"\n[exp17] === {month_key} (cutoff {cutoff}) ===", flush=True)
    tool = NewsSearchTool(backend="rag", groq=groq, max_searches=10)

    all_articles: list[tuple[str, str, str]] = []
    searches: list[dict[str, object]] = []
    for q in QUERIES:
        try:
            hits = tool.search(q, before=cutoff, limit=5)
        except Exception as exc:  # noqa: BLE001 - log and continue
            print(f"[exp17]   search {q!r} failed: {exc}", flush=True)
            searches.append({"query": q, "error": str(exc), "n_hits": 0})
            continue
        print(f"[exp17]   {q!r}: {len(hits)} hits", flush=True)
        for h in hits[:3]:
            print(f"[exp17]     [{h['date']}] {h['title'][:80]}", flush=True)
        searches.append(
            {
                "query": q,
                "n_hits": len(hits),
                "titles": [h["title"][:120] for h in hits],
            }
        )
        for h in hits:
            triple = (h["date"], h["title"], h["url"])
            if triple not in all_articles:
                all_articles.append(triple)

    # Newest first (RAG already returns newest-first; ensure it).
    all_articles.sort(key=lambda t: t[0], reverse=True)

    # Detect which backend actually served (bm25 vs bm25+rerank).
    # NewsSearchTool doesn't expose this; infer from the RAGTool if reachable.
    # (Best-effort; the result JSON records what we know.)

    desc, base_rate = _scenario(month_key)
    reading: RegimeReading | None = None
    error: str | None = None
    try:
        reading = read_regime_with_news(
            client=groq,
            description=desc,
            base_rate=base_rate,
            articles=all_articles,
            use_v2_prompt=True,
        )
        print(
            f"[exp17]   reading: {reading.regime} @ {reading.confidence:.2f}",
            flush=True,
        )
        print(f"[exp17]   reason: {reading.reason[:150]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        print(f"[exp17]   regime reader failed: {exc}", flush=True)

    return {
        "month": month_key,
        "cutoff": cutoff,
        "searches": searches,
        "n_articles": len(all_articles),
        "articles": [{"date": d, "title": t[:150], "url": u} for d, t, u in all_articles],
        "reading": (
            {
                "regime": str(reading.regime),
                "confidence": round(reading.confidence, 3),
                "reason": reading.reason,
            }
            if reading
            else None
        ),
        "error": error,
    }


def main() -> int:
    t0 = time.time()
    groq = GroqChatClient(call_budget=30, max_tokens=4000)

    results: list[dict] = []
    for month_key, cutoff, expected, why in MONTHS:
        r = run_month(month_key, cutoff, groq)
        # Ground truth: actual next-month rate move.
        months = sorted(RATE_MONTH_END)
        i = months.index(month_key)
        next_key = months[i + 1]
        move = RATE_MONTH_END[next_key] - RATE_MONTH_END[month_key]
        r["ground_truth"] = {
            "next_month": next_key,
            "rate_move_pp": round(move, 2),
            "expected_switching": expected,
            "rationale": why,
        }
        reading = r["reading"]
        if reading:
            pred = reading["regime"]
            # Score: correct if pred matches expected, ABSTAIN is neutral,
            # wrong otherwise. NONE counts as wrong when a move was expected.
            if pred == "ABSTAIN":
                r["score"] = "abstain"
            elif pred == expected:
                r["score"] = "correct"
            else:
                r["score"] = "wrong"
        else:
            r["score"] = "error"
        print(
            f"[exp17]   truth: {next_key} move {move:+.2f}pp -> expect {expected}; "
            f"score: {r['score']}",
            flush=True,
        )
        results.append(r)
        # Be nice to the free tier between months.
        time.sleep(3)

    out = {
        "experiment": "exp17_news_regime",
        "description": (
            "End-to-end news-augmented regime reader: RAG retrieval + "
            "v2 regime classification, 4 months of 2026, qualitative."
        ),
        "n_months": len(results),
        "groq_calls_used": groq.calls,
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = ROOT / "experiments" / "exp17_news_regime_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[exp17] wrote {out_path} ({groq.calls} Groq calls)", flush=True)

    # Summary table.
    print("\n[exp17] === summary ===")
    for r in results:
        rd = r["reading"]
        pred = rd["regime"] if rd else "ERROR"
        conf = f"{rd['confidence']:.2f}" if rd else "-"
        print(
            f"  {r['month']}: pred={pred}@{conf} "
            f"expected={r['ground_truth']['expected_switching']} "
            f"move={r['ground_truth']['rate_move_pp']:+.2f}pp "
            f"-> {r['score']} ({r['n_articles']} articles)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
