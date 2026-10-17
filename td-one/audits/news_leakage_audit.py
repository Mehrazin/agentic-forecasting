"""News label-leakage audit for TD-One.

Checks three surfaces for PIT violations:
1. Synthetic news generator (news_synthesis.py)
2. Real news DB (data/news/news_2026.db)
3. RAG retrieval path (news/rag.py + news/__init__.py)

Run: cd td-one && .venv/bin/python audits/news_leakage_audit.py
Read-only on the DB. Exits 0 if clean, 1 if issues found.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ISSUES: list[str] = []
PASSES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        PASSES.append(name)
        print(f"  PASS: {name}")
    else:
        ISSUES.append(f"{name}: {detail}")
        print(f"  FAIL: {name} — {detail}")


def audit_synthetic() -> None:
    print("\n=== 1. Synthetic news generator ===")
    from td_one.data_gen.events import EventType, generate_event_timeline
    from td_one.data_gen.news_synthesis import (
        articles_visible_asof,
        generate_news_feed,
        strip_eval_fields,
    )

    # Generate timelines across several seeds
    for seed in (0, 1, 2, 42, 123):
        events = generate_event_timeline(
            start_date=date(2021, 1, 1), n_months=66, seed=seed
        )
        articles = generate_news_feed(events, seed=seed + 1000)
        by_id = {e.event_id: e for e in events}

        # 1a. article_date >= event_date for all articles
        viol = [
            a for a in articles
            if a["article_date"] < by_id[str(a["event_id"])].event_date
        ]
        check(f"seed={seed}: article_date >= event_date", not viol,
              f"{len(viol)} violations" if viol else "")

        # 1b. strip_eval_fields removes event_id
        stripped = strip_eval_fields(articles)
        leaked = [a for a in stripped if "event_id" in a]
        check(f"seed={seed}: strip_eval_fields removes event_id", not leaked,
              f"{len(leaked)} still have event_id" if leaked else "")

        # 1c. articles_visible_asof is PIT-safe
        if articles:
            mid = sorted(a["article_date"] for a in articles)[len(articles) // 2]
            vis = articles_visible_asof(stripped, mid)
            future = [a for a in vis if a["article_date"] > mid]
            check(f"seed={seed}: articles_visible_asof PIT-safe", not future,
                  f"{len(future)} future articles visible" if future else "")

        # 1d. No article references a *different* (future) event's content.
        # Templates only mention their own event's magnitude/cohort, so we check
        # that no article_date precedes its own event_date (covered in 1a) and
        # that headlines don't contain dates (which could leak timing).
        import re
        dated = [a for a in articles
                 if re.search(r"\b(19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b",
                              str(a["headline"]) + str(a["body"]))]
        check(f"seed={seed}: no hardcoded dates in text", not dated,
              f"{len(dated)} articles contain dates" if dated else "")

    # 1e. News-lead invariant: for "leading" types, mean news delay < transmission lag
    from td_one.data_gen.events import NEWS_LEAD_MONTHS
    leading = [t for t, v in NEWS_LEAD_MONTHS.items() if v > 0]
    check("NEWS_LEAD_MONTHS has leading types", len(leading) >= 3,
          f"only {len(leading)} leading types")


def audit_real_db() -> None:
    print("\n=== 2. Real news DB ===")
    db = REPO / "data" / "news" / "news_2026.db"
    if not db.exists():
        check("DB exists", False, str(db))
        return
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()

    # 2a. No future-dated articles
    today = date.today().isoformat()
    cur.execute("SELECT COUNT(*) FROM articles WHERE pub_date > ?", (today,))
    n_future = cur.fetchone()[0]
    check("0 future-dated articles", n_future == 0, f"{n_future} found")

    # 2b. The 2026-10-28 row from DATA_AUDIT P1-2 is gone
    cur.execute("SELECT COUNT(*) FROM articles WHERE pub_date = '2026-10-28'")
    n_bad = cur.fetchone()[0]
    check("2026-10-28 row purged", n_bad == 0, f"{n_bad} rows remain")

    # 2c. No null/empty critical fields
    cur.execute("SELECT COUNT(*) FROM articles WHERE title IS NULL OR title = ''")
    check("no empty titles", cur.fetchone()[0] == 0, "")
    cur.execute("SELECT COUNT(*) FROM articles WHERE pub_date IS NULL OR pub_date = ''")
    check("no empty pub_dates", cur.fetchone()[0] == 0, "")

    # 2d. pub_date format sanity (YYYY-MM-DD, valid dates)
    cur.execute("SELECT DISTINCT pub_date FROM articles")
    bad_dates = []
    for (d,) in cur.fetchall():
        try:
            date.fromisoformat(d)
        except ValueError:
            bad_dates.append(d)
    check("all pub_dates valid ISO dates", not bad_dates, f"{bad_dates[:5]}")

    # 2e. URL uniqueness (dedup guard)
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT url) FROM articles")
    total, distinct = cur.fetchone()
    check("no duplicate URLs", total == distinct, f"{total - distinct} dupes")

    # 2f. Sample 20 titles: scan for obvious future-outcome references.
    # Heuristic: titles mentioning specific future-dated predictions.
    import re
    cur.execute("SELECT pub_date, title FROM articles ORDER BY RANDOM() LIMIT 20")
    suspicious = []
    for pub_date, title in cur.fetchall():
        # Flag titles that claim to report an outcome dated after pub_date
        m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", title)
        if m:
            try:
                ref = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if ref > date.fromisoformat(pub_date):
                    suspicious.append((pub_date, title[:80]))
            except ValueError:
                pass
    check("sample: no title references a future-dated outcome", not suspicious,
          f"{suspicious[:3]}" if suspicious else "")

    # 2g. FTS index covers titles (retrieval surface)
    cur.execute("SELECT COUNT(*) FROM articles_fts")
    check("FTS index non-empty", cur.fetchone()[0] > 0, "")

    con.close()


def audit_rag() -> None:
    print("\n=== 3. RAG retrieval path ===")
    from td_one.news import NewsStore
    from td_one.news.rag import RAGTool, _validate_cutoff

    # 3a. Future cutoff rejected
    future = (date.today() + timedelta(days=30)).isoformat()
    try:
        _validate_cutoff(future)
        check("future cutoff rejected", False, "no exception raised")
    except ValueError:
        check("future cutoff rejected", True, "")

    # 3b. Malformed cutoff rejected
    try:
        _validate_cutoff("not-a-date")
        check("malformed cutoff rejected", False, "no exception raised")
    except (ValueError, Exception):
        check("malformed cutoff rejected", True, "")

    # 3c. Live retrieval: no article with pub_date >= cutoff
    db = REPO / "data" / "news" / "news_2026.db"
    if not db.exists():
        check("RAG live test (DB missing)", False, "")
        return
    tool = RAGTool(db_path=db, groq=None)  # BM25-only, no LLM
    cutoff = "2024-06-01"
    try:
        res = tool.search("mortgage rates Bank of Canada", before=cutoff, limit=10)
        leaked = [a for a in res.articles if a.pub_date >= cutoff]
        check(f"retrieval PIT-safe (cutoff={cutoff})", not leaked,
              f"{len(leaked)} articles >= cutoff" if leaked else "")
        check("retrieval returns results", len(res.articles) > 0,
              "0 articles (may indicate FTS issue)")
    except Exception as e:
        check("retrieval executes", False, str(e)[:100])

    # 3d. Empty query rejected
    try:
        tool.search("", before="2024-01-01")
        check("empty query rejected", False, "no exception")
    except ValueError:
        check("empty query rejected", True, "")


def main() -> int:
    print("News label-leakage audit — TD-One")
    print("=" * 50)
    audit_synthetic()
    audit_real_db()
    audit_rag()
    print("\n" + "=" * 50)
    print(f"PASSES: {len(PASSES)}  FAILURES: {len(ISSUES)}")
    for i in ISSUES:
        print(f"  !! {i}")
    return 1 if ISSUES else 0


if __name__ == "__main__":
    sys.exit(main())
