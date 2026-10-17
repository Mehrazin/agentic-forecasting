# News Label-Leakage Audit Report

**Date:** 2026-10-17
**Auditor:** subagent (data auditor)
**Script:** `td-one/audits/news_leakage_audit.py` (33 checks, 32 pass, 1 data-quality note)
**Verdict: NO LABEL LEAKAGE FOUND.** One minor data-quality issue (7 empty titles, not leakage).

---

## 1. Synthetic news generator (`news_synthesis.py`)

**Result: CLEAN** — 21/21 checks pass across 5 seeds.

| Check | Result |
|---|---|
| `article_date >= event_date` (all articles, 5 seeds) | ✅ PASS |
| `strip_eval_fields()` removes `event_id` | ✅ PASS |
| `articles_visible_asof()` PIT-safe | ✅ PASS |
| No hardcoded dates in headline/body text | ✅ PASS |
| `NEWS_LEAD_MONTHS` has ≥3 leading types | ✅ PASS (3 leading: POLICY_SHOCK +1.87, RATE_SHIFT +0.90, HOUSING_DISRUPTION +0.92) |

**Notes:**
- Templates never mention dates, only relative language ("coming months", "within weeks"). No temporal leakage vector in text.
- `event_type` is intentionally kept after stripping (a competent reader could infer it from text anyway — not a leak, just redundant signal).
- The `reported_magnitude` has ±30% noise; agents must triangulate. This is by design, not leakage.
- LENDER_SHIFT (−0.58) and MACRO_SURPRISE (−0.67) have negative news lead (lagging indicators). This is a *usefulness* problem (documented in exp21), not a *leakage* problem — the news arrives after the DGP effect, so it can't leak the future.

---

## 2. Real news DB (`data/news/news_2026.db`)

**Result: CLEAN** — 7/8 checks pass. 1 data-quality note (not leakage).

| Check | Result |
|---|---|
| 0 future-dated articles (max `pub_date` = 2026-10-16, today = 2026-10-17) | ✅ PASS |
| 2026-10-28 row from DATA_AUDIT P1-2 purged | ✅ PASS (0 rows) |
| No empty `pub_date` | ✅ PASS |
| All `pub_date` valid ISO dates | ✅ PASS |
| No duplicate URLs | ✅ PASS |
| Sample: no title references a future-dated outcome | ✅ PASS (20 sampled) |
| FTS index non-empty | ✅ PASS |
| No empty titles | ⚠️ NOTE — 7 articles (0.07%) have empty/whitespace titles |

**The 7 empty titles:** Scraping artifacts (e.g., `bigtakeover.com` interview, `lechodelaval.ca` pages). They contain no text, so they cannot leak anything. They also can't be retrieved meaningfully via FTS (empty title matches nothing). **Recommendation:** purge or leave; harmless either way. Not a leakage vector.

**Targeted leakage test (BoC 2022-07-13 surprise +100bp hike):**
- Articles 2022-07-01 → 2022-07-12 mentioning a hike: **0** ✅
- Articles 2022-07-13 → 2022-07-19 describing the hike: **5** ✅
- This is exactly what PIT-safe looks like: the DB covers the event, but only after it happened. A query with `before='2022-07-13'` can never see the outcome.

**Schema note:** The DB is title-only (no body/content column; FTS indexes title only). This limits the leakage surface — there's no article body that could contain a detailed future-dated narrative.

---

## 3. RAG retrieval path (`news/rag.py`, `news/__init__.py`)

**Result: CLEAN** — 5/5 checks pass.

| Check | Result |
|---|---|
| Future `before` cutoff rejected (`ValueError`) | ✅ PASS |
| Malformed cutoff rejected | ✅ PASS |
| Live retrieval: 0 articles with `pub_date >= cutoff` | ✅ PASS (cutoff=2024-06-01, 10 results) |
| Retrieval returns results (FTS works) | ✅ PASS |
| Empty query rejected | ✅ PASS |

**Defense in depth (3 layers):**
1. `_validate_cutoff()` refuses future/malformed dates before any DB access.
2. SQL `WHERE a.pub_date < ?` enforces the cutoff in the database (strict `<`, not `<=` — articles published *on* the cutoff date are excluded, which is the conservative PIT choice).
3. Belt-and-braces re-check in `RAGTool.search()`: `if a.pub_date < cutoff` filters again in Python.

---

## Summary

| Surface | Checks | Pass | Issues |
|---|---|---|---|
| Synthetic generator | 21 | 21 | 0 |
| Real news DB | 8 | 7 | 1 note (empty titles, not leakage) |
| RAG retrieval | 5 | 5 | 0 |
| **Total** | **34** | **33** | **0 leakage** |

**The news is clean.** Both the synthetic feed and the real DB are PIT-safe, the retrieval path has triple-layer cutoff enforcement, and the targeted BoC test confirms no pre-event outcome leakage.

**Recommended follow-ups (non-blocking):**
1. Purge the 7 empty-title rows (cosmetic; 0.07% of DB).
2. Consider adding a periodic re-audit to CI (the DB grows; the invariant should hold).
