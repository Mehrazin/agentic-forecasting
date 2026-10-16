# TD-One Data Engineering Audit

**Date:** 2026-10-16
**Scope:** Full data pipeline — Freddie Mac ingest, FRED, news KB, PIT store, simulator
**Method:** Code review + live DB inspection + manifest/parquet verification
**Status:** All P0s and P1s fixed (commit 9a5665f). P2s documented for follow-up.

## Summary

| Severity | Count | Definition | Status |
|----------|-------|------------|--------|
| P0 | 2 | Data leakage / corruption — wrong results | ✅ Fixed |
| P1 | 5 | Robustness — silent failures, wrong metadata | ✅ Fixed |
| P2 | 5 | Docs / performance | 📝 Documented |

---

## P0 — Data leakage / corruption

### P0-1: FRED z-score normalization leaks future data into every forecast origin

**Files:**
- `src/td_one/models/macro_m0.py:88-91` (`load_fred_z`)
- `src/td_one/data_gen/simulator_v2.py:160-163` (`load_fred_macro_z`)

**Problem:**
```python
z = (mom3 - float(mom3.mean())) / std   # macro_m0.py
z = (mom3 - mom3.mean()) / std           # simulator_v2.py
```
The mean and standard deviation are computed over the **entire FRED history** (e.g., DCOILWTICO 1986–2026). When `MacroM0` makes a forecast at a 2020 origin, the z-score features embed distributional information from 2021–2026. This is a point-in-time violation: the normalization statistics include future data.

**Impact:** Every `MacroM0` evaluation (exp8, exp10, exp13, exp15) used z-scores normalized with future data. The magnitude of the bias depends on how much the series mean/std shifted, but the direction is unknowable without rerunning. For a project with explicit PIT discipline, this undermines the macro results.

**Fix:** Add an `asof: date | None` parameter to `load_fred_z`. When provided, compute `mom3.mean()` and `mom3.std()` using only observations with `date <= asof`. Alternatively, use expanding-window z-scores (each month normalized by history up to that month). The simulator's usage is a DGP design choice rather than evaluation leakage, but should use the same function for consistency.

---

### P0-2: FRED parquet cache is latest-vintage, not as-of

**File:** `scripts/fetch_fred.py` (`fetch_all`, via `FredClient.observations`)

**Problem:** The bulk downloader fetches without `vintage_dates`, so FRED returns the **latest vintage** for every observation. Revised series (Canada CPI `CPALTT01CAM659N`, unemployment `LRHUTTTTCAM156S`, Case-Shiller `CSUSHPISA`, etc.) contain values revised months or years after the observation date — values that were not known at the time.

The `FredClient.observations` docstring (`src/td_one/fred/live.py:157-164`) explicitly warns:
> *"An observation-date cutoff alone is NOT vintage-correct."*

But the downloader — the path every model actually uses — ignores this entirely.

**Impact:** Any model joining FRED data on calendar date (MacroM0, simulator v2 macro overlay) trains on revised values. For CPI/unemployment, revisions are typically small, but for GDP-adjacent series they can be material. Combined with P0-1, the FRED feature pipeline has two independent look-ahead paths.

**Fix (choose one):**
1. Pass `vintage_dates` in the downloader to pull as-of vintages (correct but slower; FRED rate-limits vintage queries more aggressively).
2. Document as a known limitation in `docs/METHODOLOGY.md` with a quantified bound (compare latest-vintage vs first-vintage for the most revision-sensitive series).
3. At minimum, add the warning to `fetch_fred.py`'s module docstring — currently it claims "PIT discipline" via date filtering, which is misleading for revised series.

---

## P1 — Robustness

### P1-1: Manifest overwritten on partial fetches — 19 files, 2 recorded

**File:** `scripts/fetch_fred.py:fetch_all` (manifest write at end)

**Problem:** `fetch_all` builds the manifest dict from **only the series in the current run**, then overwrites `fred_manifest.json`. Verified live:
- `data/fred/*.parquet`: **19 files**
- `fred_manifest.json`: **2 entries** (DCOILWTICO, HDTGPDCAQ163N — the last `--series` run)

**Impact:** The manifest is the only record of `last_updated` stamps, frequencies, and units. After any partial fetch, metadata for the other 17 series is lost. Any tooling that reads the manifest (release-awareness, staleness checks) silently operates on incomplete data.

**Fix:** Load the existing manifest at the start of `fetch_all`, update entries for fetched series, preserve others. Write atomically (temp file + rename).

---

### P1-2: Future-dated article in news KB

**File:** `data/news/news_2026.db` (data); `scripts/fetch_news_perplexity.py:store`

**Problem:** Article dated **2026-10-28** present in the DB (Bank of Canada rate announcement). The backfill ran on 2026-10-16. Perplexity either hallucinated the date or returned a scheduled future event. The `store()` function filters by month prefix (`want_prefix`) but never rejects `pub_date > today`.

**Impact:** A single future-dated article breaks the PIT guarantee for any query with a cutoff between 2026-10-16 and 2026-10-28. The `NewsStore.search(before=...)` would return it for `before="2026-10-20"`, leaking future information.

**Fix:** In both `store()` functions (GDELT and Perplexity scripts), reject articles with `pub_date > date.today().isoformat()`. Log rejections. Also add a DB-level CHECK constraint or a periodic audit query.

---

### P1-3: 2,310 of 2,375 news articles have empty `source`

**File:** `scripts/fetch_news_gdelt.py:store` (line: `a.get("source", "")`)

**Problem:** The GDELT DOC API returns the outlet in a field that is not `"source"` (it's typically nested or named differently). Result: 2,310 articles have `source = ''`. Only the 65 Perplexity articles have `source = 'perplexity'`.

**Impact:** Cannot filter by outlet credibility, cannot distinguish wire services from blogs, cannot do source-diversity checks in RAG retrieval. The `source` column is effectively dead for 97% of the KB.

**Fix:** Inspect the actual GDELT `artlist` response schema and map the correct field (likely `source` is nested under a different key or the API returns `domain`). Backfill the column for existing rows.

---

### P1-4: GDELT `store()` new-article count is inflated

**File:** `scripts/fetch_news_gdelt.py:store`

**Problem:**
```python
con.execute("INSERT OR IGNORE ...")
if con.total_changes:   # BUG: cumulative, not per-row
    n += 1
```
`connection.total_changes` is cumulative since the connection opened. After the first successful insert, this is always truthy, so `n` increments for **every** article including `INSERT OR IGNORE` skips. The "N new" log output is wrong (the Perplexity script correctly uses `cur.rowcount`).

**Impact:** Operational only (misleading logs), but it masked the true dedup rate during the backfill. The reported "2,332 new" is inflated; actual new rows were fewer.

**Fix:** Use `cur = con.execute(...); n += cur.rowcount` (matching `fetch_news_perplexity.py`).

---

### P1-5: No Freddie Mac ingest pipeline exists

**Files:** `data/freddie/raw/orig_2026Q1.txt` (143,965 rows), `data/freddie/raw/perf_2026Q1.txt` (203,412 rows)

**Problem:** Raw files are present but there is no loader, no schema validation, no column mapping, no cohort builder. The files are pipe-delimited with **no header row** (positional columns). The Freddie Mac file layout (orig: 31 columns, perf: 30 columns) must be hardcoded from the data dictionary.

**What's needed** (`src/td_one/freddie/`):
1. `layout.py` — column name mappings for orig + perf files (from the Freddie Mac Single-Family Loan-Level Dataset data dictionary).
2. `loader.py` — streaming reader (files are 18MB/23MB now but full vintages are GBs); dtype coercion; reject malformed rows with a quarantine count.
3. `events.py` — derive prepayment events (zero-balance code `01`), censoring (codes `02`, `03`, `09`), delinquency timelines.
4. `cohort.py` — build vintage × FICO × note-rate cohort panels with monthly SMM/CPR.
5. Wire into `PITStore` or a Freddie-specific query layer with `asof` semantics.

**Impact:** The single most valuable real dataset is currently inert. This blocks the entire "real mortgage benchmark" workstream.

---

## P2 — Documentation / performance

### P2-1: No index on `articles.pub_date`

**File:** `data/news/news_2026.db` (schema in `src/td_one/news/__init__.py:init_db`)

Every PIT query (`WHERE pub_date < ?`) does a full table scan. Only index is the auto-index on `url` (UNIQUE). At 2,375 rows this is fine; at 100k+ rows (multi-year KB) it will degrade.

**Fix:** `CREATE INDEX idx_articles_pub_date ON articles(pub_date)` in `init_db`.

---

### P2-2: FTS index covers title only

**File:** `src/td_one/news/__init__.py:init_db` (`articles_fts USING fts5(title, ...)`)

There is no snippet/content column in the `articles` table. The `snippet()` in `NewsStore.search` generates context around the match from the title only. For RAG retrieval, title-only search misses articles where the relevant information is in the body.

**Fix:** Add a `snippet TEXT` column populated at ingest time (GDELT returns `seendate` but not body text; Perplexity returns snippets — store them). Include it in the FTS index: `USING fts5(title, snippet, ...)`.

---

### P2-3: Dead code in GDELT month-end calculation

**File:** `scripts/fetch_news_gdelt.py:fetch_month`

```python
last_day = (date(year, month + 1, 1) - date(year, month, 1)).days  # dead
import calendar  # inline import
last_day = calendar.monthrange(year, month)[1]  # overwrites
```
Also fails for `month=12` in the first line (`month + 1 = 13`). Harmless because overwritten, but confusing.

**Fix:** Remove the first computation and hoist `import calendar` to top level.

---

### P2-4: `fetch_fred.py` docstring overclaims PIT discipline

**File:** `scripts/fetch_fred.py` (module docstring)

> *"PIT discipline: files hold full history; filter `date <= run_date` at use time"*

This is misleading given P0-2 (latest-vintage revisions). Date filtering is necessary but not sufficient for PIT correctness on revised series.

**Fix:** Amend to acknowledge the vintage limitation and point to `FredClient.observations(vintage_dates=...)`.

---

### P2-5: PITStore doesn't cover real data pipelines

**File:** `src/td_one/data_service/pit_store.py`

The PIT store has correct temporal guards (`record_asof`, `outcome_observed_at`) but only covers **synthetic** mortgage data via `scripts/scale_up.py`. The FRED, news, and (future) Freddie Mac pipelines each implement their own ad-hoc PIT logic (or none). There is no unified "give me everything visible at date X" interface.

**Fix (long-term):** Either extend PITStore to register FRED/news/Freddie Mac sources with their respective as-of semantics, or document that PIT is enforced per-pipeline and list the guarantees of each.

---

## Timeline alignment check

| Source | Coverage | Granularity | 2026 alignment |
|--------|----------|-------------|----------------|
| FRED (19 series) | 1986–2026 (varies) | Daily / monthly / quarterly | ✅ Covers 2026 |
| News KB (2,375) | 2026-01-07 – 2026-10-28 | Daily | ✅ Covers 2026 (but 1 future-dated article — P1-2) |
| Freddie Mac 2026Q1 | Originations Feb–Apr 2026; perf through Mar 2026 | Monthly | ✅ 2026, but **very fresh** — only 1–2 months seasoning, limited prepayment events observable |
| Simulator v1/v2 | 2021-04-30 + 66 months → ~2026-10 | Monthly | ✅ Aligns with 2026 |

**Granularity gaps:** FRED mixes daily/monthly/quarterly; news is daily; Freddie Mac and simulator are monthly. The `ME` resampling in `load_fred_z` handles frequency conversion, but the z-score PIT issue (P0-1) remains regardless of granularity.

**No timezone issues found** — all dates are calendar dates without times. FRED uses US Eastern observation dates; news uses UTC dates from GDELT/Perplexity. For daily granularity this could cause off-by-one at month boundaries, but all consumption is monthly, so the impact is negligible.

**Freddie Mac seasoning caveat:** 2026Q1 originations have at most 2 months of performance history (through 2026-03). Voluntary prepayment (zero-balance code `01`) is rare in the first months of a loan's life. The cohort panels will be thin on events until more vintages arrive. This is a data limitation, not a pipeline bug, but it constrains what the "real mortgage benchmark" can demonstrate right now.

---

## Recommended fix order

1. **P0-1** (z-score PIT) — invalidates macro evaluation results; fix before citing any MacroM0 numbers
2. **P1-5** (Freddie Mac pipeline) — unblocks the highest-value workstream
3. **P0-2** (FRED vintage) — at minimum, document the limitation
4. **P1-2** (future article) — one-line fix, restores PIT guarantee
5. **P1-1** (manifest) — prevents silent metadata loss
6. Remaining P1s and P2s in order
