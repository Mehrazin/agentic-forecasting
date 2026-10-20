# Loan Attrition: Grouped Forecasting + TSFM Red-Teaming — Build Plan

**Status:** Track 1 built (2026-10-19); exp66 honest negative on CRPS.
**Reframed 2026-10-19:** grouping is for *explainability*, not for beating
CRPS. exp66b reruns with full v2 architecture to measure the explainability
gain vs CRPS cost. Kill criterion is OFF — report cost/benefit honestly.

## exp66 verdict (2026-10-19)

❌ **Honest negative on CRPS.** LLM-discovered groups (CRPS 0.01705) did not
beat random groups (0.00690); global (0.00227) won. **But:** the test was
thin (140 episodes, stub forecaster). Reframed: the question is whether
LLM groups are *explainable* without degrading forecast quality vs
deterministic grouping. See `experiments/exp66_report.md` and exp66b below.

## exp66b: grouping for explainability (in progress)

Full v2 architecture per group. Prompt playground (base / +hints / +news /
+hints+news). Group-count sweep 3/5/8/12. Ensemble arm (LLM selects grouping
per window). Metrics: CRPS (parity check vs hand-written), explanation
quality (citation rate, rule readability), risk-bin calibration.
**No kill criterion** — report "costs X CRPS for Y explainability gain."

### Pivot (from exp66b): deterministic grouping + LLM explanations

LLM-proposed groupings are not reproducible run-to-run — fatal for
explainability. **exp66c** implements the pivot:
- `src/td_one/agent/deterministic_grouper.py` — KMeans (fixed seed) +
  tree-surrogate rules, all passing `validate_rule()`. Same input →
  identical groups (tested).
- LLM keeps the explanation role via citation-enforced `risk_reporter.py`.
- Results: 10 reproducible groups (2–4 conditions each), **10/10 citation
  pass** with a total citation contract; grouping costs CRPS vs global
  (+0.009–0.016) and vs hand-written (+0.006–0.010), beats random by ~40%.
- **Honest negative inside the win:** static-attribute risk bins do not
  calibrate — LOW-bin groups realized *higher* test attrition (0.128)
  than HIGH-bin (0.087). Do not ship static risk bins as predictions.
- See `experiments/exp66c_report.md`.

### exp66d: cohort recovery + LLM risk classifier (exp66c follow-up)

- `src/td_one/data_gen/cohort_labels.py` — pure-function recomputation
  of the simulator's planted cohorts (`low_rate`, `osfi_exempt`,
  `family_squeeze`, `control`).
- `src/td_one/evaluation/cohort_recovery.py` — ARI + per-cohort
  precision/recall of discovered groups vs planted cohorts.
- `src/td_one/agent/llm_risk_classifier.py` — LLM as at-risk classifier
  (HIGH/MEDIUM/LOW + justification), given group stats + PIT-safe news.
- **Recovery honest negative:** ARI 0.01 — the grouper finds real,
  reproducible segments (rate_type ARI 0.24) but not the DGP's causal
  joints. Explanations are true descriptions of the wrong segments for
  forecasting; still useful as segment narratives.
- **LLM classifier directionally better:** monotonic HIGH > MEDIUM > LOW
  on test vs inverted bins — but n=10 groups, info-asymmetric (LLM had
  news). Worth a controlled follow-up before stronger claims.
- **Showcase:** `docs/GROUPING_EXAMPLES.md` — 4 worked examples
  (rule → members → forecast → risk → explanation), including the
  failure case (g3: both reads LOW, test 0.167).
- See `experiments/exp66d_report.md`.

## Direction

Two tracks, eventually combined:

- **Track 1 (exp66):** LLM proposes borrower groups from metadata (+ news),
  group-level attrition forecasting, LLM risk reports with grounded
  explanations. Forecasting is part of the agent's toolkit.
- **Track 2 (exp67/68):** Red-team Chronos-2/TimesFM-2.5 on loan-domain
  distributional shifts; build an agentic guard; evaluate on held-out shifts.
- **Integration (exp70):** Track 2's shift scenarios stress-test Track 1's
  group forecaster. S3 (regional divergence) is the meeting point.

## Track 1 — LLM Grouping + Group-Level Forecasting (exp66)

### Data
- **Simulator** (borrower-level, rich metadata): province, age, household
  size, children, employment, income (lognormal), liquid deposits, debt
  payments, rate type, mortgage rate, term, origination date, property
  value, insured flag, occupancy, credit score band, utilization, channel,
  advisor flag, life events, prior refinances (burnout).
- **SMM: NOT usable** — aggregates only (63 rows). Track 1 is synthetic-only.
- **News:** 17,738 articles, PIT-safe accessors. Grouper reads news because
  some news is demographic-specific (e.g., "Ontario first-time buyers").

### Pipeline
1. **`llm_grouper.py`** — LLM sees: attribute distributions (histograms, not
   raw rows), training-period attrition rates by attribute, demographic-
   tagged news summaries (via RAG, see below). Outputs strict JSON:
   executable `pandas.DataFrame.query()` rules + one-sentence rationales.
   Validator rejects bad queries, outcome-column references, tiny groups,
   excessive overlap. **PIT rule:** zero post-origin info in the prompt.
2. **`group_forecaster.py`** — per-group `ProposedForecasterV2`; groups
   under `min_group_n` fall back to global (shrinkage).
3. **`risk_reporter.py`** — deterministic risk bins (LOW/MEDIUM/HIGH from
   training tertiles, widened by conformal intervals); LLM explanations
   with **citation enforcement** — every factual sentence cites a `[stat]`
   or `[url]`, post-hoc checker rejects uncited claims.

### News wiring (added 2026-10-19)
- **Live Perplexity API** for demographic-specific news fetch (user approved
  credit increase). **Cache everything** — every fetch enriches the SQLite
  cache; never re-fetch.
- **RAG upgrade:** current RAG is BM25 → Groq rerank. Upgrade to
  **semantic retrieval + rerank** using local embeddings
  (all-MiniLM-L6-v2, 384-dim — Vector proxy returns 403
  model_access_denied for embeddings, verified 2026-10-19).
  Pipeline: embed query → cosine retrieval over cached articles →
  LLM rerank → top-k to grouper/reporter.
- **Top-k tuning** (2026-10-19, `scripts/tune_rag_k.py`, 20 hand-labeled
  pairs): k=10 maximizes F1 (0.468), but retrieval uses k=50 biasing
  toward recall (0.818) since the reranker filters cheaply; final k=5
  for precision. See `src/td_one/rag/semantic.py` docstring.
- News is PIT-gated via `articles_visible_asof(origin)` everywhere.
- **Article bodies (2026-10-19):** the news DB had titles only — no bodies
  to chunk. Built `scripts/fetch_article_bodies.py` (trafilatura extraction,
  per-host throttle, `article_bodies` table keyed by URL). ~86% fetch success
  on the 500-article validation sample.
- **Chunking ablation** (`src/td_one/rag/chunking.py`,
  `scripts/tune_chunking_v2.py`): title-as-query MRR on 30 bodies —
  fixed-512 (MRR 0.731, 3.5 chunks/article) ≈ paragraph (MRR 0.715).
  **Fixed wins** (simpler, deterministic). Sentence chunking too fragmented
  (13/article); semantic chunking too slow on CPU. Chunk embeddings cached
  to disk keyed by chunk-text hash — never recomputed.
- **Full-vs-chunk ablation** (`scripts/ablate_fast.py`): chunk-only context
  matches full-article keyword coverage (0.818 vs 0.824) at **1.9x fewer
  characters**. **Verdict: pass chunks, not full articles**, to the LLM.
- **Threading fix:** torch/sentence-transformers deadlocks without
  `OMP_NUM_THREADS=1` etc. — set in `EmbeddingClient` and all RAG scripts.

### Evaluation (4-arm ablation)
| Arm | Description |
|---|---|
| A: Global | No grouping |
| B: Random | Same group count, random assignment |
| C: Hand-written | Planted simulator cohorts |
| D: LLM-discovered | This pipeline |

**No kill criterion** (reframed 2026-10-19): grouping is for explainability.
Report CRPS cost vs explainability gain honestly. If the CRPS cost is large,
say so — but don't kill grouping for losing on CRPS alone.

### Build order
1. `src/td_one/agent/grouping.py` — `GroupDefinition`, `validate_rule()`,
   `apply_groups()`, overlap resolution → `tests/test_grouping.py`
2. `src/td_one/rag/semantic.py` — embedding client (Vector proxy),
   semantic retriever, reranker → `tests/test_semantic_rag.py`
3. `src/td_one/agent/llm_grouper.py` — prompt builder (+ news summaries),
   JSON parsing, validation loop → `tests/test_llm_grouper.py` (mock)
4. `src/td_one/agent/group_forecaster.py` — per-group v2, fallback →
   `tests/test_group_forecaster.py`
5. `src/td_one/agent/risk_reporter.py` — bins, news retrieval, citation
   checker → `tests/test_risk_reporter.py`
6. `experiments/exp66_grouped_attrition.py` + `exp66_report.md`
7. News fetcher: `scripts/fetch_demographic_news.py` (Perplexity, cached)

Closes Argon Finding 3 (`discover_candidate_groups()`).

## Track 2 — TSFM Red-Teaming (exp67) + Agentic Guard (exp68)

### Attack surface
Chronos-2 and TimesFM-2.5 are strictly univariate: forecast = f(history).
Shifts invisible in the history window but visible in side info (policy
dates, rate levels, borrower mix) are structural blind spots.

### Shift taxonomy (loan domain)
| # | Shift | Why TSFM fails |
|---|---|---|
| S1 | Rate-shock regime | History is pre-shock |
| S2 | Policy intervention (e.g., stress-test change) | Jump predictable from announcement, invisible univariately |
| S3 | Regional divergence | Aggregate averages the move away |
| S4 | Seasonal break | Learned old seasonal pattern |
| S5 | Correlation breakdown (lock-in vs refi) | No covariate access |
| S6 | Volatility regime | Pre-shift spreads too narrow |
| S7 | Composition drift (burnout) | Drift misread as trend |

### Protocol
- Rolling origins across shift onset T (pre / straddling / post).
- Metrics: CRPS on common 19-level grid, 80% coverage, PIT histograms.
- Paired t-test + win rate vs model's own pre-shift baseline AND vs
  classical baselines on identical shifted data.
- Placebo control on no-shift DGP. Severity ladder (0.5σ–4σ) → breaking
  point curves, not binary pass/fail.

### Agentic guard (exp68)
1. Shift detection from covariates (CUSUM on policy rates,
   `events_visible_asof`) — fires before the target moves.
2. Context surgery: truncate TSFM input to post-shift data / switch experts.
3. Shift-aware quantile widening from past shift episodes' residuals.
- LLM classifies shift type from metadata/news → selects response template.
  (LLM outputs structure, never numbers.)
- **Holdout:** dev on S1/S4/S6; eval on S2/S3/S5/S7. Disjoint seeds,
  different magnitude ranges, pre-registered.

### Build order
1. `src/td_one/data_gen/shift_scenarios.py` — `ShiftSpec`,
   `apply_shift()`, seasonal-break + coefficient-schedule hooks
2. `src/td_one/data_gen/group_series.py` — histories → monthly per-group
   series (**shared with Track 1**)
3. `experiments/exp67_tsfm_redteam.py` + report
4. `src/td_one/agent/shift_guard.py` — detector + policies
5. `experiments/exp68_agent_vs_shift.py` + report (pre-registered holdout)

**Order is load-bearing:** red-team numbers before the guard. No guard
without failure evidence.

### exp67 outcome (2026-10-20) — Phase A COMPLETE, Phase B unblocked

- Grid finished: 7 shifts × [1σ, 4σ] × 2 seeds × 9 origins; Chronos-2 /
  TimesFM-2.5 (real weights) vs naive / M0; placebo-controlled; PIT gate
  passes (pre-shift degradation +0% in all 28 cells).
- All models break together at 4σ on S1 (+185–228%), S2 (+65–79%),
  S3 (+30–59%) — TSFMs have no *relative* robustness edge, but Chronos-2 is
  best in *absolute* CRPS on 6/8 shifts. S4/S5 never break (honest negatives).
  Coverage collapses under level shifts (M0 0.61→0.08 at S1/4σ).
### exp68 outcome (2026-10-20) — Phase B COMPLETE: honest negative

- Guard (`src/td_one/agent/shift_guard.py`): CUSUM on policy-rate covariate
  (2-month lead) + PIT-filtered announcements → context truncation +
  conformal widening (dev-calibrated). 18/18 guard tests green.
- **Claim FAILED:** guard+TSFM (Chronos-2) vs TSFM-alone on holdout
  (S2/S3/S5/S7, 2–4σ, seeds 100–101): +0.5% rel., p=0.824, win rate 14.6%.
  Do-no-harm gates pass.
- **Lesson:** context truncation hurts — Chronos-2 adapts better with full
  (contaminated) context than with 9 post-shift points; correct detection
  (S2: 12/12) gained nothing. Per pre-registration: no iteration.
- Full report: `experiments/exp68_report.md`.

### exp69 outcome (2026-10-20) — anticipatory LLM adjustment: honest negative

- Adjuster (`src/td_one/agent/anticipatory_adjuster.py`): LLM reads PIT-safe
  news → structure-only signal (type/direction/bucket/timing); calibrated
  (type, bucket) → level-shift from past episodes (seeds 200–202) applied to
  Chronos-2 quantiles on straddle origins. 18/18 tests green.
- **Claim FAILED:** B vs A on straddle origins (S1/S2, 2–4σ, seeds 300–302):
  +10.2% rel., p=0.231, win rate 36%. **Oracle ceiling +11.4% (p=0.214)** —
  even perfect anticipation doesn't significantly help; the calibrated median
  shift is blunt vs per-episode noise. LLM ≈ oracle (B vs C, p=0.73).
- Works only at S1/4σ (B +30%, C +35%); hurts at 2σ and on S2. Do-no-harm
  gates pass exactly (no hallucination, no post-onset firing).
- Per pre-registration: no iteration. Full report: `experiments/exp69_report.md`.

## Integration (exp70)
Track 2's shift scenarios stress-test Track 1's group forecaster.
S3 (regional divergence) is the natural meeting point: Track 1 groups the
entities, Track 2 shifts the groups.

## Standing constraints
- PIT safety is a gate (no `observed_asof > origin` anywhere).
- Every fetcher hit enriches the cache; commit cache snapshots.
- Perplexity: user-approved budget, resume-safe, cached.
- TDD + regression test per fix; ruff clean; docs with every change.
- Honest negatives with numbers, not softened.
