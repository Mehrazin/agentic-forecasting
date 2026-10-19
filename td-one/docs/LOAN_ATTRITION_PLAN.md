# Loan Attrition: Grouped Forecasting + TSFM Red-Teaming — Build Plan

**Status:** planning (2026-10-19). Tracks the implementation of the two new
loan-domain research directions. Updated 2026-10-19 with news-aware grouping
and semantic RAG.

## Direction

Two tracks, eventually combined:

- **Track 1 (exp66):** LLM proposes borrower groups from metadata (+ news),
  group-level attrition forecasting, LLM risk reports with grounded
  explanations. Forecasting is part of the agent's toolkit.
- **Track 2 (exp67/68):** Red-team Chronos-2/TimesFM-2.5 on loan-domain
  distributional shifts; build an agentic guard; evaluate on held-out shifts.
- **Integration (exp69):** Track 2's shift scenarios stress-test Track 1's
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
  **semantic retrieval + rerank** using Vector proxy embeddings
  (`text-embedding-3-small`, verified available 2026-10-19 via /v1/models).
  Pipeline: embed query → cosine retrieval over cached articles →
  cross-encoder/LLM rerank → top-k to grouper/reporter.
- News is PIT-gated via `articles_visible_asof(origin)` everywhere.

### Evaluation (4-arm ablation)
| Arm | Description |
|---|---|
| A: Global | No grouping |
| B: Random | Same group count, random assignment |
| C: Hand-written | Planted simulator cohorts |
| D: LLM-discovered | This pipeline |

**Kill criterion (pre-registered):** if D doesn't beat B significantly
(paired t-test on member-weighted group CRPS), the LLM grouping is dead —
honest negative, no prompt iteration.

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

## Integration (exp69)
Track 2's shift scenarios stress-test Track 1's group forecaster.
S3 (regional divergence) is the natural meeting point: Track 1 groups the
entities, Track 2 shifts the groups.

## Standing constraints
- PIT safety is a gate (no `observed_asof > origin` anywhere).
- Every fetcher hit enriches the cache; commit cache snapshots.
- Perplexity: user-approved budget, resume-safe, cached.
- TDD + regression test per fix; ruff clean; docs with every change.
- Honest negatives with numbers, not softened.
