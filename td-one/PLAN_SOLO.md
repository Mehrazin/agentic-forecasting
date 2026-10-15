# TD-One Plan (Solo Revision)

**Author**: Vandad Davoodnia (solo)
**Date**: 2026-10-15
**Status**: Revised from team plan for solo YOLO development
**Canonical note**: This file supersedes the root `plan.md` (original team
spec, retained for reference) for all solo execution.

## Changes from team plan

| Aspect | Team plan | Solo revision | Rationale |
|--------|-----------|---------------|-----------|
| Team | 5 people, 5 workstreams | Solo | Vandad solo |
| Scope | M0-M4 + stretch (ADAS/ALMA) | M0-M3 (MVP1) | M4 needs multi-persona + meta-search; defer |
| LLM | Not specified | Groq `openai/gpt-oss-120b` | Available, tested, works |
| FRED | Live API | Mocked file cache | Vandad: "mock it for development" |
| Evidence | Google ADK + Perplexity | Frozen corpus only | No API keys; date-guarded local docs |
| Timeline | Build Days 1-5 | Flexible, TDD-driven | Solo pace |

## Architecture (unchanged)

- **Problem**: Forecast P(switch | renewal) for mortgage segments
- **Data**: Synthetic 10k customers × 66 months (60 train + 6 OOT)
- **PIT**: All queries enforce `record_asof <= run_date`; leakage auditor verifies 0%
- **Tools**: data_segment, evidence, forecasting (same 3-tool design; not yet implemented)
- **Models**: M0 ✅ → M1 ✅ (Chronos-2/TimesFM-2.5, real weights) → M2 ❌ (LLM) → M3 ❌ (hybrid agent)
- **Readout**: Beta-Binomial for count coherence: A ~ BetaBinomial(N, α, β)

Ladder status: M0 ✅ done, M1 ✅ done, M2 ❌ planned, M3 ❌ planned
(hybrid math exists in `models/hybrid.py`; no agent drives it yet).

## Build order (TDD)

1. **Schemas** (from Excel): Pydantic models for mortgage history + renewal
   outcomes ✅ (renewal offers + propensity scores: not yet modeled)
2. **Synthetic DGP**: mortgage_simulator with planted shocks (rate, OSFI, family) ✅
   (structural outcome DGP + observation lag added in the 2026-10-15 audit)
3. **PIT store**: DuckDB with `record_asof` filtering ✅ (history + outcomes tables)
4. **M0 baseline**: Historical rate + Jeffreys Beta-Binomial ✅ (trailing-6)
5. **TSFM wrappers**: Chronos-2, TimesFM-2.5 (per survey results) ✅ (real weights)
6. **Hybrid readout**: TSFM + LLM log-odds adjustment (bounded ±0.75) ✅ (math only)
7. **Agent controller**: ReAct loop with Groq ❌ (helpers only: evidence-prompt
   builder + adjustment estimator; orchestration loop not yet implemented)
8. **Evaluation**: Metrics (MAE/CRPS/Brier/coverage) ✅, leakage auditor ✅,
   backtest runner ❌ (not yet implemented)
9. **Experiments**: exp1 (M0 vs M1 mocks) ✅, exp2 (M0 vs M1 real weights,
   3-scenario redesign) ✅, exp3 (M2/M3 ablation) ❌ planned

## Enterprise controls (from AI2)

- Full typing (Pydantic, no `Any`)
- Ruff + mypy strict + pytest
- TDD: test first, then implement
- Regression test for every bug
- Docs updated with every change (per AGENTS.md)

## Deferred

- M4 (silicon crowd, ALMA memory): needs more time
- Live evidence retrieval: no API keys
- Real FRED: mocked

## Deviations from the TSFM survey (recorded 2026-10-15)

- Survey (`docs/research/survey-tsfm.md`) recommended the Darts unified API;
  built custom wrappers on raw `chronos-forecasting` + `timesfm` instead.
  Rationale: the evaluation needs the native quantile grids directly
  (CRPS/coverage on the true levels), which Darts abstracts away.
