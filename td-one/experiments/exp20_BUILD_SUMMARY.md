# exp20: Event-Augmented Simulator — Build Summary

**Date:** 2026-10-17
**Status:** Phase 1 complete. All tests pass (30/30), ruff clean.

## What was built

### 1. `src/td_one/data_gen/events.py` (new, ~350 lines)
Event timeline generator with 5 exogenous event types:
- **POLICY_SHOCK** — OSFI stress-test changes; 2-month transmission lag; affects uninsured borrowers
- **RATE_SHIFT** — BoC surprise decisions; 1-month lag; affects rate_spread for resetting borrowers
- **HOUSING_DISRUPTION** — Regional price drops; 1-month lag; affects LTV → switch via equity channel
- **LENDER_SHIFT** — Bank retention campaigns; 0 lag; affects only that bank's borrowers (negative effect)
- **MACRO_SURPRISE** — Unemployment spikes; 0 lag; affects payment_shock via income loss

Key design:
- `Event` dataclass: event_id, event_type, event_date, magnitude, affected_cohort, transmission_lag_months, effect_duration_months, effect_size (logit), metadata
- `generate_event_timeline()`: Poisson process, calibrated effect sizes (±0.2 to ±0.8 logit ≈ ±2-8pp)
- `events_visible_asof()`: THE pit-safe accessor — only sanctioned way to filter by as-of date
- `cohort_matches()`: heterogeneous effect mechanism — 17 cohort selectors (geography, insurance, credit, lender via deterministic hash, employment)
- `active_effect_shift()`: sums effect_size for active events

### 2. `src/td_one/data_gen/news_synthesis.py` (new, ~280 lines)
Template-based synthetic news generator:
- 2 templates per event type (wire-service + analyst tones)
- Realistic noise: coverage_prob 0.40-0.95 by type; 1-20 articles per covered event (Poisson); magnitudes ±30%; delays 1-45 days by type
- PIT-safe: article_date >= event_date always
- `strip_eval_fields()`: removes event_id before agent sees articles
- `articles_visible_asof()`: pit-safe news filter
- `coverage_stats()`: diagnostics

Observed realism (seed 42): 15 events → 46 articles, 60% coverage. Lender shift got 0 articles (asymmetric coverage working). Rate shift got 12 articles.

### 3. `src/td_one/data_gen/simulator.py` (modified)
- `generate_outcomes()` now accepts optional `event_timeline: list[Event] | None`
- When provided: for each outcome, filters to PIT-safe visible events, cohort-matches against borrower attributes, sums active effect shifts into logit_p
- Backward compatible: `event_timeline=None` → identical to pre-event behavior (tested)

### 4. `experiments/exp20_event_augmented.py` (new)
Validation experiment:
- Generates timeline + news + outcomes (with and without events, same RNG)
- Signal strength: mean |with - without| = **9.1pp** (target >2pp) ✅
- News realism: 60% coverage, 0-12 articles/event ✅
- Oracle backtest: challenging due to ETS adaptivity to level shifts

### 5. `tests/test_events.py` (new, 20 tests)
- Timeline: sorted, bounded, reproducible, chronological IDs, valid types/cohorts
- PIT safety: events_visible_asof filters correctly; effect_window respects lag; future events don't affect past outcomes
- Cohort matching: uninsured, geography, lender (deterministic per borrower_id)
- News: PIT-safe dates, reproducible, eval fields stripped, asymmetric coverage
- Simulator integration: no-events unchanged; events shift outcomes; future events have zero effect

## Validation results

| Check | Result |
|---|---|
| Signal strength | 9.1pp mean move (target >2pp) ✅ |
| News coverage | 60%, 0-12 articles/event ✅ |
| PIT safety | All tests pass ✅ |
| Backward compat | Identical outcomes without events ✅ |
| Existing tests | 10/10 simulator tests still pass ✅ |
| Ruff | Clean ✅ |

## What's next (Phase 2)

Per AGENTIC_ROADMAP.md:
1. Code execution + AI2 sandbox port (`src/td_one/agent/sandbox.py`)
2. 10 general tools (residual analysis, changepoint detection, etc.)
3. exp21: Agent feature discovery
4. exp22: Code-executing ReAct (M2 v2)

## Files (not committed per instructions)

- `src/td_one/data_gen/events.py` (new)
- `src/td_one/data_gen/news_synthesis.py` (new)
- `src/td_one/data_gen/simulator.py` (modified: +event_timeline param)
- `experiments/exp20_event_augmented.py` (new)
- `experiments/exp20_event_augmented_results.json` (new)
- `tests/test_events.py` (new, 20 tests)
