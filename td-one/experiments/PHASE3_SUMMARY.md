# Phase 3: Reflect and Adjust — Summary

**Date:** 2026-10-17
**Status:** Complete. All four adjustments implemented, tested, and evaluated.
**Parent task:** Phase 3 adjustments based on exp21–23 learnings.

---

## What was done

### 1. Strengthened DGP toward news-leading events ✅

**Change** (`src/td_one/data_gen/events.py`):
- `HOUSING_DISRUPTION` transmission lag 1 → 2 months (appraisal/LTV effects
  take time; documented rationale). Housing news (delay ~1.1mo) now *leads*
  the DGP effect (+0.92mo) instead of coinciding with it (−0.08mo).
- New `NEWS_LEAD_MONTHS` dict: lead = transmission_lag − mean news delay,
  per event type. Kept in sync by test.
- New `DEFAULT_EVENT_TYPE_PROBS`: POLICY_SHOCK 0.30, RATE_SHIFT 0.30,
  HOUSING_DISRUPTION 0.20, LENDER_SHIFT 0.10, MACRO_SURPRISE 0.10.
- `generate_event_timeline()` accepts `event_type_probs` override (backward
  compatible).

**Verification:** empirical news-leading fraction 84% over 20 seeds (target
≥60%). 23/23 tests pass (3 new).

### 2. Fixed v2's decision policy ✅

**New module** (`src/td_one/agent/decision_policy.py`, 11 tests):
- `shrink_shift()`: James-Stein-style `n/(n+10)` shrinkage toward zero.
  n=3 → 0.23, n=10 → 0.50. Small post-break samples no longer trigger
  full adjustments.
- `calibrated_quantiles()`: interval half-width floored at
  `0.5 × hist_vol × z`, plus `sqrt(1+1/n)` inflation for level-estimation
  uncertainty. Fixes exp22's degenerate-interval failure (width 0.03 vs
  m0's 0.23).
- `corroboration_gate()`: adjust iff |shift| ≥ 2pp AND (news within 30d OR
  |t| > 3.0). v2's old p<0.10 threshold is explicitly rejected without
  news (tested).

### 3. Regime-aware selection (exp24) — honest wash ⚖️

**Script:** `experiments/exp24_regime_aware.py` → `exp24_results.json` + report.

Regime from PIT-safe news signals (article counts 30/90d + `backtest_event_impact`
with news dates as proxy events): QUIET→ARIMA, ELEVATED→ensemble, ACTIVE→ETS.

| Method | CRPS | Gap closed |
|---|---|---|
| ETS (best single) | 0.1203 | 0% |
| regime (v2) | 0.1205 | −1.4% |
| exp23 diagnostic | 0.1207 | −2.9% |

v1 (ACTIVE→event_aware) was catastrophic (−73%); v2's post-hoc mapping
correction fixed it. But v2 ties best-single (p=0.93) — **robustness, not
alpha**. Same pattern as exp23 in eventful segments (+4.1% worse than ETS):
misclassified origins get crushed exactly when adaptation matters.

### 4. Longer horizons (exp25) — hypothesis rejected ❌

**Script:** `experiments/exp25_h3_horizon.py` → `exp25_results.json` + report.

h=3 backtest: ETS vs ARIMA vs NEWS (OLS with exp21's discovered features).

| Model | CRPS (h=3) | vs ETS |
|---|---|---|
| ETS | 0.0769 | — |
| ARIMA | 0.0799 | +4% |
| NEWS | 0.0962 | **+25%, p=0.0013** |

News is *significantly worse* at h=3. OLS overfits (6 features, ~27 points);
the news→outcome link attenuates with horizon; ETS(h=3) degrades gracefully
leaving little headroom.

---

## The Phase-3 verdict

| Adjustment | Outcome |
|---|---|
| News-leading DGP | ✅ Built, verified (84% leading) |
| Decision policy fixes | ✅ Built, tested (11/11) |
| Regime-aware selection | ⚖️ Ties best single; no alpha |
| h=3 horizon | ❌ News significantly worse |

**The "news as features" line is now falsified at h=1 (exp21), for
selection (exp23/24), and at h=3 (exp25).** The infrastructure (event
simulator, sandbox, 10 tools, decision policy) is validated and reusable.
The remaining hypotheses are (a) explicit onset classification rather than
regression on counts, or (b) a fundamentally stronger event/news coupling.

## Files changed/added

- `src/td_one/data_gen/events.py` (modified: lags, probs, NEWS_LEAD_MONTHS)
- `src/td_one/agent/decision_policy.py` (new)
- `tests/test_decision_policy.py` (new, 11 tests)
- `tests/test_events.py` (3 new tests)
- `experiments/exp24_regime_aware.py` + `exp24_results.json` + `exp24_report.md`
- `experiments/exp25_h3_horizon.py` + `exp25_results.json` + `exp25_report.md`
- `experiments/PHASE3_SUMMARY.md` (this file)

All ruff-clean. Not committed per instructions.
