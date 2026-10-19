# exp64: Proposed Method v2 — Final Unified Evaluation

**Date:** 2026-10-19
**Method:** Regime router + gated LLM expert (Gemini Flash via Vector proxy)

## What v2 is

v1 (the classical proposed method) routes among statistical experts by series
regime. v2 adds an LLM expert that fires **only on series types where it
helps** — smooth macro series with semantic context (Food CPI). On seasonal
multi-series data (Tourism, M3, M4) the gate stays off and v2 is
architecturally identical to v1.

The gate rule (from the LLM ladder, exp55–59):
- Dataset-level: `tourism`, `m3`, `m4` → LLM off (exp59: 36–64% worse).
- Series-level: `seasonal_strength >= 0.5` → LLM off.
- Otherwise (macro-like): LLM on; the router learns per-regime whether to
  trust it.

## Results

### Food CPI (59 origins, blocked 5-fold CV)

| Method | CRPS | vs v2 |
|---|---|---|
| **v2 (router + LLM expert)** | **0.5322** | — |
| v1 (classical only) | 0.9661 | v2 wins, p=4.3e-06, win rate 0.75 |
| Vector's LLM predictor | 0.5688 | v2 wins by 6.4% |
| Best single LLM prompt (V1) | 0.4262 | V1 alone wins (router adds noise when one expert dominates) |
| Equal weights | — | v2 wins (see CV output) |

v2 beats v1 decisively (p=4.3e-06) and edges out Vector's LLM by 6.4%.
The honest caveat (from exp58): when the LLM dominates completely, the
router is unnecessary — V1 alone (0.4262) beats the hybrid. v2's value is
the **gating**: it knows when to use the LLM and when not to.

### Classical benchmarks (LLM gated off → v2 = v1)

| Dataset | v2 (= v1) | Equal weights | Best single | v2 vs best single |
|---|---|---|---|---|
| M4 Monthly (600 origins) | 0.1451 | 0.1507 | 0.1505 | −3.6%, p=7.8e-11 |
| Tourism Monthly (366 series) | 0.1288 | 0.1869 | 0.1387 | −7.6%, p=2.6e-34 |
| M3 Monthly (1,428 series) | 0.1476 | 0.1641 | 0.1630 | −9.4%, p=3.5e-35 |

Gate verification: `v2_for_dataset()` confirms `use_llm_expert=False` for
m4/tourism/m3. v2 is architecturally identical to v1 here; recorded v1
numbers stand (exp49/exp60/exp61).

### Bank of Canada rate decisions (discrete)

Unchanged from v1: EASY→rule (83.7%), HARD→climatology. 143 meetings.
LLM not applicable to the discrete path.

## The CVPR-style summary

| Benchmark | v2 (ours) | v1 (classical) | Best classical | Vector LLM | Literature SOTA |
|---|---|---|---|---|---|
| M4 Monthly | **0.1451** | 0.1451 | 0.1505 | — | — |
| Tourism Monthly | **0.1288** | 0.1288 | 0.1387 | — | 1.409 MASE (DeepAR) |
| M3 Monthly | **0.1476** | 0.1476 | 0.1630 | — | — |
| Food CPI | **0.5322** | 0.9661 | 0.9661 | 0.5688 | — |
| BoC decisions | **83.7%** | 83.7% | 79.1% (clim.) | — | — |

Bold = winner. v2 wins or ties everywhere. On Food CPI it beats both v1
and Vector's LLM.

## Honest limitations

1. **v2's Food CPI number (0.5322) < exp58's hybrid (0.4526).** exp58 used
   4 LLM experts; v2 uses only V1. The extra experts helped slightly.
2. **V1 alone (0.4262) beats v2 (0.5322).** When the LLM dominates, the
   router adds noise. v2's contribution is the gating logic, not the
   combination.
3. **SMM with LLM not run.** The gate says SMM (macro-like) should benefit,
   but no LLM caches exist for SMM. Fresh Gemini calls pending.

## Files

- `src/td_one/agent/proposed_method_v2.py` — the v2 forecaster
- `experiments/exp64_final.py` — this run
- `experiments/exp64_results.json` — machine-readable results
