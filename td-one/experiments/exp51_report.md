# exp51: Kitchen Sink — Final Report

**Date:** 2026-10-18
**Status:** Complete
**Question:** we excluded many positive-but-nonsignificant effects from the
proposed method. Were we wrong? What happens when we combine ALL of them?

## Method

Built `KitchenSinkForecaster` (subclass of `ProposedForecaster`) adding every
*honestly combinable* positive-direction component (see
`docs/NONSIGNIFICANT_POSITIVES.md` for the full audit and exclusion reasons):

| Component | Source | Applies to |
|---|---|---|
| +tsfm: TimesFM-2.5 as 5th expert (real weights, normalized space) | exp18b | SMM |
| +macro: ff_3m_chg + term_spread (PIT-safe month-end FRED) | exp33 | SMM |
| +meta: 0.5·w_router + 0.5·P_meta(model\|features) | exp41 | all |
| +gates: grid-searched gate thresholds per fold | exp50 | all |
| +lossreg: loss-vector KMeans regimes instead of hand rules | exp50 | all |
| sink: all applicable combined | — | all |

Excluded with reasons: exp22 (needs live LLM — no honest proxy), exp44
(superseded by exp46's definitive kill), exp36 (diagnostic, not a mechanism),
exp11/exp5/exp4 (already integrated), exp10 (no hierarchy), exp13 (was
negative), exp17 (superseded by kill).

Design: ablations + full sink vs proposed baseline on **identical
records/folds**, blocked 5-fold CV, paired t-tests. M4 n=600, Synthetic
n=480, SMM n=27, Food CPI n=24.

## Results

### M4 (n=600)

| Variant | Mean CRPS | Δ vs proposed | p | Win rate |
|---|---|---|---|---|
| proposed | 0.14501 | — | — | — |
| +meta | 0.14483 | −0.00018 | 0.63 | 0.46 |
| +gates | 0.14496 | −0.00005 | 0.25 | 0.02 |
| +lossreg | 0.14728 | **+0.00227** | **0.0027** | 0.43 |
| **sink** | 0.14574 | +0.00073 | 0.25 | 0.48 |

### Synthetic (n=480)

| Variant | Mean CRPS | Δ vs proposed | p | Win rate |
|---|---|---|---|---|
| proposed | 0.10328 | — | — | — |
| +meta | 0.10253 | −0.00075 | 0.23 | 0.46 |
| +gates | 0.10313 | −0.00015 | 0.17 | 0.01 |
| +lossreg | 0.10385 | +0.00057 | 0.68 | 0.42 |
| **sink** | 0.10285 | −0.00043 | 0.72 | 0.50 |

### SMM (n=27)

| Variant | Mean CRPS | Δ vs proposed | p | Win rate |
|---|---|---|---|---|
| proposed | 0.13891 | — | — | — |
| +tsfm | 0.14538 | +0.00647 | 0.62 | **0.11** |
| +macro | 0.13891 | 0.00000 | — | — |
| +meta | 0.14023 | +0.00132 | 0.71 | 0.37 |
| +gates | 0.13891 | 0.00000 | — | — |
| +lossreg | 0.14042 | +0.00151 | 0.25 | 0.00 |
| **sink** | 0.14057 | +0.00166 | 0.89 | 0.26 |

### Food CPI (n=24)

All variants identical (0.01222) — dataset too small to discriminate;
gates never fire differentially, meta-learner degenerates to single class.

## Key findings

1. **The kitchen sink is not better than the proposed method on any
   dataset.** M4: +0.00073 (p=0.25). SMM: +0.00166 (p=0.89, win rate 0.26).
   The significance filter was correct — the nonsignificant positives were
   noise, or don't transfer into the combination.

2. **+tsfm hurts on SMM** (win rate 0.11). Mechanism: per-fold, TimesFM
   sometimes wins train-CRPS and gets selected as expert (or global best for
   the gate), then loses out-of-fold. The exp18b standalone effect (−12.2%)
   does not survive *as a regime expert*. Consistent with exp50's 0%
   selection finding on M4 — now confirmed on SMM too.

3. **+macro is a null** — the router ignores the extra features (identical
   forecasts). exp33's +1.21% came from a different mechanism (OLS residual
   overlay), not from regime discrimination.

4. **+lossreg significantly hurts on M4** (p=0.0027). Third confirmation
   (exp45, exp50, exp51) that hand-rule regimes beat learned ones.

5. **+meta is the only component with a (weak) positive direction** on
   M4 (−0.00018, p=0.63) and Synthetic (−0.00075, p=0.23) — but far from
   significant, and negative on SMM. Not enough to overturn the filter.

6. **+gates is a null** — learned thresholds don't change behavior
   (gates fire ~5%; threshold choice doesn't matter). Confirms exp50.

## Verdict

**The proposed method stands.** Combining all nonsignificant positives
makes it *directionally worse*, not better. This validates the significance
filter used in exp47: the excluded effects were noise, not underpowered
signal. No changes to the proposed method.

## Files

- `experiments/exp51_kitchen_sink.py` — full pipeline (record generation,
  TSFM inference with disk cache, macro features, KitchenSinkForecaster,
  variant CV harness)
- `experiments/exp51_results.json` — all numbers
- `experiments/exp51_tsfm_cache.json` — cached TimesFM quantiles (27 SMM origins)
- `docs/NONSIGNIFICANT_POSITIVES.md` — the audit
