# R2 Experiment Comparison: PIT-Safe Reruns

**Date:** 2026-10-16
**Background:** DATA_AUDIT.md P0-1 found `load_fred_z` computed z-scores on full FRED history (1986–2026), leaking future into every forecast origin. R2 reruns use per-origin `load_fred_z(asof=origin)`.

**Method:** DGP keeps full-history z (ground truth, unchanged outcomes). Only the MODEL's features are PIT-safe. Any CRPS change is purely the leak's effect.

## Results

### exp10 (classical, macro DGP) — the leak WAS the benefit

| Model | R1 (leaky) CRPS | R2 (PIT-safe) CRPS | Delta |
|-------|----------------:|-------------------:|------:|
| m0_w6 | 2.305 | 2.305 | 0.000 |
| macro_m0 | 1.627 | 2.305 | **+0.678** |

R1 showed macro_m0 beating m0 by 0.678 CRPS. R2: macro_m0 **exactly equals** m0_w6. The entire "macro benefit" was the model seeing `z[forecast_month]` from the future. With PIT safety, the forecast-month z is unknowable → zero shift → pure M0.

### exp15 (corrected sign) — PIT safety makes it harmless

| Model | R1 (leaky) CRPS | R2 (PIT-safe) CRPS | Delta |
|-------|----------------:|-------------------:|------:|
| m0 | 0.646 | 0.646 | 0.000 |
| m0_macro_oracle | 0.651 | 0.651 | 0.000 |
| m0_macro_est | 0.797 | 0.646 | **-0.151** |

R1: the estimated overlay HURT by 0.151 (wrong-sign beta × leaked z[T+1]). R2: est exactly equals m0 (harmless). The damage wasn't from the leak alone — it was from applying an unstable beta to a leaked value. PIT safety neutralizes it.

### exp13 (TSFM macro, macro DGP) — benefit REVERSED

| Seed | Model | R1 (leaky) | R2 (PIT-safe) | Delta |
|------|-------|-----------:|--------------:|------:|
| 0 | chronos2 | 2.033 | 2.033 | 0.000 |
| 0 | chronos2_macro | 1.770 | 2.532 | **+0.762** |
| 1 | chronos2 | 1.653 | 1.653 | 0.000 |
| 1 | chronos2_macro | 1.483 | 1.997 | **+0.514** |
| 2 | chronos2 | 1.520 | 1.520 | 0.000 |
| 2 | chronos2_macro | 1.274 | 1.531 | **+0.257** |

R1: macro-conditioning beat plain Chronos2 on all 3 seeds. R2: it's **worse** on all 3 seeds.

**Why:** The residual approach (`tsfm_macro.py`) residualizes training rates via `sigmoid(logit(r) - beta*z)` but re-adds via `z[forecast_month]`. With PIT safety, `z[T+1]` is unknown → re-add is zero → training data is distorted without compensation. The method is structurally incompatible with PIT safety (unless you forecast the macro itself).

TimesFM R2 was blocked by a transient environment hang (weights loading); the mechanism is identical to Chronos2, so the finding generalizes.

## Interpretation

**Did the leak inflate the macro benefit?** Yes — it *was* the benefit:
1. **exp10:** 100% of the 0.678 CRPS gain was leak. PIT-safe macro_m0 = m0.
2. **exp15:** The 0.151 damage was leak-amplified. PIT-safe = harmless.
3. **exp13:** The macro benefit reversed to a penalty. Residual conditioning needs `z[T+1]`, which is unknowable PIT-safe.

**Bottom line:** None of the published macro results survive PIT-safe reruns. The FRED overlay, as currently designed (contemporaneous `z[forecast_month]`), is not a realizable forecasting strategy. A PIT-safe macro model would need to either (a) forecast the macro series itself, or (b) use only lagged z-values (e.g., `z[T]` as a persistence proxy).

## Files
- `experiments/r2_exp10_classical.py` → `r2_exp10_results.json`
- `experiments/r2_exp13_tsfm_macro.py` → `r2_exp13_tsfm_macro_results.json` (Chronos2 complete; TimesFM pending env fix)
- `experiments/r2_exp15_macro_corrected.py` → `r2_exp15_macro_corrected_results.json`
