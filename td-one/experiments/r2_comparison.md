# R2 Experiment Comparison: PIT-Safe Reruns

**Updated:** 2026-10-17 — supersedes the 2026-10-16 version below-the-fold.

> **RETRACTION (2026-10-17, per Google Argon audit):** The 2026-10-16 version of
> this document concluded the macro benefit was 100% leak. That conclusion was
> based on a **dict-lookup bug**: `fred_z.get(forecast_month)` looked up `t+1`
> in a dict truncated at `t`, always returning `0.0` — silently zeroing the macro
> signal. The macro track was never properly tested PIT-safe. The fixed rerun
> uses **lagged `z[t]`** (predictive regression). Results below replace the
> 2026-10-16 findings. The old text is kept under "Superseded analysis" for
> provenance.

**Method:** DGP keeps full-history z (ground truth, unchanged outcomes). Only the
MODEL's features are PIT-safe (`load_fred_z(asof=origin)`). Any CRPS change is
purely the leak's effect — and, after the fix, purely the signal's effect.

## R2-1: exp10 (classical) — lagged macro shows a real signal

| Model | R1 (leaky) CRPS | R2-old (buggy) CRPS | R2-new (lagged) CRPS |
|-------|---------------:|--------------------:|---------------------:|
| m0_w6 | 2.305 | 2.305 | 2.280 |
| m0_w3 | — | — | **1.457** |
| macro_m0 | 1.627 | 2.305 | **1.923** |

Lagged macro_m0 beats m0_w6 by **−0.357 CRPS** (~53% of the leaky −0.678 gain).
The signal is real PIT-safe — not a leak.

**Caveats (statistical):** single seed (seed=0), no uncertainty quantification.
Against the best M0 window, **m0_w3 (1.457) beats macro_m0 (1.923) by 32%** —
macro has not been shown to add value over a responsive baseline (window
confounding: macro_m0 uses trailing_window=6; the −0.357 may reflect the 3-month
`z[t]` compensating for m0_w6's stale average rather than true macro value).

`r2_exp10_classical.py` → `r2_exp10_results.json` (49s runtime, 2026-10-17).

## R2-2: exp13 (TSFM macro) — residual conditioning hurts Chronos2 4/6

| DGP | Seed | chronos2 | chronos2_macro | Δ% |
|-----|------|---------:|---------------:|---:|
| standard | 0 | 1.541 | 1.523 | −1.2% |
| macro | 0 | 1.987 | 2.046 | +3.0% |
| standard | 1 | 1.331 | 1.564 | +17.5% |
| macro | 1 | 1.654 | 1.851 | +11.9% |
| standard | 2 | 1.235 | 1.522 | +23.2% |
| macro | 2 | 1.511 | 1.513 | +0.1% |

1 win (−1.2%), 4 losses (+3% to +23%), 1 tie — sign test p=0.375 two-sided
(1 win, 4 losses, 1 tie; n=5 non-tied). The residual approach (subtract β·z from
training inputs, re-add at forecast time) distorts the input distribution without
adequate compensation. TimesFM R2 was blocked by a transient environment hang
(weights loading) — Chronos2 results only.

`r2_exp13_tsfm_macro.py` → `r2_exp13_tsfm_macro_results.json` (2026-10-17).

## R2-3: exp15 (corrected-sign macro, simulator v2) — PIT-safe = harmless

Mean CRPS over 3 seeds (true β=−0.3, v2 lock-in economics):

| Model | mean CRPS |
|-------|----------:|
| m0 | 0.6092 |
| m0_macro_oracle | 0.6199 |
| m0_macro_est | 0.6920 |

Oracle gain −0.011 (slightly negative), est gain −0.083. On the corrected-sign
v2 DGP the oracle macro is ~zero and the estimated overlay actively hurts —
the "93% of oracle" claim stays retracted.

`r2_exp15_macro_corrected.py` → `r2_exp15_macro_corrected_results.json`.

## R2-4: exp14 (M4 real) on a common 0.10–0.90 grid — TSFM win survives

| Model | CRPS | vs naive |
|-------|-----:|---------:|
| naive | 462.84 | — |
| conf_naive | 463.65 | +0.2% |
| timesfm25 | 358.13 | **−22.6%** |
| chronos2 | 351.18 | **−24.1%** |

Same quantile grid for all models (classical residual quantiles computed directly
on the grid; Chronos2 interpolated interior, no extrapolation; TimesFM exact).
The TSFM edge is not a scoring artifact.

`r2_exp14_common_grid.py` → `r2_exp14_common_grid_results.json` (run
2026-10-17T06:14 UTC).

## Interpretation

1. **exp10:** the original "leak was 100%" verdict is retracted — lagged macro
   recovers ~53% of the leaky gain. But macro has **not beaten m0_w3**; single
   seed, window-confounded. Exploratory positive.
2. **exp13:** residual macro conditioning **hurts Chronos2 in 4/6 cases**
   (n=6, p=0.375 — anecdotal, not established). The track is method-specific:
   classical macro is exploratory-positive, TSFM residual conditioning is
   exploratory-negative.
3. **exp15:** corrected-sign v2 DGP — oracle macro ≈ zero, estimated overlay
   hurts. Retraction stands.
4. **exp14/r2-4:** TSFM win is robust to the common grid.

**Bottom line:** we don't know yet whether PIT-safe macro adds value — the
findings are suggestive but need multi-seed confirmatory evaluation.

## Files
- `experiments/r2_exp10_classical.py` → `r2_exp10_results.json`
- `experiments/r2_exp13_tsfm_macro.py` → `r2_exp13_tsfm_macro_results.json`
- `experiments/r2_exp15_macro_corrected.py` → `r2_exp15_macro_corrected_results.json`
- `experiments/r2_exp14_common_grid.py` → `r2_exp14_common_grid_results.json`

---

# Superseded analysis (2026-10-16, based on the dict-lookup bug)

**Background:** DATA_AUDIT.md P0-1 found `load_fred_z` computed z-scores on full
FRED history (1986–2026), leaking future into every forecast origin. R2 reruns
use per-origin `load_fred_z(asof=origin)`.

**Old verdict (WRONG):** "None of the published macro results survive PIT-safe
reruns. The FRED overlay, as currently designed, is not a realizable forecasting
strategy." exp10's R2-old table showed macro_m0 (2.305) **exactly equal** to
m0_w6 (2.305), and exp13 showed the benefit reversing to a penalty on all 3
seeds. Both tables were artifacts of the bug: with `fred_z.get(forecast_month)`
always returning 0.0, the model saw no macro signal at all — PIT-safety was
never actually tested.
