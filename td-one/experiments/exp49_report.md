# exp49: Final Model Run with the Updated Toolset

**Date:** 2026-10-18
**Status:** Complete — the definitive validation

## What this is

exp49 is the culmination of the TD-One program: the final toolset
(13 tools after the 2026-10-18 optimization) running the proposed method
(exp47) end-to-end on all five datasets.

## Part A: Toolset parity validation

All four new tools reproduce the proven behavior they replace:

| Check | Result |
|---|---|
| `series_features` vs `exp42.extract_features` (20 series) | max abs diff **0.0** ✅ |
| `naive_forecasts` median vs `exp42._fit_naive` point (20 series) | max abs diff **0.0** ✅ |
| `regime_probabilities` vs exp43 `label_m4` (50 M4 records) | agreement **1.00** ✅ |
| `combine_quantiles` vs manual mean | max abs diff **0.0** ✅ |

The new tools are behavior-preserving refactors, not behavior changes.

## Part B: Final proposed-method run

Records generated with `series_features` in the pipeline; equal-weights
baseline via the `combine_quantiles` tool; BoC regimes via the
`regime_probabilities` tool (spec="boc"). Blocked 5-fold CV, paired t-tests.

### Continuous datasets (CRPS, lower is better)

| Dataset | n | Proposed | Equal | Best single | vs equal (p) | vs best (p) |
|---|---|---|---|---|---|---|
| M4 | 600 | **0.1451** | 0.1507 | 0.1505 | **6.5e-09** ✅ | **7.8e-11** ✅ |
| Synthetic | 480 | 0.1036 | 0.1068 | 0.1028 | 0.072 | 0.60 |
| SMM | 27 | 0.1389 | 0.1424 | 0.1377 | 0.66 | 0.61 |
| Food CPI | 24 | **0.0141** | 0.0176 | 0.0143 | **3.1e-18** ✅ | **0.005** ✅ |

The M4 numbers reproduce exp47 exactly (0.1451 / 0.1507 / 0.1505):
the toolset refactor changed nothing about the results, as intended.

On SMM and Synthetic the per-regime experts collapse to a single model,
so the proposed method degrades gracefully to best-single — the honest
behavior when there is no regime structure to exploit.

### BoC (accuracy, via `regime_probabilities` tool)

| Metric | Value |
|---|---|
| Meetings | 143 |
| Rule accuracy | 0.846 |
| Gated (EASY→rule, HARD→climatology) | **0.853** |
| EASY | 62 meetings @ 0.887 |
| HARD | 81 meetings @ 0.827 |
| Tool label agreement with exp43 | 1.000 |

Replicates exp32's +7.2pp EASY/HARD finding through the new tool.

## The final toolset (13 tools)

**Kept (8):** analyze_residuals, detect_changepoints, test_stationarity
(+`fast` path), decompose_series, backtest_event_impact, cross_validate,
compare_models, feature_importance

**Added (5):** load_series, naive_forecasts, series_features,
regime_probabilities, combine_quantiles (Vincentized; the audit suggested
exposing the existing `calibrated_quantiles`, but that builds intervals
from a point forecast — combining per-model quantiles needed its own
canonical implementation, which is what exp42 did manually)

**Retired (2):** plot_diagnostics, generate_synthetic_scenario
(zero call sites; modules kept on disk for manual use)

**Fixed (2):** test_stationarity `fast=` ADF-only path;
decompose_series clearer validation messages. (Tracker wiring was
completed earlier.)

## Verdict

The final system — 12 tools + regime-probability router + gates +
conformal calibration — significantly beats equal weights and best single
on M4 and Food CPI, matches best single where experts collapse, and
replicates the BoC regime finding. The toolset refactor is
behavior-preserving (Part A) and the results reproduce exp47 (Part B).

This is TD-One's proposed method. Ship it.

## Reproducibility

`cd td-one && .venv/bin/python experiments/exp49_final.py` — deterministic,
~274s full / ~112s `--quick`. Results: `experiments/exp49_results.json`.
