# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=0, models=chronos2.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 144.4s (simulate 56.3s, ingest 17.8s, segments 4.7s, backtest 65.4s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | chronos2 | 2.586 | 3.65 | 1.00 | 5 | 53.4 |
| payment_shock_q2 | chronos2 | 6.319 | 8.49 | 0.80 | 5 | 53.4 |
| payment_shock_q3 | chronos2 | 2.349 | 2.65 | 1.00 | 5 | 48.0 |
| payment_shock_q4 | chronos2 | 4.146 | 5.07 | 0.60 | 5 | 48.0 |
| payment_shock_q5 | chronos2 | 2.336 | 3.76 | 1.00 | 5 | 43.8 |
| rate_type_fixed | chronos2 | 9.290 | 12.11 | 0.80 | 5 | 173.2 |
| rate_type_variable | chronos2 | 1.804 | 2.65 | 1.00 | 5 | 73.4 |
| region_AB | chronos2 | 2.199 | 2.49 | 1.00 | 5 | 36.4 |
| region_BC | chronos2 | 5.164 | 8.11 | 0.80 | 5 | 50.2 |
| region_MB | chronos2 | 2.290 | 2.97 | 1.00 | 5 | 26.8 |
| region_ON | chronos2 | 7.782 | 12.32 | 1.00 | 5 | 98.2 |
| region_QC | chronos2 | 2.938 | 4.26 | 1.00 | 5 | 35.0 |
| term_long | chronos2 | 2.217 | 2.89 | 1.00 | 5 | 18.4 |
| term_medium | chronos2 | 11.065 | 14.65 | 0.80 | 5 | 228.2 |

## Interpretation (read before comparing models)

- **Real model, single run.** This report covers one real-weight model
  (Chronos-2) vs the same run's M0 panels. There are no `*_mock` rows
  here — the mock scale-up (`scale_up_report.md`) is plumbing only and
  is not compared in this report.
- **Paired tests vs M0: no significant difference.** On the 14 shared
  segments (same seed, same DGP, same origins — pairing valid),
  Chronos-2 wins 7/14 on mean CRPS (paired-t p=0.212, Wilcoxon
  p=0.583). The mean gap (4.463 vs M0 5.360) is driven by a few
  segments; the median picture is a coin flip. This reproduces exp2 v2's
  conclusion ("no model significantly beats M0," all p > 0.18) at scale.
  Single seed, single DGP — any future superiority claim must be
  pre-registered across seeds/DGP variants.
- **Coverage overcovers** (values {0.6, 0.8, 1.0} across segments, mean
  0.914 vs the 0.80 target), consistent with exp2 v2's Chronos-2
  overcoverage finding. Interval calibration remains open work.
- **Methods note:** real-model outputs were monotone-sorted (ascending
  values reassigned to ascending levels) to fix quantile crossing — the
  standard monotone rearrangement, disclosed not hidden. No diagnostic
  on crossing frequency was recorded; a future run should log it.
- **Framing:** this comparison is count-aware Bayes (M0 sees
  `(n_renewals, n_switches)`) vs rate-only zero-shot on short panels
  (5–12 monthly points per origin — far below the models' design
  context). It tests few-shot priors on this DGP, not TSFM capability
  in general.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: chronos2.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
