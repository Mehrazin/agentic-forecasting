# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=0, models=timesfm25.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 764.7s (simulate 63.9s, ingest 17.7s, segments 5.4s, backtest 677.7s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | timesfm25 | 3.421 | 4.91 | 1.00 | 5 | 53.4 |
| payment_shock_q2 | timesfm25 | 6.294 | 8.49 | 0.60 | 5 | 53.4 |
| payment_shock_q3 | timesfm25 | 2.934 | 4.16 | 1.00 | 5 | 48.0 |
| payment_shock_q4 | timesfm25 | 4.372 | 5.75 | 0.60 | 5 | 48.0 |
| payment_shock_q5 | timesfm25 | 3.089 | 4.75 | 0.80 | 5 | 43.8 |
| rate_type_fixed | timesfm25 | 10.514 | 14.01 | 0.40 | 5 | 173.2 |
| rate_type_variable | timesfm25 | 2.084 | 2.47 | 1.00 | 5 | 73.4 |
| region_AB | timesfm25 | 2.549 | 3.94 | 0.80 | 5 | 36.4 |
| region_BC | timesfm25 | 5.244 | 8.22 | 0.80 | 5 | 50.2 |
| region_MB | timesfm25 | 2.208 | 2.69 | 1.00 | 5 | 26.8 |
| region_ON | timesfm25 | 9.518 | 14.80 | 0.40 | 5 | 98.2 |
| region_QC | timesfm25 | 3.155 | 4.67 | 1.00 | 5 | 35.0 |
| term_long | timesfm25 | 1.801 | 2.65 | 1.00 | 5 | 18.4 |
| term_medium | timesfm25 | 7.096 | 9.37 | 0.80 | 5 | 228.2 |

## Interpretation (read before comparing models)

- **Real model, single run.** This report covers one real-weight model
  (TimesFM-2.5) vs the same run's M0 panels. There are no `*_mock` rows
  here — the mock scale-up (`scale_up_report.md`) is plumbing only and
  is not compared in this report.
- **Paired tests vs M0: no significant difference.** On the 14 shared
  segments (same seed, same DGP, same origins — pairing valid),
  TimesFM-2.5 wins 5/14 on mean CRPS (paired-t p=0.341, Wilcoxon
  p=0.583). The mean gap (4.591 vs M0 5.360) is driven by a few
  segments; the median picture is a coin flip. This reproduces exp2 v2's
  conclusion ("no model significantly beats M0," all p > 0.18) at scale.
  Single seed, single DGP — any future superiority claim must be
  pre-registered across seeds/DGP variants.
- **Coverage is near nominal** (mean 0.80 across segments, vs the 0.80
  target), consistent with exp2 v2's "TimesFM closest to nominal"
  finding. Interval calibration remains open work for the other rungs.
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

- Models wired via `td_one.tools.forecasting`: timesfm25.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
