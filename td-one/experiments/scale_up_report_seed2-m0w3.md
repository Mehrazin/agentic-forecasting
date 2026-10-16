# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=2, models=m0.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 114.3s (simulate 57.3s, ingest 22.7s, segments 4.7s, backtest 29.6s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | m0 | 2.310 | 2.60 | 0.80 | 5 | 48.4 |
| payment_shock_q2 | m0 | 2.494 | 4.04 | 0.80 | 5 | 43.8 |
| payment_shock_q3 | m0 | 2.481 | 3.37 | 0.80 | 5 | 46.0 |
| payment_shock_q4 | m0 | 4.763 | 5.93 | 0.60 | 5 | 46.0 |
| payment_shock_q5 | m0 | 1.610 | 2.20 | 1.00 | 5 | 41.6 |
| rate_type_fixed | m0 | 8.147 | 11.33 | 0.60 | 5 | 159.0 |
| rate_type_variable | m0 | 4.264 | 5.92 | 0.60 | 5 | 66.8 |
| region_AB | m0 | 4.322 | 5.87 | 0.40 | 5 | 35.2 |
| region_BC | m0 | 2.994 | 4.81 | 0.80 | 5 | 40.8 |
| region_MB | m0 | 1.136 | 1.28 | 1.00 | 5 | 25.4 |
| region_ON | m0 | 4.465 | 6.29 | 0.80 | 5 | 91.6 |
| region_QC | m0 | 1.478 | 2.35 | 1.00 | 5 | 32.8 |
| term_long | m0 | 1.061 | 1.61 | 1.00 | 5 | 14.6 |
| term_medium | m0 | 10.209 | 14.31 | 0.40 | 5 | 211.2 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: none (local m0 fallback).
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
