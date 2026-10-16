# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=0, models=m0.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 143.7s (simulate 70.9s, ingest 21.2s, segments 5.7s, backtest 45.7s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | m0 | 1.704 | 2.55 | 1.00 | 5 | 53.4 |
| payment_shock_q2 | m0 | 5.696 | 7.40 | 0.40 | 5 | 53.4 |
| payment_shock_q3 | m0 | 1.982 | 2.72 | 1.00 | 5 | 48.0 |
| payment_shock_q4 | m0 | 3.336 | 4.52 | 0.80 | 5 | 48.0 |
| payment_shock_q5 | m0 | 1.828 | 2.87 | 1.00 | 5 | 43.8 |
| rate_type_fixed | m0 | 7.417 | 10.66 | 0.40 | 5 | 173.2 |
| rate_type_variable | m0 | 1.950 | 2.85 | 1.00 | 5 | 73.4 |
| region_AB | m0 | 2.147 | 2.57 | 0.80 | 5 | 36.4 |
| region_BC | m0 | 2.523 | 3.28 | 0.80 | 5 | 50.2 |
| region_MB | m0 | 1.440 | 2.40 | 1.00 | 5 | 26.8 |
| region_ON | m0 | 5.079 | 8.24 | 0.60 | 5 | 98.2 |
| region_QC | m0 | 2.467 | 3.10 | 0.80 | 5 | 35.0 |
| term_long | m0 | 1.658 | 2.61 | 1.00 | 5 | 18.4 |
| term_medium | m0 | 5.518 | 8.38 | 0.80 | 5 | 228.2 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: none (local m0 fallback).
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
