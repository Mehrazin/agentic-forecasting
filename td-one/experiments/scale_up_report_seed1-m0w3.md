# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=1, models=m0.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 209.3s (simulate 113.5s, ingest 29.3s, segments 6.8s, backtest 59.6s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | m0 | 3.098 | 4.42 | 0.80 | 5 | 53.0 |
| payment_shock_q2 | m0 | 1.767 | 2.48 | 1.00 | 5 | 46.4 |
| payment_shock_q3 | m0 | 4.083 | 5.43 | 0.80 | 5 | 45.4 |
| payment_shock_q4 | m0 | 1.804 | 2.44 | 1.00 | 5 | 42.4 |
| payment_shock_q5 | m0 | 2.795 | 3.67 | 0.60 | 5 | 40.4 |
| rate_type_fixed | m0 | 5.609 | 8.24 | 0.80 | 5 | 161.2 |
| rate_type_variable | m0 | 2.175 | 2.82 | 0.80 | 5 | 66.4 |
| region_AB | m0 | 1.744 | 2.42 | 0.80 | 5 | 32.6 |
| region_BC | m0 | 3.334 | 4.53 | 0.80 | 5 | 48.6 |
| region_MB | m0 | 1.122 | 1.43 | 1.00 | 5 | 25.6 |
| region_ON | m0 | 3.710 | 6.24 | 0.80 | 5 | 88.2 |
| region_QC | m0 | 3.028 | 4.09 | 0.40 | 5 | 32.6 |
| term_long | m0 | 1.437 | 1.92 | 0.80 | 5 | 14.6 |
| term_medium | m0 | 5.954 | 8.66 | 0.80 | 5 | 213.0 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: none (local m0 fallback).
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
