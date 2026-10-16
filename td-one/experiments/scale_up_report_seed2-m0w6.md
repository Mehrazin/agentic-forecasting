# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=2, models=m0.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 154.5s (simulate 92.1s, ingest 24.6s, segments 5.2s, backtest 32.5s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | m0 | 1.932 | 2.95 | 1.00 | 5 | 48.4 |
| payment_shock_q2 | m0 | 2.140 | 2.96 | 1.00 | 5 | 43.8 |
| payment_shock_q3 | m0 | 2.213 | 3.60 | 1.00 | 5 | 46.0 |
| payment_shock_q4 | m0 | 3.289 | 4.44 | 0.60 | 5 | 46.0 |
| payment_shock_q5 | m0 | 1.943 | 3.36 | 1.00 | 5 | 41.6 |
| rate_type_fixed | m0 | 8.080 | 11.96 | 0.80 | 5 | 159.0 |
| rate_type_variable | m0 | 3.456 | 4.91 | 0.60 | 5 | 66.8 |
| region_AB | m0 | 2.126 | 3.80 | 1.00 | 5 | 35.2 |
| region_BC | m0 | 2.424 | 4.03 | 0.80 | 5 | 40.8 |
| region_MB | m0 | 1.255 | 1.71 | 1.00 | 5 | 25.4 |
| region_ON | m0 | 3.498 | 4.68 | 0.80 | 5 | 91.6 |
| region_QC | m0 | 1.255 | 1.62 | 1.00 | 5 | 32.8 |
| term_long | m0 | 0.934 | 1.44 | 1.00 | 5 | 14.6 |
| term_medium | m0 | 6.279 | 8.66 | 0.80 | 5 | 211.2 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: m0.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
