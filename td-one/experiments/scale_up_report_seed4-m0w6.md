# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=4, models=m0.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 106.3s (simulate 55.4s, ingest 15.0s, segments 4.8s, backtest 31.1s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | m0 | 2.594 | 3.91 | 1.00 | 5 | 51.2 |
| payment_shock_q2 | m0 | 1.328 | 1.50 | 1.00 | 5 | 46.8 |
| payment_shock_q3 | m0 | 3.646 | 4.83 | 0.60 | 5 | 49.0 |
| payment_shock_q4 | m0 | 4.303 | 6.34 | 0.40 | 5 | 40.2 |
| payment_shock_q5 | m0 | 2.749 | 3.58 | 0.80 | 5 | 40.8 |
| rate_type_fixed | m0 | 11.776 | 15.45 | 0.40 | 5 | 156.6 |
| rate_type_variable | m0 | 2.050 | 3.31 | 1.00 | 5 | 71.4 |
| region_AB | m0 | 1.968 | 3.10 | 0.80 | 5 | 32.8 |
| region_BC | m0 | 1.879 | 2.02 | 1.00 | 5 | 44.8 |
| region_MB | m0 | 2.481 | 3.44 | 0.80 | 5 | 25.0 |
| region_ON | m0 | 5.702 | 7.86 | 0.60 | 5 | 93.6 |
| region_QC | m0 | 3.104 | 4.42 | 0.60 | 5 | 31.8 |
| term_long | m0 | 1.242 | 1.76 | 1.00 | 5 | 15.4 |
| term_medium | m0 | 9.403 | 12.58 | 0.60 | 5 | 212.6 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: m0.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
