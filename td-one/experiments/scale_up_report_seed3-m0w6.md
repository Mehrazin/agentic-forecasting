# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=3, models=m0.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 104.8s (simulate 55.8s, ingest 14.6s, segments 4.8s, backtest 29.7s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | m0 | 4.036 | 5.50 | 0.60 | 5 | 53.8 |
| payment_shock_q2 | m0 | 3.123 | 5.21 | 0.80 | 5 | 49.0 |
| payment_shock_q3 | m0 | 3.395 | 5.12 | 0.80 | 5 | 44.4 |
| payment_shock_q4 | m0 | 1.944 | 3.09 | 1.00 | 5 | 45.6 |
| payment_shock_q5 | m0 | 2.952 | 4.22 | 0.80 | 5 | 42.8 |
| rate_type_fixed | m0 | 15.564 | 19.51 | 0.20 | 5 | 162.4 |
| rate_type_variable | m0 | 2.021 | 2.11 | 1.00 | 5 | 73.2 |
| region_AB | m0 | 2.523 | 3.38 | 0.80 | 5 | 33.6 |
| region_BC | m0 | 2.821 | 4.63 | 0.80 | 5 | 45.6 |
| region_MB | m0 | 2.188 | 3.51 | 1.00 | 5 | 22.6 |
| region_ON | m0 | 4.464 | 5.67 | 0.80 | 5 | 97.4 |
| region_QC | m0 | 2.648 | 3.40 | 0.80 | 5 | 36.4 |
| term_long | m0 | 1.380 | 1.86 | 1.00 | 5 | 18.0 |
| term_medium | m0 | 12.150 | 15.79 | 0.60 | 5 | 217.6 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: m0.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
