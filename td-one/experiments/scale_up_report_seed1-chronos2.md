# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=1, models=chronos2.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 139.9s (simulate 55.4s, ingest 15.5s, segments 5.2s, backtest 63.8s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | chronos2 | 3.514 | 5.01 | 1.00 | 5 | 53.0 |
| payment_shock_q2 | chronos2 | 3.942 | 5.85 | 1.00 | 5 | 46.4 |
| payment_shock_q3 | chronos2 | 4.157 | 5.99 | 0.80 | 5 | 45.4 |
| payment_shock_q4 | chronos2 | 2.590 | 3.26 | 1.00 | 5 | 42.4 |
| payment_shock_q5 | chronos2 | 4.401 | 5.84 | 0.80 | 5 | 40.4 |
| rate_type_fixed | chronos2 | 9.330 | 13.81 | 1.00 | 5 | 161.2 |
| rate_type_variable | chronos2 | 2.407 | 2.89 | 1.00 | 5 | 66.4 |
| region_AB | chronos2 | 2.987 | 3.88 | 0.80 | 5 | 32.6 |
| region_BC | chronos2 | 3.786 | 5.08 | 0.80 | 5 | 48.6 |
| region_MB | chronos2 | 1.506 | 2.30 | 1.00 | 5 | 25.6 |
| region_ON | chronos2 | 6.678 | 9.40 | 1.00 | 5 | 88.2 |
| region_QC | chronos2 | 3.650 | 5.61 | 0.80 | 5 | 32.6 |
| term_long | chronos2 | 1.800 | 2.54 | 0.80 | 5 | 14.6 |
| term_medium | chronos2 | 8.938 | 15.03 | 1.00 | 5 | 213.0 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: chronos2.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
