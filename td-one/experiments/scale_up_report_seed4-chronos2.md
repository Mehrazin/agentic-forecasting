# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=4, models=chronos2.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 139.0s (simulate 55.2s, ingest 14.6s, segments 4.8s, backtest 64.3s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | chronos2 | 2.741 | 3.98 | 1.00 | 5 | 51.2 |
| payment_shock_q2 | chronos2 | 2.372 | 2.71 | 1.00 | 5 | 46.8 |
| payment_shock_q3 | chronos2 | 4.030 | 5.09 | 0.80 | 5 | 49.0 |
| payment_shock_q4 | chronos2 | 4.455 | 6.39 | 0.80 | 5 | 40.2 |
| payment_shock_q5 | chronos2 | 2.656 | 3.18 | 1.00 | 5 | 40.8 |
| rate_type_fixed | chronos2 | 7.158 | 9.92 | 1.00 | 5 | 156.6 |
| rate_type_variable | chronos2 | 1.838 | 2.75 | 1.00 | 5 | 71.4 |
| region_AB | chronos2 | 2.277 | 3.18 | 1.00 | 5 | 32.8 |
| region_BC | chronos2 | 2.186 | 2.77 | 1.00 | 5 | 44.8 |
| region_MB | chronos2 | 2.905 | 3.61 | 0.60 | 5 | 25.0 |
| region_ON | chronos2 | 5.591 | 7.93 | 0.80 | 5 | 93.6 |
| region_QC | chronos2 | 3.261 | 4.33 | 0.80 | 5 | 31.8 |
| term_long | chronos2 | 1.793 | 2.50 | 0.80 | 5 | 15.4 |
| term_medium | chronos2 | 7.395 | 9.94 | 1.00 | 5 | 212.6 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: chronos2.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
