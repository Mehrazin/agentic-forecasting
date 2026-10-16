# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=2, models=chronos2.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 134.3s (simulate 56.1s, ingest 14.7s, segments 4.7s, backtest 58.7s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | chronos2 | 2.646 | 3.40 | 1.00 | 5 | 48.4 |
| payment_shock_q2 | chronos2 | 3.692 | 5.14 | 1.00 | 5 | 43.8 |
| payment_shock_q3 | chronos2 | 2.486 | 3.60 | 1.00 | 5 | 46.0 |
| payment_shock_q4 | chronos2 | 4.496 | 6.95 | 0.80 | 5 | 46.0 |
| payment_shock_q5 | chronos2 | 2.287 | 3.15 | 1.00 | 5 | 41.6 |
| rate_type_fixed | chronos2 | 9.447 | 12.19 | 0.80 | 5 | 159.0 |
| rate_type_variable | chronos2 | 2.896 | 4.67 | 1.00 | 5 | 66.8 |
| region_AB | chronos2 | 3.406 | 4.13 | 0.60 | 5 | 35.2 |
| region_BC | chronos2 | 3.932 | 5.45 | 0.60 | 5 | 40.8 |
| region_MB | chronos2 | 1.449 | 1.51 | 1.00 | 5 | 25.4 |
| region_ON | chronos2 | 4.479 | 5.57 | 0.80 | 5 | 91.6 |
| region_QC | chronos2 | 1.725 | 1.42 | 1.00 | 5 | 32.8 |
| term_long | chronos2 | 0.975 | 1.51 | 1.00 | 5 | 14.6 |
| term_medium | chronos2 | 10.490 | 14.37 | 1.00 | 5 | 211.2 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: chronos2.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
