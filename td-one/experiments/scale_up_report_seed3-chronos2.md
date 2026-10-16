# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=3, models=chronos2.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 134.5s (simulate 56.0s, ingest 14.8s, segments 4.8s, backtest 58.8s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | chronos2 | 3.563 | 4.77 | 1.00 | 5 | 53.8 |
| payment_shock_q2 | chronos2 | 2.536 | 3.78 | 1.00 | 5 | 49.0 |
| payment_shock_q3 | chronos2 | 3.360 | 5.51 | 1.00 | 5 | 44.4 |
| payment_shock_q4 | chronos2 | 2.947 | 3.58 | 1.00 | 5 | 45.6 |
| payment_shock_q5 | chronos2 | 2.957 | 3.77 | 1.00 | 5 | 42.8 |
| rate_type_fixed | chronos2 | 4.621 | 4.87 | 1.00 | 5 | 162.4 |
| rate_type_variable | chronos2 | 2.106 | 2.36 | 1.00 | 5 | 73.2 |
| region_AB | chronos2 | 3.425 | 4.28 | 0.80 | 5 | 33.6 |
| region_BC | chronos2 | 2.922 | 5.16 | 1.00 | 5 | 45.6 |
| region_MB | chronos2 | 2.174 | 3.46 | 1.00 | 5 | 22.6 |
| region_ON | chronos2 | 2.707 | 3.18 | 1.00 | 5 | 97.4 |
| region_QC | chronos2 | 3.594 | 5.10 | 0.80 | 5 | 36.4 |
| term_long | chronos2 | 1.742 | 2.28 | 1.00 | 5 | 18.0 |
| term_medium | chronos2 | 9.844 | 13.16 | 1.00 | 5 | 217.6 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: chronos2.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
