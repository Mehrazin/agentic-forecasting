# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=0, models=timesfm25.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 764.7s (simulate 63.9s, ingest 17.7s, segments 5.4s, backtest 677.7s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | timesfm25 | 3.421 | 4.91 | 1.00 | 5 | 53.4 |
| payment_shock_q2 | timesfm25 | 6.294 | 8.49 | 0.60 | 5 | 53.4 |
| payment_shock_q3 | timesfm25 | 2.934 | 4.16 | 1.00 | 5 | 48.0 |
| payment_shock_q4 | timesfm25 | 4.372 | 5.75 | 0.60 | 5 | 48.0 |
| payment_shock_q5 | timesfm25 | 3.089 | 4.75 | 0.80 | 5 | 43.8 |
| rate_type_fixed | timesfm25 | 10.514 | 14.01 | 0.40 | 5 | 173.2 |
| rate_type_variable | timesfm25 | 2.084 | 2.47 | 1.00 | 5 | 73.4 |
| region_AB | timesfm25 | 2.549 | 3.94 | 0.80 | 5 | 36.4 |
| region_BC | timesfm25 | 5.244 | 8.22 | 0.80 | 5 | 50.2 |
| region_MB | timesfm25 | 2.208 | 2.69 | 1.00 | 5 | 26.8 |
| region_ON | timesfm25 | 9.518 | 14.80 | 0.40 | 5 | 98.2 |
| region_QC | timesfm25 | 3.155 | 4.67 | 1.00 | 5 | 35.0 |
| term_long | timesfm25 | 1.801 | 2.65 | 1.00 | 5 | 18.4 |
| term_medium | timesfm25 | 7.096 | 9.37 | 0.80 | 5 | 228.2 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: timesfm25.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
