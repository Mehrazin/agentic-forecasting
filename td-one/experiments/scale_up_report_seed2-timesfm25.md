# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=2, models=timesfm25.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 744.5s (simulate 59.8s, ingest 22.0s, segments 5.2s, backtest 657.5s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | timesfm25 | 2.525 | 3.72 | 1.00 | 5 | 48.4 |
| payment_shock_q2 | timesfm25 | 4.088 | 5.78 | 0.60 | 5 | 43.8 |
| payment_shock_q3 | timesfm25 | 3.002 | 5.12 | 1.00 | 5 | 46.0 |
| payment_shock_q4 | timesfm25 | 4.229 | 6.61 | 0.80 | 5 | 46.0 |
| payment_shock_q5 | timesfm25 | 2.892 | 4.57 | 0.80 | 5 | 41.6 |
| rate_type_fixed | timesfm25 | 11.215 | 14.03 | 0.80 | 5 | 159.0 |
| rate_type_variable | timesfm25 | 3.170 | 4.73 | 1.00 | 5 | 66.8 |
| region_AB | timesfm25 | 3.094 | 4.28 | 0.80 | 5 | 35.2 |
| region_BC | timesfm25 | 3.494 | 4.90 | 0.80 | 5 | 40.8 |
| region_MB | timesfm25 | 1.469 | 1.69 | 1.00 | 5 | 25.4 |
| region_ON | timesfm25 | 5.621 | 7.67 | 0.80 | 5 | 91.6 |
| region_QC | timesfm25 | 1.914 | 2.09 | 1.00 | 5 | 32.8 |
| term_long | timesfm25 | 0.864 | 1.20 | 1.00 | 5 | 14.6 |
| term_medium | timesfm25 | 11.176 | 16.26 | 0.80 | 5 | 211.2 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: timesfm25.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
