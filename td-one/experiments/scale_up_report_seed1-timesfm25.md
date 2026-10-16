# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=1, models=timesfm25.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 720.6s (simulate 55.3s, ingest 15.4s, segments 5.2s, backtest 644.6s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | timesfm25 | 3.590 | 5.45 | 1.00 | 5 | 53.0 |
| payment_shock_q2 | timesfm25 | 4.421 | 6.41 | 0.80 | 5 | 46.4 |
| payment_shock_q3 | timesfm25 | 4.829 | 6.52 | 0.60 | 5 | 45.4 |
| payment_shock_q4 | timesfm25 | 2.697 | 3.31 | 1.00 | 5 | 42.4 |
| payment_shock_q5 | timesfm25 | 4.798 | 6.37 | 0.60 | 5 | 40.4 |
| rate_type_fixed | timesfm25 | 9.122 | 14.17 | 1.00 | 5 | 161.2 |
| rate_type_variable | timesfm25 | 2.605 | 2.81 | 1.00 | 5 | 66.4 |
| region_AB | timesfm25 | 3.317 | 4.47 | 0.80 | 5 | 32.6 |
| region_BC | timesfm25 | 4.062 | 5.57 | 0.80 | 5 | 48.6 |
| region_MB | timesfm25 | 1.706 | 2.38 | 1.00 | 5 | 25.6 |
| region_ON | timesfm25 | 7.336 | 10.39 | 0.60 | 5 | 88.2 |
| region_QC | timesfm25 | 4.185 | 6.33 | 0.80 | 5 | 32.6 |
| term_long | timesfm25 | 1.633 | 2.22 | 0.80 | 5 | 14.6 |
| term_medium | timesfm25 | 8.809 | 13.14 | 1.00 | 5 | 213.0 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: timesfm25.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
