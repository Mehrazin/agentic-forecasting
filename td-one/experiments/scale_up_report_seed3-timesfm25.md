# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=3, models=timesfm25.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 710.9s (simulate 58.1s, ingest 15.7s, segments 4.9s, backtest 632.2s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | timesfm25 | 3.354 | 4.83 | 1.00 | 5 | 53.8 |
| payment_shock_q2 | timesfm25 | 3.074 | 4.97 | 1.00 | 5 | 49.0 |
| payment_shock_q3 | timesfm25 | 3.925 | 5.93 | 0.80 | 5 | 44.4 |
| payment_shock_q4 | timesfm25 | 2.403 | 2.44 | 1.00 | 5 | 45.6 |
| payment_shock_q5 | timesfm25 | 3.268 | 4.89 | 1.00 | 5 | 42.8 |
| rate_type_fixed | timesfm25 | 6.343 | 9.90 | 1.00 | 5 | 162.4 |
| rate_type_variable | timesfm25 | 2.204 | 2.50 | 1.00 | 5 | 73.2 |
| region_AB | timesfm25 | 3.157 | 3.74 | 0.80 | 5 | 33.6 |
| region_BC | timesfm25 | 2.144 | 3.31 | 1.00 | 5 | 45.6 |
| region_MB | timesfm25 | 2.443 | 3.98 | 1.00 | 5 | 22.6 |
| region_ON | timesfm25 | 3.362 | 4.01 | 1.00 | 5 | 97.4 |
| region_QC | timesfm25 | 3.627 | 5.14 | 0.80 | 5 | 36.4 |
| term_long | timesfm25 | 1.556 | 2.12 | 1.00 | 5 | 18.0 |
| term_medium | timesfm25 | 9.834 | 11.52 | 0.80 | 5 | 217.6 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: timesfm25.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
