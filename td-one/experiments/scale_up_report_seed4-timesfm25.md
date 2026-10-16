# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=4, models=timesfm25.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 714.0s (simulate 56.7s, ingest 14.7s, segments 5.0s, backtest 637.6s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | timesfm25 | 2.770 | 4.16 | 1.00 | 5 | 51.2 |
| payment_shock_q2 | timesfm25 | 2.437 | 3.64 | 1.00 | 5 | 46.8 |
| payment_shock_q3 | timesfm25 | 4.085 | 5.46 | 0.60 | 5 | 49.0 |
| payment_shock_q4 | timesfm25 | 4.336 | 6.63 | 0.60 | 5 | 40.2 |
| payment_shock_q5 | timesfm25 | 2.494 | 3.03 | 1.00 | 5 | 40.8 |
| rate_type_fixed | timesfm25 | 8.546 | 12.24 | 0.80 | 5 | 156.6 |
| rate_type_variable | timesfm25 | 2.118 | 2.64 | 1.00 | 5 | 71.4 |
| region_AB | timesfm25 | 2.260 | 3.18 | 1.00 | 5 | 32.8 |
| region_BC | timesfm25 | 2.345 | 3.11 | 1.00 | 5 | 44.8 |
| region_MB | timesfm25 | 3.153 | 3.88 | 0.60 | 5 | 25.0 |
| region_ON | timesfm25 | 5.997 | 8.76 | 0.60 | 5 | 93.6 |
| region_QC | timesfm25 | 3.173 | 4.45 | 0.80 | 5 | 31.8 |
| term_long | timesfm25 | 1.603 | 2.32 | 0.80 | 5 | 15.4 |
| term_medium | timesfm25 | 9.092 | 11.77 | 0.80 | 5 | 212.6 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: timesfm25.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
