# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-15. **Config:** n_customers=10000, n_months=66, seed=0, models=m0,chronos2_mock,timesfm_mock.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 176.7s (simulate 54.6s, ingest 28.8s, segments 5.5s, backtest 87.7s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | chronos2_mock | 2.185 | 2.81 | 0.80 | 5 | 53.4 |
| payment_shock_q2 | chronos2_mock | 5.948 | 7.50 | 0.40 | 5 | 53.4 |
| payment_shock_q3 | chronos2_mock | 1.450 | 2.08 | 1.00 | 5 | 48.0 |
| payment_shock_q4 | chronos2_mock | 3.458 | 4.89 | 0.60 | 5 | 48.0 |
| payment_shock_q5 | chronos2_mock | 1.739 | 2.72 | 0.80 | 5 | 43.8 |
| rate_type_fixed | chronos2_mock | 6.611 | 8.83 | 0.60 | 5 | 173.2 |
| rate_type_variable | chronos2_mock | 1.641 | 2.53 | 1.00 | 5 | 73.4 |
| region_AB | chronos2_mock | 1.962 | 2.78 | 0.80 | 5 | 36.4 |
| region_BC | chronos2_mock | 2.532 | 3.80 | 0.80 | 5 | 50.2 |
| region_MB | chronos2_mock | 2.368 | 3.13 | 0.60 | 5 | 26.8 |
| region_ON | chronos2_mock | 3.941 | 5.87 | 0.60 | 5 | 98.2 |
| region_QC | chronos2_mock | 3.694 | 4.90 | 0.60 | 5 | 35.0 |
| term_long | chronos2_mock | 1.474 | 2.21 | 0.80 | 5 | 18.4 |
| term_medium | chronos2_mock | 5.713 | 7.53 | 0.60 | 5 | 228.2 |
| payment_shock_q1 | m0 | 2.042 | 3.01 | 1.00 | 5 | 53.4 |
| payment_shock_q2 | m0 | 5.860 | 7.74 | 0.40 | 5 | 53.4 |
| payment_shock_q3 | m0 | 3.148 | 4.53 | 0.60 | 5 | 48.0 |
| payment_shock_q4 | m0 | 4.196 | 5.46 | 0.60 | 5 | 48.0 |
| payment_shock_q5 | m0 | 3.578 | 5.09 | 0.60 | 5 | 43.8 |
| rate_type_fixed | m0 | 17.974 | 21.96 | 0.40 | 5 | 173.2 |
| rate_type_variable | m0 | 1.904 | 2.68 | 1.00 | 5 | 73.4 |
| region_AB | m0 | 2.034 | 2.67 | 0.80 | 5 | 36.4 |
| region_BC | m0 | 3.932 | 5.68 | 0.80 | 5 | 50.2 |
| region_MB | m0 | 1.436 | 2.33 | 1.00 | 5 | 26.8 |
| region_ON | m0 | 9.426 | 13.09 | 0.20 | 5 | 98.2 |
| region_QC | m0 | 2.864 | 4.26 | 0.80 | 5 | 35.0 |
| term_long | m0 | 1.978 | 3.06 | 0.80 | 5 | 18.4 |
| term_medium | m0 | 14.671 | 19.57 | 0.40 | 5 | 228.2 |
| payment_shock_q1 | timesfm_mock | 2.178 | 2.81 | 0.80 | 5 | 53.4 |
| payment_shock_q2 | timesfm_mock | 5.888 | 7.50 | 0.40 | 5 | 53.4 |
| payment_shock_q3 | timesfm_mock | 1.439 | 2.08 | 1.00 | 5 | 48.0 |
| payment_shock_q4 | timesfm_mock | 3.467 | 4.89 | 0.60 | 5 | 48.0 |
| payment_shock_q5 | timesfm_mock | 1.743 | 2.72 | 0.80 | 5 | 43.8 |
| rate_type_fixed | timesfm_mock | 6.642 | 8.83 | 0.60 | 5 | 173.2 |
| rate_type_variable | timesfm_mock | 1.648 | 2.53 | 1.00 | 5 | 73.4 |
| region_AB | timesfm_mock | 1.964 | 2.78 | 0.80 | 5 | 36.4 |
| region_BC | timesfm_mock | 2.548 | 3.80 | 0.80 | 5 | 50.2 |
| region_MB | timesfm_mock | 2.371 | 3.13 | 0.60 | 5 | 26.8 |
| region_ON | timesfm_mock | 3.888 | 5.87 | 0.60 | 5 | 98.2 |
| region_QC | timesfm_mock | 3.707 | 4.90 | 0.60 | 5 | 35.0 |
| term_long | timesfm_mock | 1.471 | 2.21 | 0.80 | 5 | 18.4 |
| term_medium | timesfm_mock | 5.508 | 7.53 | 0.60 | 5 | 228.2 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: m0, chronos2_mock, timesfm_mock.
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
