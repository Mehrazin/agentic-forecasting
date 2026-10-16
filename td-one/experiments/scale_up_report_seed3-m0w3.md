# Scale-up Report: TD-One segment backtest

**Date:** 2026-10-16. **Config:** n_customers=10000, n_months=66, seed=3, models=m0.
**Origins:** 2026-04-30, 2026-05-31, 2026-06-30, 2026-07-31, 2026-08-31, 2026-09-30.
**Runtime:** 103.5s (simulate 54.5s, ingest 14.3s, segments 4.9s, backtest 29.8s).

## Per-segment x model scores (mean over origins)

| segment | model | crps | mae | coverage_80 | n_origins | mean_n_renewals |
|---|---|---|---|---|---|---|
| payment_shock_q1 | m0 | 2.606 | 3.78 | 1.00 | 5 | 53.8 |
| payment_shock_q2 | m0 | 1.371 | 1.55 | 1.00 | 5 | 49.0 |
| payment_shock_q3 | m0 | 1.945 | 2.68 | 1.00 | 5 | 44.4 |
| payment_shock_q4 | m0 | 1.450 | 2.17 | 1.00 | 5 | 45.6 |
| payment_shock_q5 | m0 | 1.526 | 1.84 | 1.00 | 5 | 42.8 |
| rate_type_fixed | m0 | 3.748 | 5.20 | 1.00 | 5 | 162.4 |
| rate_type_variable | m0 | 2.396 | 3.60 | 1.00 | 5 | 73.2 |
| region_AB | m0 | 2.645 | 3.60 | 0.80 | 5 | 33.6 |
| region_BC | m0 | 1.399 | 1.55 | 1.00 | 5 | 45.6 |
| region_MB | m0 | 1.469 | 2.11 | 1.00 | 5 | 22.6 |
| region_ON | m0 | 2.259 | 2.63 | 1.00 | 5 | 97.4 |
| region_QC | m0 | 1.684 | 2.04 | 1.00 | 5 | 36.4 |
| term_long | m0 | 1.348 | 1.87 | 1.00 | 5 | 18.0 |
| term_medium | m0 | 3.135 | 2.57 | 1.00 | 5 | 217.6 |

## Interpretation (read before comparing models)

- **The `*_mock` models are interface-only plumbing.** They center on the last observed rate with fixed tight probit spreads, so in a stable-rate environment they score better on CRPS than M0's wider Jeffreys posterior. That gap is a sharpness artifact of the placeholder, NOT evidence about Chronos-2/TimesFM. The real M0-vs-M1 comparison is `experiments/exp2_report.md` (real weights, 3 pre-registered scenarios).
- **Coverage is below nominal for every model** (target 0.80). M0 undercovers and the mocks undercover; interval calibration remains open work at this sample size.
- Segment panels are PIT-masked per origin (`observed_asof`): training months count only outcomes with `outcome_observed_at <= origin`. Future months keep actuals purely as scoring targets.
- Maturities spread year-round (simulator draws a uniform origination month per customer); origins are the last 6 panel month-ends.

## Notes

- Models wired via `td_one.tools.forecasting`: none (local m0 fallback).
- PIT ingest used the parquet fast path (DuckDB-native bulk load into the PIT schema); the `PITStore` row-wise API is too slow at this scale (~40 rows/s).
