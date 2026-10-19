# `evaluation/` — metrics and honesty checks

How we score forecasts and how we keep ourselves honest. CRPS is the primary
metric (proper scoring rule for quantile forecasts); the leakage auditor is the
backstop that fails any experiment peeking at the future.

| Module | What it is |
|---|---|
| `metrics.py` | CRPS (primary), Brier, MAE, 80% interval coverage |
| `leakage.py` | Leakage auditor — fails experiments that peek at the future |
| `real_benchmark.py` | Real-data benchmark harness |

Up: [`td_one/`](../README.md).
