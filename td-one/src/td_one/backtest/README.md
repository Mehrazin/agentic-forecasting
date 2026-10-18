# `backtest/` — backtest runner

`runner.py`: walks forecast origins, builds PIT-safe panels via the data
service, runs the model ladder, and records CRPS/Brier/coverage per origin.
Used by the scale-up and all real-data benchmarks.

Runbook: [`docs/SCALE_UP_RUNBOOK.md`](../../docs/SCALE_UP_RUNBOOK.md).
Up: [`td_one/`](../README.md).
