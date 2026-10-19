# `backtest/` — backtest runner

`runner.py`: walks forecast origins, builds PIT-safe panels via the data
service, runs the model ladder, and records CRPS/Brier/coverage per origin.
This is the harness behind the scale-ups and all real-data benchmarks — every
number in the reports came through here.

Runbook: [`docs/SCALE_UP_RUNBOOK.md`](../../docs/SCALE_UP_RUNBOOK.md).
Up: [`td_one/`](../README.md).
