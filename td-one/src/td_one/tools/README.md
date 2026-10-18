# `tools/` — the 10 agent tools

Statistical tools the agents call through the ReAct loop. Every tool is
wrapped with `@tracked()` (see `agent/tool_tracker.py`); runtime data lands in
`docs/TOOL_USAGE_DATA.md` and drives `docs/TOOL_OPTIMIZATION.md`.

| Module | Tools |
|---|---|
| `data_segment.py` | `data_segment`, `load_series` |
| `evidence.py` | `evidence`, `backtest_event_impact` |
| `forecasting.py` | `forecasting`, `naive_forecasts`, `compare_models`, `cross_validate` |

Plus `test_stationarity`, `detect_changepoints`, `decompose_series`,
`analyze_residuals`, `feature_importance`, `plot_diagnostics`,
`generate_synthetic_scenario` (see audit for the add/drop/fix list).

Audits: [`docs/TOOL_AUDIT.md`](../../docs/TOOL_AUDIT.md).
Up: [`td_one/`](../README.md).
