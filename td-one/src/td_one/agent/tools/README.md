# `agent/tools/` — tool implementations

The actual tool functions the agents call. Each is a pure function with
Pydantic-validated inputs, PIT-safety checks where temporal data is involved,
and JSON-serializable outputs (so results flow back into the ReAct trace as
observations).

| Module | Tools |
|---|---|
| `analysis.py` | `analyze_residuals`, `feature_importance`, `test_stationarity`, `decompose_series` |
| `regimes.py` | `detect_changepoints`, `backtest_event_impact` |
| `evaluation.py` | `cross_validate`, `compare_models` |
| `plotting.py` | `plot_diagnostics` |
| `data.py` | `data_segment`, `load_series` |
| `scenarios.py` | `generate_synthetic_scenario` |

All carry `@tracked()` (`../tool_tracker.py`). The agent-facing wrappers live
in [`td_one/tools/`](../../tools/README.md). Audits:
[`docs/TOOL_AUDIT.md`](../../../docs/TOOL_AUDIT.md).
Up: [`agent/`](../README.md).
