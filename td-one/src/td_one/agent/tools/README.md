# `agent/tools/` — the 13 agent tools

The granular tool functions the agents call through the ReAct v2 loop. Each is
a pure function with Pydantic-validated inputs, PIT-safety checks where temporal
data is involved, and JSON-serializable outputs (so results flow back into the
trace as observations). Every tool carries `@tracked()`
(`../tool_tracker.py`); runtime data lands in `docs/TOOL_USAGE_DATA.md` and
drives `docs/TOOL_OPTIMIZATION.md`.

This is the **final namespace** (13 tools, post-exp50 audit). Two modules are
kept on disk but **not** exposed to agents: `plotting.py`
(`plot_diagnostics`, 0 real invocations) and `scenarios.py`
(`generate_synthetic_scenario`, 0 real invocations).

| Module | Tools in the agent namespace |
|---|---|
| `analysis.py` | `analyze_residuals`, `feature_importance`, `test_stationarity`, `decompose_series` |
| `regimes.py` | `detect_changepoints`, `backtest_event_impact`, `regime_probabilities` |
| `evaluation.py` | `cross_validate`, `compare_models`, `combine_quantiles` |
| `data.py` | `load_series`, `naive_forecasts`, `series_features` |

| Module | Kept on disk, not in the namespace |
|---|---|
| `plotting.py` | `plot_diagnostics` |
| `scenarios.py` | `generate_synthetic_scenario` |

The v1 action tools (`data_segment`, `evidence`, `forecasting`) live in
[`td_one/tools/`](../../tools/README.md) — a separate, coarser layer the ReAct
loop dispatches by name. Audits:
[`docs/TOOL_AUDIT.md`](../../../docs/TOOL_AUDIT.md).
Up: [`agent/`](../README.md).
