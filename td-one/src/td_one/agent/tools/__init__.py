"""Reusable analysis tools for the TD-One forecast agent.

Each tool is a pure function with Pydantic-validated inputs, PIT-safety
checks where temporal data is involved, and JSON-serializable outputs so
results can flow back into the ReAct trace as observations.

Tool inventory:
- analysis: analyze_residuals, feature_importance, test_stationarity,
  decompose_series
- regimes: detect_changepoints, backtest_event_impact
- evaluation: cross_validate, compare_models
- plotting: plot_diagnostics
- scenarios: generate_synthetic_scenario
"""

from td_one.agent.tools.analysis import (
    analyze_residuals,
    decompose_series,
    feature_importance,
    test_stationarity,
)
from td_one.agent.tools.evaluation import compare_models, cross_validate
from td_one.agent.tools.plotting import plot_diagnostics
from td_one.agent.tools.regimes import backtest_event_impact, detect_changepoints
from td_one.agent.tools.scenarios import generate_synthetic_scenario

__all__ = [
    "analyze_residuals",
    "backtest_event_impact",
    "compare_models",
    "cross_validate",
    "decompose_series",
    "detect_changepoints",
    "feature_importance",
    "generate_synthetic_scenario",
    "plot_diagnostics",
    "test_stationarity",
]
