"""Reusable analysis tools for the TD-One forecast agent.

Each tool is a pure function with Pydantic-validated inputs, PIT-safety
checks where temporal data is involved, and JSON-serializable outputs so
results can flow back into the ReAct trace as observations.

Tool inventory:
- analysis: analyze_residuals, feature_importance, test_stationarity,
  decompose_series
- data: load_series, naive_forecasts, series_features
- regimes: detect_changepoints, backtest_event_impact, regime_probabilities
- evaluation: cross_validate, compare_models, combine_quantiles

Retired from the agent namespace (modules kept for manual use):
- plotting.plot_diagnostics — zero call sites, slowest tool
- scenarios.generate_synthetic_scenario — zero call sites
"""

from td_one.agent.tools.analysis import (
    analyze_residuals,
    decompose_series,
    feature_importance,
    test_stationarity,
)
from td_one.agent.tools.data import load_series, naive_forecasts, series_features
from td_one.agent.tools.evaluation import (
    combine_quantiles,
    compare_models,
    cross_validate,
)
from td_one.agent.tools.regimes import (
    backtest_event_impact,
    detect_changepoints,
    regime_probabilities,
)

__all__ = [
    "analyze_residuals",
    "backtest_event_impact",
    "combine_quantiles",
    "compare_models",
    "cross_validate",
    "decompose_series",
    "detect_changepoints",
    "feature_importance",
    "load_series",
    "naive_forecasts",
    "regime_probabilities",
    "series_features",
    "test_stationarity",
]
