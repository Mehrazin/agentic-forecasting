"""Counterfactual scenario tool for what-if analysis.

``generate_synthetic_scenario`` applies a hypothetical shock to a series so
the agent can reason about causal impact without touching the real DGP.
"""

from __future__ import annotations

from td_one.agent.tool_tracker import tracked

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

ShockType = Literal["level_shift", "trend_change", "volatility_spike", "pulse"]


class ScenarioResult(BaseModel):
    """Structured output of :func:`generate_synthetic_scenario`."""

    shocked_series: list[float]
    shock_type: str
    magnitude: float
    start_index: int
    duration: int
    mean_shift: float = Field(description="Mean(shocked) - mean(base) over the window")


@tracked()
def generate_synthetic_scenario(
    base_series: Any,
    shock_type: ShockType,
    magnitude: float,
    start_index: int | None = None,
    duration: int = 6,
    seed: int = 0,
) -> ScenarioResult:
    """Apply a hypothetical shock to a series (counterfactual).

    Shock semantics (additive on the series scale):
    - ``level_shift``: +magnitude for ``duration`` steps, then persists.
    - ``trend_change``: slope of magnitude/step accumulates over ``duration``.
    - ``volatility_spike``: Gaussian noise with std=|magnitude| for ``duration``.
    - ``pulse``: +magnitude for ``duration`` steps, then reverts.

    Args:
        base_series: 1-D numeric base series.
        shock_type: One of the four shock types.
        magnitude: Shock size in series units (std for volatility_spike).
        start_index: Where the shock starts (default: 3/4 through).
        duration: Shock length in steps.
        seed: RNG seed (only used for volatility_spike).

    Returns:
        ScenarioResult with the shocked series and the realized mean shift.

    """
    x = np.asarray(base_series, dtype=float).ravel()
    if x.size < 12:
        raise ValueError(f"base_series needs at least 12 observations, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("base_series contains NaN or inf")
    if duration < 1 or duration > x.size:
        raise ValueError(f"duration must be in [1, {x.size}]")

    start = start_index if start_index is not None else (3 * x.size) // 4
    if not 0 <= start < x.size:
        raise ValueError(f"start_index {start} out of range [0, {x.size})")
    end = min(x.size, start + duration)

    y = x.copy()
    rng = np.random.default_rng(seed)
    if shock_type == "level_shift":
        y[start:] += magnitude
    elif shock_type == "trend_change":
        ramp = np.arange(1, end - start + 1, dtype=float) * magnitude
        y[start:end] += ramp
        y[end:] += ramp[-1] if ramp.size else 0.0
    elif shock_type == "volatility_spike":
        y[start:end] += rng.normal(0.0, abs(magnitude), size=end - start)
    elif shock_type == "pulse":
        y[start:end] += magnitude
    else:  # pragma: no cover - Literal guards this
        raise ValueError(f"unknown shock_type: {shock_type}")

    window = y[start:end]
    base_window = x[start:end]
    return ScenarioResult(
        shocked_series=[float(v) for v in y],
        shock_type=shock_type,
        magnitude=float(magnitude),
        start_index=start,
        duration=end - start,
        mean_shift=float(np.mean(window) - np.mean(base_window)),
    )
