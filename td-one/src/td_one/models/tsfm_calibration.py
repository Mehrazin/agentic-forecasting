"""Scale-aware calibration for TSFM forecasts on near-zero series.

Problem (exp16): Chronos-2 on Freddie Mac SMM data (values 0.0003-0.003)
produces correct medians but 29x worse CRPS than a trailing mean. The
TSFM's predictive spreads are calibrated in absolute terms — a "normal"
spread of 0.01 is 10x the signal when the signal is 0.001.

Solution: For positive series with small magnitudes, forecast in log-space
(spreads become relative/multiplicative), then exponentiate quantiles back.
Quantiles transform correctly under monotonic transforms, so this is
principled, not a hack.

Usage:
    from td_one.models.tsfm import Chronos2Wrapper
    from td_one.models.tsfm_calibration import LogScaleCalibrator

    base = Chronos2Wrapper(use_mock=False)
    cal = LogScaleCalibrator(base)  # auto-detects tiny series
    levels, vals = cal.forecast_quantiles(smm_series)
"""

from __future__ import annotations

import numpy as np

#: Below this max value, a series is considered "tiny" and log-transformed.
#: Above it, the calibrator passes through unchanged.
TINY_THRESHOLD = 0.01

#: Floor for log transform to avoid log(0) = -inf.
LOG_EPS = 1e-6


class LogScaleCalibrator:
    """Wraps a TSFM wrapper with log-space forecasting for near-zero series.

    Args:
        wrapper: A Chronos2Wrapper or TimesFMWrapper (or any object with
            forecast_quantiles(series, horizon) -> (levels, values)).
        tiny_threshold: Max series value below which log-transform applies.
        eps: Floor for log transform (avoids log(0)).

    The wrapper's forecast_quantiles must accept log-space inputs (negative
    values). Both Chronos2Wrapper and TimesFMWrapper support this via
    validate_input=False.

    """

    def __init__(
        self,
        wrapper: object,
        tiny_threshold: float = TINY_THRESHOLD,
        eps: float = LOG_EPS,
    ) -> None:
        self.wrapper = wrapper
        self.tiny_threshold = tiny_threshold
        self.eps = eps

    def forecast_quantiles(
        self,
        series: np.ndarray,
        horizon: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast quantiles, using log-space for tiny series.

        Args:
            series: Non-negative rates. If max(series) < tiny_threshold,
                forecasts in log-space; otherwise passes through.
            horizon: Forecast steps (passed to inner wrapper).

        Returns:
            (levels, values): Same format as the inner wrapper, values
            in original scale, clipped to [0, 1].

        """
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            raise ValueError("series must be non-empty")
        if bool(np.isnan(arr).any()):
            raise ValueError("series must not contain NaN")
        if bool((arr < 0.0).any()):
            raise ValueError("log-scale calibration requires non-negative series")

        # Not tiny: pass through unchanged.
        if float(arr.max()) >= self.tiny_threshold:
            fn = self.wrapper.forecast_quantiles  # type: ignore[attr-defined]
            result: tuple[np.ndarray, np.ndarray] = fn(arr, horizon)
            return result

        # Tiny: forecast in log-space.
        log_arr = np.log(np.maximum(arr, self.eps))
        fn = self.wrapper.forecast_quantiles  # type: ignore[attr-defined]
        # Bypass [0,1] validation since log values are negative.
        levels, log_vals = fn(log_arr, horizon, validate_input=False)
        # Exponentiate back to original scale; quantiles are preserved
        # under monotonic transforms.
        vals = np.clip(np.exp(np.asarray(log_vals, dtype=float)), 0.0, 1.0)
        # Ensure monotonicity across levels (mocks may produce non-monotonic
        # draws; real TSFMs are monotonic but belt-and-braces is cheap).
        # Sort each horizon column by the level order.
        levels_arr = np.asarray(levels, dtype=float)
        order = np.argsort(levels_arr, kind="stable")
        vals_sorted = np.empty_like(vals)
        for h_idx in range(vals.shape[1]):
            vals_sorted[order, h_idx] = np.sort(vals[:, h_idx], kind="stable")
        return levels_arr, vals_sorted

    def forecast_mean(
        self,
        series: np.ndarray,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Median (q50) trajectory in original scale."""
        levels, values = self.forecast_quantiles(series, horizon)
        q50 = int(np.argmin(np.abs(levels - 0.5)))
        return np.asarray(values[q50])
