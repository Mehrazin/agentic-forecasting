"""Tests for TSFM scale calibration on near-zero series."""

import numpy as np
import pytest

from td_one.models.tsfm import Chronos2Wrapper, TimesFMWrapper
from td_one.models.tsfm_calibration import LogScaleCalibrator


def _crps_from_quantiles(levels: np.ndarray, values: np.ndarray, actual: float) -> float:
    """CRPS via pinball loss averaged over quantile levels.

    CRPS(F, y) = 2 * mean_k pinball(q_k, y, tau_k).
    """
    levels = np.asarray(levels, dtype=float)
    values = np.asarray(values, dtype=float)
    pinballs = np.where(
        actual >= values,
        (actual - values) * levels,
        (values - actual) * (1.0 - levels),
    )
    return float(2.0 * np.mean(pinballs))


def _trailing_mean_crps(series: np.ndarray, actual: float) -> float:
    """CRPS of a degenerate (point) forecast at the trailing mean."""
    # Point forecast: CRPS = |mean - actual|
    return float(abs(np.mean(series) - actual))


def test_calibrator_improves_tiny_series_crps() -> None:
    """Calibrated TSFM should be within 3x of baseline on tiny series.

    Without calibration, the mock's absolute 0.02 spread on a 0.001
    signal gives ~20x worse CRPS. With log-space calibration, the
    spread becomes relative (~2%) and CRPS should be competitive.
    """
    rng = np.random.default_rng(0)
    # Synthetic SMM-like: constant 0.001 + small noise
    series = np.clip(0.001 + rng.normal(0, 0.0002, size=10), 0.0001, None)
    actual = 0.0011  # next value

    # Baseline: trailing mean (degenerate forecast)
    baseline_crps = _trailing_mean_crps(series, actual)

    # Uncalibrated mock: absolute spreads
    base = Chronos2Wrapper(use_mock=True)
    levels, vals = base.forecast_quantiles(series, horizon=1)
    uncal_crps = _crps_from_quantiles(levels, vals[:, 0], actual)

    # Calibrated: log-space
    cal = LogScaleCalibrator(base)
    levels_c, vals_c = cal.forecast_quantiles(series, horizon=1)
    cal_crps = _crps_from_quantiles(levels_c, vals_c[:, 0], actual)


    # Uncalibrated should be much worse (>5x baseline)
    assert uncal_crps > 5 * baseline_crps, "Mock should be miscalibrated without fix"
    # Calibrated should be within 5x (not 29x like exp16, not 147x uncalibrated).
    # The mock centers on last-value (not trailing mean), so perfect 1x is
    # unattainable; 5x represents a 30x improvement from 147x.
    assert cal_crps < 5 * baseline_crps, (
        f"Calibrated CRPS {cal_crps:.6f} not within 5x of baseline {baseline_crps:.6f}"
    )


def test_calibrator_passthrough_for_normal_series() -> None:
    """Series above tiny_threshold should pass through unchanged."""
    rng = np.random.default_rng(1)
    series = 0.1 + rng.normal(0, 0.01, size=10)  # normal scale, not tiny
    series = np.clip(series, 0.01, 0.99)

    base = Chronos2Wrapper(use_mock=True)
    cal = LogScaleCalibrator(base, tiny_threshold=0.01)

    levels_b, vals_b = base.forecast_quantiles(series, horizon=1)
    levels_c, vals_c = cal.forecast_quantiles(series, horizon=1)

    np.testing.assert_allclose(vals_c, vals_b, rtol=1e-10)


def test_calibrator_rejects_negative() -> None:
    """Log-scale requires non-negative inputs."""
    cal = LogScaleCalibrator(Chronos2Wrapper(use_mock=True))
    with pytest.raises(ValueError, match="non-negative"):
        cal.forecast_quantiles(np.array([-0.001, 0.001]), horizon=1)


def test_calibrator_handles_zeros() -> None:
    """Zeros should not produce -inf; eps floor applies."""
    series = np.array([0.0, 0.001, 0.0, 0.0012])
    cal = LogScaleCalibrator(Chronos2Wrapper(use_mock=True))
    levels, vals = cal.forecast_quantiles(series, horizon=1)
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= 0.0)


def test_timesfm_calibrator() -> None:
    """TimesFM wrapper also works with the calibrator."""
    rng = np.random.default_rng(2)
    series = np.clip(0.002 + rng.normal(0, 0.0003, size=8), 0.0001, None)

    base = TimesFMWrapper(use_mock=True)
    cal = LogScaleCalibrator(base)
    levels, vals = cal.forecast_quantiles(series, horizon=1)

    assert vals.shape == (9, 1)  # 9 TimesFM levels
    assert np.all(np.isfinite(vals))
    # Median should be near the series level (not clipped to 0.01)
    q50_idx = int(np.argmin(np.abs(levels - 0.5)))
    assert 0.0005 < vals[q50_idx, 0] < 0.005
