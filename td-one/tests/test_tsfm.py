"""Tests for TSFM wrappers (Chronos-2, TimesFM-2.5).

These test the interface, not the actual models (which require downloads).
"""

import numpy as np
import pytest

from td_one.models.tsfm import Chronos2Wrapper, TimesFMWrapper


def test_chronos2_interface() -> None:
    """Chronos-2 wrapper has correct interface."""
    wrapper = Chronos2Wrapper()
    assert wrapper.model_id == "amazon/chronos-2"
    assert wrapper.prediction_length == 6


def test_timesfm_interface() -> None:
    """TimesFM wrapper has correct interface."""
    wrapper = TimesFMWrapper()
    assert "timesfm" in wrapper.model_id.lower()
    assert wrapper.prediction_length == 6


def test_chronos2_forecast_shape() -> None:
    """Forecast returns (n_samples, horizon) with mock."""
    wrapper = Chronos2Wrapper(use_mock=True)
    series = np.random.RandomState(42).randn(60) * 0.05 + 0.2
    forecast = wrapper.forecast(series, horizon=6, n_samples=100)
    assert forecast.shape == (100, 6)
    # Rates should be in [0, 1] (clipped)
    assert (forecast >= 0).all()
    assert (forecast <= 1).all()


def test_timesfm_forecast_shape() -> None:
    """TimesFM forecast returns quantiles with mock."""
    wrapper = TimesFMWrapper(use_mock=True)
    series = np.random.RandomState(42).randn(60) * 0.05 + 0.2
    mean, quantiles = wrapper.forecast_quantiles(series, horizon=6)
    assert mean.shape == (6,)
    assert quantiles.shape == (6, 10)  # 10 quantiles (q10..q90)
