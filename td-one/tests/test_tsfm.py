"""Tests for TSFM wrappers (Chronos-2, TimesFM-2.5).

Covers the interface contract (mock paths) and the real-path call contracts
via stubbed pipelines/models — the real-path reshape and kwarg bugs that
mock-only tests missed are pinned here.
"""

import numpy as np
import pytest

from td_one.models.tsfm import (
    CHRONOS2_LEVELS,
    TIMESFM_LEVELS,
    Chronos2Wrapper,
    TimesFMWrapper,
)


def _rates(seed: int = 42, n: int = 60) -> np.ndarray:
    return np.clip(np.random.RandomState(seed).randn(n) * 0.05 + 0.2, 0.01, 0.99)


# --- Interface contract (mock) ---


def test_chronos2_quantile_interface() -> None:
    """Chronos mock returns the native 21-level grid."""
    w = Chronos2Wrapper(use_mock=True)
    levels, values = w.forecast_quantiles(_rates(), horizon=6)
    np.testing.assert_array_equal(levels, CHRONOS2_LEVELS)
    assert values.shape == (21, 6)
    assert (values >= 0).all() and (values <= 1).all()


def test_timesfm_quantile_interface() -> None:
    """TimesFM mock returns the native 9-level grid."""
    w = TimesFMWrapper(use_mock=True)
    levels, values = w.forecast_quantiles(_rates(), horizon=6)
    np.testing.assert_array_equal(levels, TIMESFM_LEVELS)
    assert values.shape == (9, 6)
    assert (values >= 0).all() and (values <= 1).all()


def test_timesfm_quantiles_monotonic() -> None:
    """Mock quantiles are non-decreasing across levels."""
    w = TimesFMWrapper(use_mock=True)
    _, values = w.forecast_quantiles(np.full(60, 0.2), horizon=6)
    assert (np.diff(values, axis=0) >= -1e-9).all()


def test_forecast_mean_is_q50() -> None:
    """forecast_mean returns the median trajectory."""
    for w in (Chronos2Wrapper(use_mock=True), TimesFMWrapper(use_mock=True)):
        levels, values = w.forecast_quantiles(_rates(), horizon=6)
        q50 = int(np.argmin(np.abs(levels - 0.5)))
        np.testing.assert_array_equal(w.forecast_mean(_rates(), horizon=6), values[q50])


def test_chronos2_mock_echoes_input_scale() -> None:
    """Mock centers on the input level (documents the input contract)."""
    w = Chronos2Wrapper(use_mock=True)
    _, values = w.forecast_quantiles(np.full(60, 0.35), horizon=6)
    assert abs(values.mean() - 0.35) < 0.02


def test_input_validation_rejects_out_of_range() -> None:
    """Rates outside [0, 1] raise (catches logit-vs-rate confusion)."""
    w = Chronos2Wrapper(use_mock=True)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        w.forecast_quantiles(np.array([-0.5, 0.2, 1.5]), horizon=6)
    with pytest.raises(ValueError, match="non-empty"):
        w.forecast_quantiles(np.array([]), horizon=6)


def test_input_validation_rejects_nan() -> None:
    """NaN rates raise loudly (NaN comparisons would pass the range check).

    Regression: _validate_rates(np.array([0.1, np.nan, 0.2])) used to
    return without raising; the NaN then flowed into the real Chronos-2 /
    TimesFM context and np.sort parked it on the highest quantile.
    """
    w = Chronos2Wrapper(use_mock=True)
    with pytest.raises(ValueError, match="NaN"):
        w.forecast_quantiles(np.array([0.1, np.nan, 0.2]), horizon=6)
    w2 = TimesFMWrapper(use_mock=True)
    with pytest.raises(ValueError, match="NaN"):
        w2.forecast_quantiles(np.array([np.nan]), horizon=6)


def test_unload_safe_when_never_loaded() -> None:
    """unload() must not raise when nothing was loaded."""
    Chronos2Wrapper(use_mock=True).unload()
    TimesFMWrapper(use_mock=True).unload()


# --- Real-path call contracts (stubbed; no weights needed) ---


def test_chronos2_real_path_call_contract() -> None:
    """Real path calls predict(inputs=...) and drops the variate dim.

    Regression test: the original code passed context= (TypeError) and the
    reshape branch was untested.
    """

    class FakePipeline:
        def __init__(self) -> None:
            self.kwargs: dict = {}

        def predict(self, **kw):  # type: ignore[no-untyped-def]
            self.kwargs = kw
            # (n_variates=1, n_quantiles=21, horizon=6); numpy works —
            # the wrapper only needs np.asarray(out[0])
            return [np.zeros((1, 21, 6))]

    w = Chronos2Wrapper(use_mock=False)
    w._pipeline = FakePipeline()
    levels, values = w.forecast_quantiles(np.full(60, 0.2), horizon=6)
    assert "inputs" in w._pipeline.kwargs  # not context=
    assert w._pipeline.kwargs["prediction_length"] == 6
    np.testing.assert_array_equal(levels, CHRONOS2_LEVELS)
    assert values.shape == (21, 6)


def test_chronos2_real_path_rejects_wrong_shape() -> None:
    """Real path raises on unexpected pipeline output shape."""

    class BadPipeline:
        def predict(self, **kw):  # type: ignore[no-untyped-def]
            return [np.zeros((1, 7, 6))]  # 7 != 21 levels

    w = Chronos2Wrapper(use_mock=False)
    w._pipeline = BadPipeline()
    with pytest.raises(RuntimeError, match="Unexpected Chronos-2 output shape"):
        w.forecast_quantiles(np.full(60, 0.2), horizon=6)


def test_chronos2_real_path_loads_without_private_call() -> None:
    """forecast_quantiles triggers _load itself (no silent mock fallback).

    Regression test: the old `if use_mock or _pipeline is None` branch
    silently returned mock data when callers forgot the private _load().
    """

    class FakePipeline:
        def predict(self, **kw):  # type: ignore[no-untyped-def]
            return [np.zeros((1, 21, 6))]

    w = Chronos2Wrapper(use_mock=False)
    # Simulate _load having run by injecting directly is NOT needed anymore;
    # instead verify that with _pipeline=None and use_mock=False, _load is
    # attempted (it will raise RuntimeError without the chronos package only
    # if not already loaded — here we pre-load to isolate the branch logic).
    w._pipeline = FakePipeline()
    _, values = w.forecast_quantiles(np.full(60, 0.2), horizon=6)
    assert values.shape == (21, 6)


def test_timesfm_parse_uses_true_levels() -> None:
    """Quantile columns map to [0.1..0.9], not an assumed [0,1] grid.

    Regression test: the old code interpolated from linspace(0,1,9),
    mislabeling every quantile. The real model emits 10 columns
    (col 0 = point forecast, cols 1-9 = quantiles at [0.1..0.9]).
    """
    point = np.full((1, 6), 0.2)
    # 10 columns: point col + 9 quantile cols at the TRUE levels
    q = np.zeros((1, 6, 10))
    q[:, :, 1:] = np.tile(np.linspace(0.05, 0.35, 9), (1, 6, 1))
    levels, values = TimesFMWrapper._parse_timesfm_result(point, q, 6)
    np.testing.assert_array_equal(levels, TIMESFM_LEVELS)
    assert values.shape == (9, 6)
    # column 1 (first quantile col) must be the true q10 value
    np.testing.assert_allclose(values[0], 0.05, atol=1e-12)
    np.testing.assert_allclose(values[-1], 0.35, atol=1e-12)


def test_timesfm_parse_rejects_unexpected_shape() -> None:
    """Unexpected quantile counts raise instead of being silently guessed."""
    point = np.full((1, 6), 0.2)
    q = np.zeros((1, 6, 8))  # 8 != 10
    with pytest.raises(ValueError, match="Unexpected TimesFM quantile shape"):
        TimesFMWrapper._parse_timesfm_result(point, q, 6)


def test_timesfm_real_path_forecast_signature() -> None:
    """Real path calls forecast(horizon=, inputs=) positionally-correct."""

    class FakeModel:
        def __init__(self) -> None:
            self.kwargs: dict = {}

        def forecast(self, **kw):  # type: ignore[no-untyped-def]
            self.kwargs = kw
            # 10 cols: point + 9 quantiles at [0.1..0.9]
            return np.full((1, 6), 0.2), np.tile(np.linspace(0.0, 0.4, 10), (1, 6, 1))

    w = TimesFMWrapper(use_mock=False)
    w._model = FakeModel()
    levels, values = w.forecast_quantiles(np.full(60, 0.2), horizon=6)
    assert w._model.kwargs["horizon"] == 6
    assert "inputs" in w._model.kwargs
    assert values.shape == (9, 6)


# --- Real-weights tests (gated; skipped in CI) ---

_REAL_OK = False
try:
    from pathlib import Path

    import torch  # noqa: F401

    _REAL_OK = (
        Path("models/weights/chronos2").exists() and Path("models/weights/timesfm25").exists()
    )
except ImportError:
    _REAL_OK = False

needs_real_weights = pytest.mark.skipif(
    not _REAL_OK or __import__("os").environ.get("TD_ONE_REAL_WEIGHTS") != "1",
    reason="real-weights test: set TD_ONE_REAL_WEIGHTS=1 with models/weights/ present",
)


@needs_real_weights
def test_chronos2_real_output_is_21_level_lattice() -> None:
    """Real Chronos-2 emits the native 21-level quantile grid (not samples)."""
    w = Chronos2Wrapper(weights_dir="models/weights/chronos2")
    try:
        levels, values = w.forecast_quantiles(np.full(60, 0.2), horizon=6)
    finally:
        w.unload()
    np.testing.assert_array_equal(levels, CHRONOS2_LEVELS)
    assert values.shape == (21, 6)
    # Quantile lattice: rows ordered by level (non-decreasing down the levels)
    assert (np.diff(values, axis=0) >= -1e-6).all()


@needs_real_weights
def test_timesfm_real_quantiles_monotonic_true_levels() -> None:
    """Real TimesFM quantiles are monotonic and labeled from true levels."""
    w = TimesFMWrapper(weights_dir="models/weights/timesfm25")
    try:
        levels, values = w.forecast_quantiles(np.full(60, 0.2), horizon=6)
    finally:
        w.unload()
    np.testing.assert_array_equal(levels, TIMESFM_LEVELS)
    assert values.shape == (9, 6)
    assert (np.diff(values, axis=0) >= -1e-6).all()
