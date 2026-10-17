"""TSFM wrappers for zero-shot time-series forecasting.

- Chronos2Wrapper: amazon/chronos-2 (120M, best calibration)
- TimesFMWrapper: google/timesfm-2.5-200m-pytorch (200M)

Both support `use_mock=True` for testing without downloads.
Real inference uses the `chronos-forecasting` and `timesfm>=3` packages.

Interface contract (both wrappers, mock and real):
- Input: historical rates in [0, 1] (validated; values outside raise).
- `forecast_quantiles()` returns (levels, values):
    - Chronos-2: 21 native levels [0.01, 0.05, 0.10, ..., 0.99]
    - TimesFM-2.5: 9 native levels [0.1, 0.2, ..., 0.9]
- `forecast_mean()` returns the median (q50) trajectory, shape (horizon,).

Mock/real divergence (documented, not hidden):
- Mocks are interface-only: they center on the last input value and are NOT
  statistical surrogates. Mock Chronos returns i.i.d. draws; real Chronos
  returns a fixed 21-level quantile lattice. Do not compute sample statistics
  on real Chronos output.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Native quantile grids (from model configs; do not guess)
CHRONOS2_LEVELS = np.array(
    [
        0.01,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.99,
    ]
)
TIMESFM_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def _validate_rates(series: np.ndarray) -> np.ndarray:
    """Validate input is rates in [0, 1]; return float array."""
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        raise ValueError("series must be non-empty")
    # NaN comparisons are all False, so the range check below would pass
    # NaN silently — reject explicitly before it reaches a model.
    if bool(np.isnan(arr).any()):
        raise ValueError("series must not contain NaN")
    if bool((arr < 0.0).any() or (arr > 1.0).any()):
        raise ValueError(
            f"series must be rates in [0, 1]; got range [{arr.min():.3f}, {arr.max():.3f}]"
        )
    return arr


class Chronos2Wrapper:
    """Chronos-2 zero-shot forecaster (amazon/chronos-2)."""

    model_id = "amazon/chronos-2"

    def __init__(
        self,
        prediction_length: int = 6,
        use_mock: bool = False,
        weights_dir: str | None = None,
    ) -> None:
        """Initialize.

        Args:
            prediction_length: Forecast horizon (default 6 months).
            use_mock: If True, use mock forecasts (for tests).
            weights_dir: Local snapshot dir (from bootstrap_models.py).
                If None, downloads from HF Hub on first real use.

        """
        self.prediction_length = prediction_length
        self.use_mock = use_mock
        self.weights_dir = weights_dir
        self._pipeline: Any = None

    def _load(self) -> None:
        """Lazy-load the Chronos-2 pipeline (no-op if loaded or mock)."""
        if self._pipeline is not None or self.use_mock:
            return
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError as e:
            raise RuntimeError(
                "chronos-forecasting not installed. Run: uv pip install chronos-forecasting"
            ) from e

        pretrained = self.weights_dir or self.model_id
        self._pipeline = Chronos2Pipeline.from_pretrained(
            pretrained,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    def unload(self) -> None:
        """Release model weights (frees RAM for the next model)."""
        self._pipeline = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def forecast_quantiles(
        self,
        series: np.ndarray,
        horizon: int | None = None,
        validate_input: bool = True,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast quantiles on the native 21-level grid.

        Args:
            series: Historical rates in [0, 1].
            horizon: Forecast steps (default: prediction_length).
            validate_input: If False, skip the [0, 1] range check (for
                log-space inputs from LogScaleCalibrator).
            seed: If set, seed torch RNG before predict for reproducibility.

        Returns:
            (levels, values): levels shape (21,), values shape (21, horizon).
            Mock returns i.i.d. draws reshaped to the same interface —
            see module docstring for the divergence caveat.

        """
        arr = np.asarray(series, dtype=float)
        if validate_input:
            arr = _validate_rates(arr)
        elif arr.size == 0:
            raise ValueError("series must be non-empty")
        elif bool(np.isnan(arr).any()):
            raise ValueError("series must not contain NaN")
        h = horizon or self.prediction_length
        if self.use_mock:
            rng = np.random.default_rng(42)
            if validate_input:
                last = float(np.clip(arr[-1], 0.01, 0.99))
                draws = rng.normal(loc=last, scale=0.02, size=(len(CHRONOS2_LEVELS), h))
                return CHRONOS2_LEVELS.copy(), np.clip(draws, 0.0, 1.0)
            else:
                # Log-space (or other transformed) inputs: don't clip, use
                # last value directly. Fixed 0.02 scale becomes relative
                # after exponentiation.
                last = float(arr[-1])
                draws = rng.normal(loc=last, scale=0.02, size=(len(CHRONOS2_LEVELS), h))
                return CHRONOS2_LEVELS.copy(), draws
        # Real path: always load (never silently fall back to mock)
        self._load()
        assert self._pipeline is not None
        try:
            import torch

            if seed is not None:
                torch.manual_seed(seed)
            context: Any = torch.tensor(arr, dtype=torch.float32)
        except ImportError:
            # torch absent: only viable when _pipeline was stubbed (tests);
            # the real Chronos2Pipeline requires torch.
            context = arr
        out = self._pipeline.predict(
            inputs=[context],
            prediction_length=h,
            batch_size=32,
        )
        vals = np.asarray(out[0])
        if vals.ndim == 3:
            vals = vals[0]  # drop variate dim -> (21, h)
        if vals.shape != (len(CHRONOS2_LEVELS), h):
            raise RuntimeError(
                f"Unexpected Chronos-2 output shape {vals.shape}; "
                f"expected ({len(CHRONOS2_LEVELS)}, {h})"
            )
        # Skip the [0, 1] output clip for transformed (e.g. log-space)
        # inputs: the caller (e.g. LogScaleCalibrator) owns the
        # back-transform and clips afterwards.
        if validate_input:
            vals = np.clip(vals, 0.0, 1.0)
        return CHRONOS2_LEVELS.copy(), vals

    def forecast_mean(
        self,
        series: np.ndarray,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Median (q50) trajectory, shape (horizon,)."""
        levels, values = self.forecast_quantiles(series, horizon)
        q50 = int(np.argmin(np.abs(levels - 0.5)))
        return np.asarray(values[q50])


class TimesFMWrapper:
    """TimesFM-2.5 zero-shot forecaster."""

    model_id = "google/timesfm-2.5-200m-pytorch"

    def __init__(
        self,
        prediction_length: int = 6,
        use_mock: bool = False,
        weights_dir: str | None = None,
    ) -> None:
        """Initialize.

        Args:
            prediction_length: Forecast horizon.
            use_mock: If True, use mock forecasts.
            weights_dir: Local snapshot dir (from bootstrap_models.py).

        """
        self.prediction_length = prediction_length
        self.use_mock = use_mock
        self.weights_dir = weights_dir
        self._model: Any = None

    def _load(self) -> None:
        """Lazy-load the TimesFM-2.5 model (timesfm>=3.x API)."""
        if self._model is not None or self.use_mock:
            return
        try:
            from timesfm.configs import ForecastConfig
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
        except ImportError as e:
            raise RuntimeError("timesfm>=3 not installed. Run: uv pip install timesfm") from e

        repo = self.weights_dir or self.model_id
        # from_pretrained accepts a local dir (uses model.safetensors inside)
        model = TimesFM_2p5_200M_torch.from_pretrained(repo)
        model.compile(
            ForecastConfig(
                max_context=512,
                max_horizon=64,
                normalize_inputs=True,
                per_core_batch_size=32,
            )
        )
        self._model = model

    def unload(self) -> None:
        """Release model weights (frees RAM for the next model)."""
        self._model = None
        try:
            import gc

            gc.collect()
        except ImportError:
            pass

    def forecast_quantiles(
        self,
        series: np.ndarray,
        horizon: int | None = None,
        validate_input: bool = True,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast quantiles on the native 9-level grid.

        Args:
            series: Historical rates in [0, 1].
            horizon: Forecast steps.
            validate_input: If False, skip the [0, 1] range check (for
                log-space inputs from LogScaleCalibrator).
            seed: If set, seed torch RNG before forecast for reproducibility.

        Returns:
            (levels, values): levels shape (9,) = [0.1..0.9],
            values shape (9, horizon).

        """
        arr = np.asarray(series, dtype=float)
        if validate_input:
            arr = _validate_rates(arr)
        elif arr.size == 0:
            raise ValueError("series must be non-empty")
        elif bool(np.isnan(arr).any()):
            raise ValueError("series must not contain NaN")
        h = horizon or self.prediction_length
        if self.use_mock:
            return TIMESFM_LEVELS.copy(), self._mock_quantiles(arr, h, validate_input)
        self._load()
        assert self._model is not None
        if seed is not None:
            try:
                import torch

                torch.manual_seed(seed)
            except ImportError:
                pass
        # forecast(horizon, inputs) -> (point, quantiles); quantiles shape
        # (batch, horizon, 9) at TIMESFM_LEVELS
        point, q = self._model.forecast(horizon=h, inputs=[arr])
        return self._parse_timesfm_result(point, q, h, validate_input)

    @staticmethod
    def _parse_timesfm_result(
        point: object, q: object, h: int, validate_input: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse timesfm forecast output into (levels, values).

        The compiled model returns 10 columns: column 0 is the point
        forecast, columns 1-9 are the quantiles at TIMESFM_LEVELS
        ([0.1..0.9]). Verified against the installed timesfm 3.x
        `_compiled_decode` (index 5 = median; 1-4 lower, 6-9 upper).
        """
        q_arr = np.asarray(q)
        # (batch, horizon, 10) -> (horizon, 10)
        if q_arr.ndim == 3:
            q_arr = q_arr[0]
        if q_arr.shape != (h, len(TIMESFM_LEVELS) + 1):
            raise ValueError(
                f"Unexpected TimesFM quantile shape {q_arr.shape}; "
                f"expected ({h}, {len(TIMESFM_LEVELS) + 1}) "
                "(point col + 9 quantile cols)"
            )
        # Drop column 0 (point forecast); keep the 9 quantile columns.
        q_vals = q_arr[:, 1:]
        # Transpose to (levels, horizon) for a uniform interface.
        # Skip the [0, 1] output clip for transformed (e.g. log-space)
        # inputs: the caller owns the back-transform and clips afterwards.
        if validate_input:
            q_vals = np.clip(q_vals, 0.0, 1.0)
        return TIMESFM_LEVELS.copy(), q_vals.T

    def forecast_mean(
        self,
        series: np.ndarray,
        horizon: int | None = None,
    ) -> np.ndarray:
        """Median (q50) trajectory, shape (horizon,)."""
        levels, values = self.forecast_quantiles(series, horizon)
        q50 = int(np.argmin(np.abs(levels - 0.5)))
        return np.asarray(values[q50])

    @staticmethod
    def _mock_quantiles(arr: np.ndarray, h: int, validate_input: bool = True) -> np.ndarray:
        """Mock quantiles: symmetric spreads around last value."""
        # Log-space: use last value directly (no clipping).
        last = float(np.clip(arr[-1], 0.01, 0.99)) if validate_input else float(arr[-1])
        # 9 levels matching TIMESFM_LEVELS: spreads widen away from median
        spreads = np.array([0.05, 0.035, 0.02, 0.01, 0.0, 0.01, 0.02, 0.035, 0.05])
        vals = np.tile(spreads[:, None], (1, h))
        # lower levels subtract, upper add; median = last
        signs = np.array([-1, -1, -1, -1, 0, 1, 1, 1, 1])
        result = last + signs[:, None] * vals
        if validate_input:
            result = np.clip(result, 0.0, 1.0)
        return np.asarray(result)
