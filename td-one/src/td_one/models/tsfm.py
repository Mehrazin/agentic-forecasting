"""TSFM wrappers for zero-shot time-series forecasting.

- Chronos2Wrapper: amazon/chronos-2 (120M, best calibration)
- TimesFMWrapper: google/timesfm-2.5-200m-pytorch (200M)

Both support `use_mock=True` for testing without downloads.
Real inference uses the `chronos-forecasting` and `timesfm` packages.

Per survey: Darts unified API is an alternative; these wrappers provide
a minimal interface focused on our use case (univariate rate forecasting).
"""

from __future__ import annotations

import numpy as np


class Chronos2Wrapper:
    """Chronos-2 zero-shot forecaster (amazon/chronos-2)."""

    model_id = "amazon/chronos-2"

    def __init__(
        self,
        prediction_length: int = 6,
        use_mock: bool = False,
    ) -> None:
        """Initialize.

        Args:
            prediction_length: Forecast horizon (default 6 months).
            use_mock: If True, use mock forecasts (for tests).
        """
        self.prediction_length = prediction_length
        self.use_mock = use_mock
        self._pipeline: object | None = None

    def _load(self, weights_dir: str | None = None) -> None:
        """Lazy-load the Chronos-2 pipeline.

        Args:
            weights_dir: Local snapshot dir (from bootstrap_models.py).
                If None, downloads from HF Hub on first use.
        """
        if self._pipeline is not None or self.use_mock:
            return
        try:
            import torch
            from chronos import Chronos2Pipeline

            pretrained = weights_dir or self.model_id
            self._pipeline = Chronos2Pipeline.from_pretrained(
                pretrained,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        except ImportError as e:
            raise RuntimeError(
                "chronos-forecasting not installed. "
                "Run: uv pip install chronos-forecasting"
            ) from e

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

    def forecast(
        self,
        series: np.ndarray,
        horizon: int | None = None,
        n_samples: int = 1000,
    ) -> np.ndarray:
        """Generate probabilistic forecast.

        Args:
            series: Historical values (e.g. logit-transformed rates).
            horizon: Forecast steps (default: prediction_length).
            n_samples: Number of sample trajectories.

        Returns:
            Array shape (n_samples, horizon) of sampled trajectories,
            clipped to [0, 1] for rates.
        """
        h = horizon or self.prediction_length
        if self.use_mock or self._pipeline is None:
            # Mock: random walk around last value
            rng = np.random.default_rng(42)
            last = float(series[-1]) if len(series) > 0 else 0.2
            # Clip to valid rate range
            last = float(np.clip(last, 0.01, 0.99))
            samples = rng.normal(
                loc=last, scale=0.02, size=(n_samples, h)
            )
            return np.clip(samples, 0.0, 1.0)
        # Real inference
        self._load()
        assert self._pipeline is not None
        import torch

        # Type narrowing for mypy
        pipeline: object = self._pipeline
        context = torch.tensor(series, dtype=torch.float32)
        # Chronos2Pipeline.predict(inputs, prediction_length, ...) -> list[Tensor]
        # Each tensor shape: (n_variates, num_samples, prediction_length)
        predict_fn = getattr(pipeline, "predict")
        out = predict_fn(
            inputs=[context],
            prediction_length=h,
            batch_size=32,
        )
        arr = np.asarray(out[0])  # (1, n_samples, h) or (n_samples, h)
        if arr.ndim == 3:
            arr = arr[0]  # drop variate dim -> (n_samples, h)
        return np.clip(arr, 0.0, 1.0)


class TimesFMWrapper:
    """TimesFM-2.5 zero-shot forecaster."""

    model_id = "google/timesfm-2.5-200m-pytorch"

    def __init__(
        self,
        prediction_length: int = 6,
        use_mock: bool = False,
    ) -> None:
        """Initialize.

        Args:
            prediction_length: Forecast horizon.
            use_mock: If True, use mock forecasts.
        """
        self.prediction_length = prediction_length
        self.use_mock = use_mock
        self._model: object | None = None

    def _load(self, weights_dir: str | None = None) -> None:
        """Lazy-load the TimesFM-2.5 model (timesfm>=3.x API).

        Args:
            weights_dir: Local snapshot dir (from bootstrap_models.py).
                If None, downloads from the HF Hub on first use.
        """
        if self._model is not None or self.use_mock:
            return
        try:
            from timesfm.configs import ForecastConfig
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
        except ImportError as e:
            raise RuntimeError(
                "timesfm>=3 not installed. Run: uv pip install timesfm"
            ) from e

        repo = weights_dir or self.model_id
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
        weights_dir: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast mean and quantiles.

        Args:
            series: Historical values (rates in [0, 1]).
            horizon: Forecast steps.
            weights_dir: Local snapshot dir (from bootstrap_models.py).

        Returns:
            (mean, quantiles) where mean shape is (horizon,) and
            quantiles shape is (horizon, 10) for q10..q90.
        """
        h = horizon or self.prediction_length
        if self.use_mock:
            return self._mock_forecast(series, h)
        self._load(weights_dir)
        assert self._model is not None
        forecast_fn = getattr(self._model, "forecast")
        # timesfm forecast returns (mean, quantiles); adapt defensively
        result = forecast_fn(inputs=[series.tolist()], horizon=h)
        return self._parse_timesfm_result(result, h)

    @staticmethod
    def _parse_timesfm_result(
        result: object, h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse timesfm forecast output into (mean, quantiles)."""
        # timesfm>=2 returns (point_forecast, quantile_forecast) tuple
        if isinstance(result, tuple) and len(result) == 2:
            mean_arr = np.asarray(result[0]).reshape(-1)[:h]
            q_arr = np.asarray(result[1])
            # q_arr shape: (horizon, n_quantiles) or (1, horizon, n_quantiles)
            q_arr = q_arr.reshape(q_arr.shape[-2], q_arr.shape[-1])[:h]
            # Resample/interpolate to exactly 10 quantiles (q10..q90)
            target_q = np.linspace(0.1, 0.9, 10)
            if q_arr.shape[1] != 10:
                # Assume source quantiles span [0,1]; interpolate
                src_q = np.linspace(0.0, 1.0, q_arr.shape[1])
                q_out = np.stack(
                    [np.interp(target_q, src_q, q_arr[i]) for i in range(h)]
                )
            else:
                q_out = q_arr
            mean = np.clip(mean_arr, 0.0, 1.0)
            return mean, np.clip(q_out, 0.0, 1.0)
        raise RuntimeError(f"Unexpected timesfm output type: {type(result)}")

    def _mock_forecast(
        self, series: np.ndarray, h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mock forecast: mean = last value, fixed quantile spreads."""
        last = float(np.clip(series[-1], 0.01, 0.99)) if len(series) > 0 else 0.2
        mean = np.full(h, last)
        spreads = np.array(
            [0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
        )
        quantiles = np.tile(mean[:, None], (1, 10))
        for i in range(10):
            if i < 5:
                quantiles[:, i] = mean - spreads[i]
            else:
                quantiles[:, i] = mean + spreads[i]
        return mean, np.clip(quantiles, 0.0, 1.0)
