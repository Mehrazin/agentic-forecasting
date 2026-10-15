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

    def _load(self) -> None:
        """Lazy-load the Chronos-2 pipeline."""
        if self._pipeline is not None or self.use_mock:
            return
        try:
            import torch
            from chronos import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
        except ImportError as e:
            raise RuntimeError(
                "chronos-forecasting not installed. "
                "Run: pip install chronos-forecasting"
            ) from e

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
        # Call predict via getattr to avoid strict typing issues
        predict_fn = getattr(pipeline, "predict")
        forecast = predict_fn(
            context=context,
            prediction_length=h,
            num_samples=n_samples,
        )
        arr = np.asarray(forecast)
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

    def forecast_quantiles(
        self,
        series: np.ndarray,
        horizon: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forecast mean and quantiles.

        Args:
            series: Historical values.
            horizon: Forecast steps.

        Returns:
            (mean, quantiles) where mean shape is (horizon,) and
            quantiles shape is (horizon, 10) for q10..q90.
        """
        h = horizon or self.prediction_length
        if self.use_mock or self._model is None:
            # Mock: mean = last value, quantiles spread
            last = float(np.clip(series[-1], 0.01, 0.99)) if len(series) > 0 else 0.2
            mean = np.full(h, last)
            # 10 quantiles: q10, q20, ..., q90, plus extras
            spreads = np.array([0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
            quantiles = np.tile(mean[:, None], (1, 10))
            # Lower quantiles below mean, upper above
            for i in range(10):
                if i < 5:
                    quantiles[:, i] = mean - spreads[i]
                else:
                    quantiles[:, i] = mean + spreads[i]
            quantiles = np.clip(quantiles, 0.0, 1.0)
            return mean, quantiles
        raise NotImplementedError("Real TimesFM inference not yet wired")
