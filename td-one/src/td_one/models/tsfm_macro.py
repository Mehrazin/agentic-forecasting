"""Macro-conditioned TSFM via the residual approach.

Neither Chronos-2 nor TimesFM-2.5 accepts exogenous covariates natively:
Chronos-2's pipeline ``predict`` takes only ``inputs`` (univariate
context), and the installed timesfm 3.x ``forecast(horizon, inputs)``
exposes no covariate path. Both are pure zero-shot univariate
forecasters. So macro-conditioning uses the honest residual fallback:

1. Estimate the macro sensitivity ``beta`` by OLS of trailing
   ``logit(rates)`` on the FRED z-series (the exact estimator in
   :class:`td_one.models.macro_m0.MacroM0`, reused by composition).
2. Residualize the training rates in logit space::
       r_resid = sigmoid(logit(r) - beta_hat * z)
   (always in (0, 1) by construction).
3. Forecast one step ahead with the plain TSFM on the residualized series.
4. Re-add the macro effect per quantile level::
       q' = sigmoid(logit(q) + beta_hat * z_forecast)
   ``sigmoid`` is monotone, so this introduces no new quantile crossings
   (the existing sort fix for neural crossings still applies).

On DGPs without macro structure ``beta_hat`` is ~0 and this reduces
EXACTLY to the plain TSFM forecast (regression-tested). The
residualization is the structural inverse of the macro DGP overlay,
which injects macro as an additive logit shift
(``simulator.generate_outcomes(..., macro_shifts=...)``).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np

from td_one.models.macro_m0 import MacroM0
from td_one.models.macro_m0 import (
    _logit as _m_logit,
)
from td_one.models.macro_m0 import (
    _sigmoid as _m_sigmoid,
)
from td_one.models.tsfm import Chronos2Wrapper, TimesFMWrapper

#: Wrapper types this module conditions. Both expose
#: ``forecast_quantiles(series, horizon) -> (levels, values)``.
TSFMWrapperT = Chronos2Wrapper | TimesFMWrapper

#: Clip for rates before logit (matches macro_m0).
_RATE_CLIP = 1e-6


def _as_rates(n: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Monthly switching rates, clipped into (0, 1) for logit use."""
    r = np.asarray(a, dtype=float) / np.maximum(np.asarray(n, dtype=float), 1.0)
    return np.asarray(np.clip(r, _RATE_CLIP, 1.0 - _RATE_CLIP))


class MacroConditionedTSFM:
    """TSFM forecaster with a FRED macro overlay via residualization.

    The macro sensitivity is re-estimated at every forecast origin from
    trailing data only (no lookahead): ``fit`` is called per origin
    inside :meth:`forecast_quantiles`.
    """

    def __init__(
        self,
        wrapper: TSFMWrapperT,
        fred_z: dict[date, float] | None = None,
        trailing_window: int = 6,
        est_months: int = 24,
    ) -> None:
        """Initialize.

        Args:
            wrapper: A (real or mock) Chronos2Wrapper/TimesFMWrapper.
            fred_z: Month-end date -> z-score mapping (see
                :func:`td_one.models.macro_m0.load_fred_z`). Empty/None
                means no macro information (beta falls back to 0).
            trailing_window: Window for the underlying M0 beta estimator.
            est_months: Max trailing months for the OLS estimation.

        """
        self._wrapper = wrapper
        self._beta_est = MacroM0(
            trailing_window=trailing_window,
            fred_z=fred_z,
            est_months=est_months,
        )

    @property
    def beta_hat_(self) -> float:
        """Last estimated macro sensitivity (0.0 before first forecast)."""
        try:
            return self._beta_est.beta_hat_
        except RuntimeError:
            return 0.0

    def forecast_quantiles(
        self,
        n_renewals: np.ndarray,
        n_switches: np.ndarray,
        months: Sequence[date],
        forecast_month: date,
    ) -> dict[float, float]:
        """One-step-ahead rate quantiles on the wrapper's native grid.

        Args:
            n_renewals: Renewal counts per month (oldest first).
            n_switches: Switch counts per month (oldest first).
            months: Month-end dates aligned with n/a.
            forecast_month: Month-end date being forecast.

        Returns:
            Mapping quantile level -> forecast rate, monotone in level
            (neural crossings repaired by sort, as in the tools path).

        """
        n = np.asarray(n_renewals, dtype=float)
        a = np.asarray(n_switches, dtype=float)
        if n.shape != a.shape or n.size == 0:
            raise ValueError("n_renewals and n_switches must be non-empty and aligned")
        if len(months) != n.size:
            raise ValueError("months must align with n_renewals/n_switches")

        # Beta re-estimated per origin from trailing data only.
        self._beta_est.fit(n, a, list(months))
        beta = self._beta_est.beta_hat_

        rates = _as_rates(n, a)
        if beta != 0.0:
            # Lagged residualization: logit(r[i]) - beta*z[i-1], matching
            # the predictive setup (re-add uses z[t] at forecast time).
            # First month has no lagged z, uses 0.0.
            z_list: list[float] = []
            for j in range(len(months)):
                if j == 0:
                    z_list.append(0.0)
                else:
                    zv = self._beta_est.fred_z.get(months[j - 1], 0.0)
                    z_list.append(float(zv) if np.isfinite(zv) else 0.0)
            z = np.array(z_list, dtype=float)
            resid = np.array(
                [
                    _m_sigmoid(_m_logit(float(r)) - beta * float(zz))
                    for r, zz in zip(rates, z, strict=True)
                ]
            )
        else:
            resid = rates

        levels, values = self._wrapper.forecast_quantiles(resid, horizon=1)
        raw = np.asarray(values[:, 0], dtype=float)

        if beta != 0.0:
            # PIT-safe: re-add using latest z at origin (not forecast_month,
            # which is unknowable). Falls back to last training month.
            avail = [d for d in self._beta_est.fred_z if d < forecast_month]
            if avail:
                z_f = self._beta_est.fred_z[max(avail)]
            else:
                z_f = self._beta_est.fred_z.get(months[-1], 0.0)
            shift = beta * (z_f if np.isfinite(z_f) else 0.0)
            raw = np.array([_m_sigmoid(_m_logit(float(v)) + shift) for v in raw])

        # Repair neural quantile crossings by sort (same as tools path).
        vals = np.sort(np.clip(raw, 0.0, 1.0))
        lvls = np.asarray(levels, dtype=float)
        order = np.argsort(lvls, kind="stable")
        return {float(lvls[i]): float(vals[j]) for j, i in enumerate(order)}
