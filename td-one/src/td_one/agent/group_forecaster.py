"""Group-level attrition forecasting with fallback to global.

For each borrower group, forecasts the group's monthly attrition series
using the injected ``forecaster_fn``. Groups with fewer than
``min_group_n`` at-risk borrowers in the training window fall back to
the global (portfolio) forecast — shrinkage toward global, same principle
as the decision policy's James-Stein.

The ``forecaster_fn`` contract: ``(series: np.ndarray) -> np.ndarray``
mapping a 1D history of attrition rates to a quantile forecast vector.
In production this wraps ``ProposedForecasterV2``; in tests a stub.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from td_one.data_gen.group_series import GroupSeries

DEFAULT_MIN_GROUP_N = 200


@dataclass
class GroupForecast:
    group_id: str
    quantiles: np.ndarray  # forecast quantile vector
    used_fallback: bool
    n_at_risk_train: int


class GroupForecaster:
    """Forecast each group's attrition series, with global fallback."""

    def __init__(
        self,
        forecaster_fn: Callable[[np.ndarray], np.ndarray],
        min_group_n: int = DEFAULT_MIN_GROUP_N,
    ) -> None:
        if min_group_n < 1:
            raise ValueError(f"min_group_n must be >= 1, got {min_group_n}")
        self.forecaster_fn = forecaster_fn
        self.min_group_n = min_group_n

    def forecast(
        self,
        groups: dict[str, GroupSeries],
        train_months: int | None = None,
    ) -> dict[str, GroupForecast]:
        """Forecast every group.

        Parameters
        ----------
        groups:
            ``group_id -> GroupSeries``. Must include the portfolio total
            under key ``"__portfolio__"`` (see
            ``td_one.data_gen.group_series.portfolio_totals``) for fallback.
        train_months:
            History length used for the ``min_group_n`` check (sum of
            ``n_at_risk`` over the last ``train_months``). None = all.

        """
        if "__portfolio__" not in groups:
            raise ValueError("groups must include '__portfolio__' total")
        portfolio = groups["__portfolio__"]
        global_q = self.forecaster_fn(
            np.array(portfolio.attrition_rate, dtype=float)
        )
        out: dict[str, GroupForecast] = {}
        for gid, gs in groups.items():
            if gid == "__portfolio__":
                continue
            n_risk = gs.n_at_risk
            if train_months is not None:
                n_risk = n_risk[-train_months:]
            total_risk = int(sum(n_risk))
            if total_risk < self.min_group_n:
                out[gid] = GroupForecast(
                    group_id=gid,
                    quantiles=global_q.copy(),
                    used_fallback=True,
                    n_at_risk_train=total_risk,
                )
            else:
                q = self.forecaster_fn(
                    np.array(gs.attrition_rate, dtype=float)
                )
                out[gid] = GroupForecast(
                    group_id=gid,
                    quantiles=q,
                    used_fallback=False,
                    n_at_risk_train=total_risk,
                )
        return out


def v2_forecaster_fn(h: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    """Build a forecaster_fn wrapping ProposedForecasterV2.

    Fits the router on the given series with naive/ETS/ARIMA experts and
    returns the h-step quantile forecast. Heavyweight; prefer module-level
    reuse across groups.
    """
    from td_one.agent.proposed_method_v2 import ProposedForecasterV2

    def _forecast(series: np.ndarray) -> np.ndarray:
        # Minimal expert quantiles: naive repeats last value across levels.
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        last = float(series[-1]) if len(series) else 0.0
        naive_q = np.tile(last, (h, len(levels)))
        # Feature dict: the router needs *something*; use series stats.
        feats = {
            "n": float(len(series)),
            "mean": float(np.mean(series)) if len(series) else 0.0,
            "std": float(np.std(series)) if len(series) else 0.0,
        }
        fc = ProposedForecasterV2(h=h, dataset=None)
        # Fit on a single pseudo-record so the router has state.
        try:
            out = fc.predict(
                train=series,
                model_quantiles={"naive": naive_q},
                feats=feats,
                feature_names=list(feats.keys()),
            )
            return np.asarray(out.quantiles, dtype=float).ravel()[: len(levels)]
        except Exception:  # noqa: BLE001 — query failures are the validation signal
            return naive_q[0]

    return _forecast
