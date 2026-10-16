"""Conformalized M0: split-conformal intervals over the M0 forecast.

M0 undercovers at scale (0.671 vs 0.80 nominal). This wraps the
trailing-window Beta-Binomial with a split-conformal correction
(CQR-style): a calibration block at the end of the trailing history is
scored rolling-origin, and the final [q10, q90] rate interval is widened
by the (1-alpha) quantile of the nonconformity scores.

Scores are in *rate* space (each calibration month normalized by its own
exposure), so the correction transfers across varying N. With too little
history for a calibration block the model falls back to plain M0
(``qhat_ == 0.0``); the fallback is explicit, not silent.
"""

from __future__ import annotations

import numpy as np

from td_one.models.baseline import TRAILING_WINDOW, HistoricalBaseline
from td_one.models.tsfm import TIMESFM_LEVELS

#: Minimum calibration months for the conformal correction to apply.
_MIN_CAL_MONTHS = 2


class ConformalM0:
    """M0 with a split-conformal widening of the 80% interval."""

    def __init__(
        self,
        trailing_window: int = TRAILING_WINDOW,
        cal_months: int = 4,
        alpha: float = 0.2,
    ) -> None:
        """Initialize unfitted conformal baseline.

        Args:
            trailing_window: Months pooled by the underlying M0.
            cal_months: Max calibration months (capped by history length).
            alpha: Miscoverage target; the interval aims at 1 - alpha.

        """
        if trailing_window < 1:
            raise ValueError("trailing_window must be >= 1")
        if cal_months < 1:
            raise ValueError("cal_months must be >= 1")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.trailing_window = trailing_window
        self.cal_months = cal_months
        self.alpha = alpha
        self._n: np.ndarray | None = None
        self._a: np.ndarray | None = None
        self.qhat_: float = 0.0
        self.n_cal_: int = 0
        self._fitted = False

    def fit(self, n_renewals: np.ndarray, n_switches: np.ndarray) -> None:
        """Fit M0 history and calibrate the conformal widening.

        Args:
            n_renewals: Renewal counts per period (oldest first).
            n_switches: Switch counts per period (oldest first).

        """
        n = np.asarray(n_renewals, dtype=float)
        a = np.asarray(n_switches, dtype=float)
        if n.shape != a.shape or n.size == 0:
            raise ValueError("n_renewals and n_switches must be non-empty and aligned")
        if bool((a < 0).any() or (a > n).any()):
            raise ValueError("switch counts must satisfy 0 <= A <= N")
        self._n, self._a = n, a
        self.qhat_, self.n_cal_ = self._calibrate(n, a)
        self._fitted = True

    def _calibrate(self, n: np.ndarray, a: np.ndarray) -> tuple[float, int]:
        """Rolling-origin calibration scores; returns (qhat, n_cal)."""
        n_cal = min(self.cal_months, n.size - 2)
        if n_cal < _MIN_CAL_MONTHS:
            return 0.0, 0
        scores: list[float] = []
        first_cal = n.size - n_cal
        for t in range(first_cal, n.size):
            if n[t] <= 0:
                continue  # no exposure: nothing to score
            m0 = HistoricalBaseline(trailing_window=self.trailing_window)
            m0.fit(n[:t], a[:t])
            q10_c, _, q90_c = m0.predict_quantiles(int(n[t]))
            lo, hi = q10_c / n[t], q90_c / n[t]
            y = a[t] / n[t]
            scores.append(max(lo - y, y - hi, 0.0))
        if len(scores) < _MIN_CAL_MONTHS:
            return 0.0, 0
        # Finite-sample conformal quantile: ceil((m+1)(1-alpha))/m.
        m = len(scores)
        k = int(np.ceil((m + 1) * (1.0 - self.alpha))) - 1
        qhat = float(np.sort(np.asarray(scores))[min(max(k, 0), m - 1)])
        return qhat, m

    def predict_rate_quantiles(
        self,
        n_upcoming: int,
        levels: np.ndarray | None = None,
    ) -> dict[float, float]:
        """Conformalized rate quantiles on the 9-level TIMESFM grid.

        The underlying M0 lattice is computed, then the 0.1/0.9 levels
        are widened by ``qhat_`` (in rate units) and clipped to [0, 1].
        Interior levels are untouched; monotonicity is preserved because
        they already lie inside the raw [q10, q90].

        Args:
            n_upcoming: Upcoming renewals (exposure, >= 1).
            levels: Quantile levels; defaults to TIMESFM_LEVELS.

        Returns:
            Mapping level -> rate quantile.

        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        if n_upcoming < 1:
            raise ValueError("n_upcoming must be >= 1")
        assert self._n is not None and self._a is not None
        lv = (
            np.asarray(TIMESFM_LEVELS, dtype=float)
            if levels is None
            else np.asarray(levels, dtype=float)
        )
        if bool((lv <= 0.0).any() or (lv >= 1.0).any()):
            raise ValueError("levels must be in (0, 1)")
        m0 = HistoricalBaseline(trailing_window=self.trailing_window)
        m0.fit(self._n, self._a)
        lattice = m0.predict_rate_quantiles(n_upcoming, lv)
        if self.qhat_ > 0.0:
            widened = dict(lattice)
            for target, sign in ((0.1, -1.0), (0.9, 1.0)):
                # Find the grid level closest to the target (exact on the grid).
                key = min(widened, key=lambda k: abs(k - target))
                widened[key] = float(np.clip(widened[key] + sign * self.qhat_, 0.0, 1.0))
            # Re-enforce monotonicity (clip preserves order here, but be safe).
            vals = np.array([widened[float(k)] for k in lv])
            vals = np.maximum.accumulate(vals)
            widened = {float(k): float(v) for k, v in zip(lv, vals, strict=True)}
            return widened
        return lattice
