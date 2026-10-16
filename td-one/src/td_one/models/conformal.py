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

    def _calibration_scores(self, n: np.ndarray, a: np.ndarray) -> list[float]:
        """Rolling-origin nonconformity scores in chronological order."""
        n_cal = min(self.cal_months, n.size - 2)
        scores: list[float] = []
        if n_cal < _MIN_CAL_MONTHS:
            return scores
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
        return scores

    @staticmethod
    def _finite_quantile(sorted_scores: np.ndarray, level: float) -> float:
        """Finite-sample conformal quantile: ceil((m+1)*level)/m."""
        m = len(sorted_scores)
        k = int(np.ceil((m + 1) * level)) - 1
        return float(sorted_scores[min(max(k, 0), m - 1)])

    def _calibrate(self, n: np.ndarray, a: np.ndarray) -> tuple[float, int]:
        """Rolling-origin calibration scores; returns (qhat, n_cal)."""
        scores = self._calibration_scores(n, a)
        if len(scores) < _MIN_CAL_MONTHS:
            return 0.0, 0
        qhat = self._finite_quantile(np.sort(np.asarray(scores)), 1.0 - self.alpha)
        return qhat, len(scores)

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


#: Bounds for the adaptive miscoverage level (keeps quantiles well-defined).
_ACI_ALPHA_MIN = 0.01
_ACI_ALPHA_MAX = 0.99


class AdaptiveConformalM0(ConformalM0):
    """Conformal M0 with Adaptive Conformal Inference (ACI).

    Gibbs & Candès (NeurIPS 2021): the miscoverage level adapts online
    through the calibration block via::

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t)

    where ``err_t = 1`` if calibration month ``t`` miscovered. After
    misses (regime drift) alpha drops and intervals widen; after hits
    alpha rises and intervals narrow. One tuning parameter (gamma);
    literature suggests small values (0.01-0.05).

    The adaptive alpha is clipped to [0.01, 0.99] so the quantile level
    stays well-defined under persistent drift.
    """

    def __init__(
        self,
        trailing_window: int = TRAILING_WINDOW,
        cal_months: int = 4,
        alpha: float = 0.2,
        gamma: float = 0.05,
    ) -> None:
        """Initialize adaptive conformal baseline.

        Args:
            trailing_window: Months pooled by the underlying M0.
            cal_months: Max calibration months (capped by history length).
            alpha: Target miscoverage rate; the long-run aim.
            gamma: ACI learning rate (> 0). Larger adapts faster but
                noisier; 0.05 is a robust default.

        """
        super().__init__(trailing_window=trailing_window, cal_months=cal_months, alpha=alpha)
        if not gamma > 0.0:
            raise ValueError("gamma must be positive")
        self.gamma = gamma
        self.alpha_path_: list[float] = []

    def _calibrate(self, n: np.ndarray, a: np.ndarray) -> tuple[float, int]:
        """ACI calibration: adapt alpha sequentially, then quantize.

        At step t the quantile uses only scores before t (strictly past),
        matching the online ACI protocol. The first step warm-starts from
        all scores.
        """
        scores = self._calibration_scores(n, a)
        m = len(scores)
        if m < _MIN_CAL_MONTHS:
            self.alpha_path_ = []
            return 0.0, 0
        alpha_t = self.alpha
        self.alpha_path_ = [alpha_t]
        for t, s in enumerate(scores):  # chronological order
            prev = scores if t == 0 else scores[:t]
            ordered = np.sort(np.asarray(prev))
            level = min(max(1.0 - alpha_t, _ACI_ALPHA_MIN), _ACI_ALPHA_MAX)
            q_t = self._finite_quantile(ordered, level)
            err = 1.0 if s > q_t else 0.0
            alpha_t = min(
                max(alpha_t + self.gamma * (self.alpha - err), _ACI_ALPHA_MIN),
                _ACI_ALPHA_MAX,
            )
            self.alpha_path_.append(alpha_t)
        final_level = min(max(1.0 - alpha_t, _ACI_ALPHA_MIN), _ACI_ALPHA_MAX)
        qhat = self._finite_quantile(np.sort(np.asarray(scores)), final_level)
        return qhat, m
