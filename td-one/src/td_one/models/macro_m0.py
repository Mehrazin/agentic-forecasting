"""MacroM0: M0 baseline with a FRED macro overlay.

Extends the trailing-window Beta-Binomial with a macro adjustment: OLS
of trailing logit(rates) on a z-scored FRED series, then shifts the
forecast in logit space by ``beta_hat * z[forecast_month]``.

On DGPs without macro structure the estimated beta is ~0 and the model
reduces to plain M0 (honest null -- see ``experiments/exp10_classical.py``).
On the v1 macro DGP it recovered most of the oracle gain (see
``experiments/exp8_macro.py``: oracle -32% CRPS, estimated captures 93%)
-- BUT that DGP used the wrong sign (beta=+0.6). With the corrected
negative lock-in on simulator v2 (``experiments/exp15_macro_corrected.py``),
the oracle gain is ~0 and the OLS overlay hurts (unstable beta_hat, often
the wrong sign). The 93% was a DGP artifact; do not cite it as a general
result.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from td_one.models.baseline import TRAILING_WINDOW, HistoricalBaseline
from td_one.models.tsfm import TIMESFM_LEVELS

#: Default FRED series for the macro overlay (30y mortgage rate).
DEFAULT_FRED_ID = "MORTGAGE30US"

#: Minimum months of (rate, z) pairs for the OLS beta estimation.
_MIN_EST_MONTHS = 12

#: Clip for logit to avoid infinities on 0/1 rates.
_LOGIT_CLIP = 1e-6


def _logit(p: float) -> float:
    p = float(np.clip(p, _LOGIT_CLIP, 1.0 - _LOGIT_CLIP))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _repo_fred_dir() -> Path:
    # td_one/models/macro_m0.py -> td-one/data/fred/
    return Path(__file__).resolve().parents[3] / "data" / "fred"


def load_fred_z(
    fred_id: str = DEFAULT_FRED_ID,
    data_dir: Path | str | None = None,
    asof: date | None = None,
) -> dict[date, float]:
    """Load z-scored 3-month percent change of a FRED series.

    Weekly/daily series are resampled to month-end. The 3-month percent
    change is z-scored.

    PIT SAFETY: if ``asof`` is given, the z-score mean/std are computed
    using only observations with date <= asof. Without ``asof``, stats use
    the full history — which leaks future information into past forecasts.
    Always pass the forecast origin as ``asof`` in evaluations.

    Args:
        fred_id: FRED series ID (parquet file stem).
        data_dir: Directory with ``<fred_id>.parquet``; defaults to the
            repo's ``data/fred/``.
        asof: PIT cutoff date. Stats computed on date <= asof only.
            If None, uses full history (LEAKY — for exploration only).

    Returns:
        Mapping month-end :class:`datetime.date` -> z-score. Months with
        undefined change (leading NaNs) are omitted. When ``asof`` is
        given, only months <= asof are returned.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
        ValueError: If the series has no usable observations.

    """
    d = Path(data_dir) if data_dir is not None else _repo_fred_dir()
    path = d / f"{fred_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"FRED parquet not found: {path}")
    df = pd.read_parquet(path)
    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{path} must have 'date' and 'value' columns")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date")["value"].resample("ME").last().dropna()
    if len(monthly) < 4:
        raise ValueError(f"{fred_id}: fewer than 4 monthly observations")
    # PIT: restrict to asof BEFORE computing stats.
    if asof is not None:
        cutoff = pd.Timestamp(asof)
        monthly = monthly[monthly.index <= cutoff]
        if len(monthly) < 4:
            raise ValueError(f"{fred_id}: fewer than 4 monthly observations at asof={asof}")
    mom3 = monthly.pct_change(3)
    std = float(mom3.std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"{fred_id}: degenerate 3m change series")
    z = (mom3 - float(mom3.mean())) / std
    out: dict[date, float] = {}
    # z has a DatetimeIndex (month-ends from the ME resample).
    for i, ts in enumerate(z.index):
        val = z.iloc[i]
        if np.isfinite(val):
            out[pd.Timestamp(ts).date()] = float(val)
    if not out:
        raise ValueError(f"{fred_id}: no finite z-scores")
    return out


class MacroM0:
    """M0 with a FRED macro overlay estimated by OLS.

    The underlying M0 lattice is computed as usual; the forecast is then
    shifted in logit space by ``beta_hat * z[forecast_month]``, where
    ``beta_hat`` comes from OLS of trailing logit(monthly rates) on the
    FRED z-series. With too little overlapping history the beta falls
    back to 0.0 (plain M0) -- the fallback is explicit via
    :attr:`beta_hat_`.
    """

    def __init__(
        self,
        trailing_window: int = TRAILING_WINDOW,
        fred_z: dict[date, float] | None = None,
        est_months: int = 24,
    ) -> None:
        """Initialize unfitted macro-adjusted baseline.

        Args:
            trailing_window: Months pooled by the underlying M0.
            fred_z: Month-end date -> z-score mapping. If None, an empty
                mapping is used (beta falls back to 0.0); pass
                :func:`load_fred_z` output for real macro data.
            est_months: Max trailing months used for the OLS estimation.

        """
        if trailing_window < 1:
            raise ValueError("trailing_window must be >= 1")
        if est_months < 1:
            raise ValueError("est_months must be >= 1")
        self.trailing_window = trailing_window
        self.fred_z: dict[date, float] = dict(fred_z) if fred_z else {}
        self.est_months = est_months
        self._m0 = HistoricalBaseline(trailing_window=trailing_window)
        self._beta_hat = 0.0
        self._n_est = 0
        self._fitted = False

    @property
    def beta_hat_(self) -> float:
        """Estimated macro sensitivity (0.0 when not estimable)."""
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        return self._beta_hat

    @property
    def n_est_(self) -> int:
        """Number of (rate, z) pairs used in the OLS estimation."""
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        return self._n_est

    def fit(
        self,
        n_renewals: np.ndarray,
        n_switches: np.ndarray,
        months: np.ndarray | list[date] | pd.DatetimeIndex,
    ) -> None:
        """Fit M0 and estimate the macro beta by OLS.

        Args:
            n_renewals: Renewal counts per period (oldest first).
            n_switches: Switch counts per period (oldest first).
            months: Month-end dates aligned with n/a (any datetime-like).

        Raises:
            ValueError: On misaligned/empty inputs or 0 <= A <= N
                violations.

        """
        n = np.asarray(n_renewals, dtype=float)
        a = np.asarray(n_switches, dtype=float)
        m = pd.to_datetime(months)
        if n.shape != a.shape or n.size == 0:
            raise ValueError("n_renewals and n_switches must be non-empty and aligned")
        if len(m) != n.size:
            raise ValueError("months must align with n_renewals/n_switches")
        if bool((a < 0).any() or (a > n).any()):
            raise ValueError("switch counts must satisfy 0 <= A <= N")
        self._m0.fit(n, a)

        # OLS of trailing logit(rates) on LAGGED z (predictive regression).
        # For a 1-step-ahead forecast from origin t we need z[t]; training
        # regresses logit(rate[i]) on z[i-1]. Using contemporaneous z[i]
        # would require knowing z[t+1] at forecast time (PIT violation).
        z_vals: list[float] = []
        y_vals: list[float] = []
        start = max(1, n.size - self.est_months)
        for i in range(start, n.size):
            d_prev = m[i - 1].date()
            z = self.fred_z.get(d_prev)
            if z is None or not np.isfinite(z):
                continue
            denom = max(n[i], 1.0)
            z_vals.append(float(z))
            y_vals.append(_logit(float(a[i] / denom)))
        self._n_est = len(z_vals)
        self._beta_hat = 0.0
        if self._n_est >= _MIN_EST_MONTHS:
            design = np.column_stack([np.ones(self._n_est), np.asarray(z_vals)])
            y = np.asarray(y_vals)
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            b = float(coef[1])
            self._beta_hat = b if np.isfinite(b) else 0.0
        self._fitted = True

    def predict_rate_quantiles(
        self,
        n_upcoming: int,
        forecast_month: date,
        levels: np.ndarray | None = None,
    ) -> dict[float, float]:
        """Rate quantiles shifted by ``beta_hat * z[origin]`` (lagged).

        The M0 lattice is computed, each level mapped through
        ``sigmoid(logit(rate) + beta_hat * z)``. Monotonicity is
        preserved (sigmoid is monotone). With ``beta_hat == 0`` or a
        missing z this is exactly the M0 lattice.

        PIT note: ``forecast_month`` is t+1 but z[t+1] is unknowable at
        origin t. We use the latest z at or before the origin
        (z[t]), matching the lagged training regression.

        Args:
            n_upcoming: Upcoming renewals (exposure, >= 1).
            forecast_month: Month-end date being forecast; the z used is
                the latest available strictly before it.
            levels: Quantile levels in (0, 1); defaults to TIMESFM_LEVELS.

        Returns:
            Mapping level -> rate quantile.

        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        lv = (
            np.asarray(TIMESFM_LEVELS, dtype=float)
            if levels is None
            else np.asarray(levels, dtype=float)
        )
        if bool((lv <= 0.0).any() or (lv >= 1.0).any()):
            raise ValueError("levels must be in (0, 1)")
        lattice = self._m0.predict_rate_quantiles(n_upcoming, lv)
        # Latest z strictly before forecast_month (= origin when asof is set).
        avail = [d for d in self.fred_z if d < forecast_month]
        z = self.fred_z[max(avail)] if avail else 0.0
        shift = self._beta_hat * (z if np.isfinite(z) else 0.0)
        if shift == 0.0:
            return lattice
        return {float(k): _sigmoid(_logit(v) + shift) for k, v in lattice.items()}
