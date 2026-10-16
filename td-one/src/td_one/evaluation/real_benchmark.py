"""Real-data benchmark utilities for M4-style evaluation.

Pure functions for evaluating forecasters on arbitrary real-valued
monthly series (M4 Monthly). Supports two framings:

A. Continuous level forecasting: quantile-based CRPS for H-step-ahead
   predictive distributions.
B. Binary direction forecasting: is the series higher at t+H than at t?

All functions are deterministic given their inputs and have no model
dependencies (TSFM wrappers stay in the experiment script).
"""

from __future__ import annotations

import numpy as np

#: Common quantile grid for cross-model CRPS comparability (19 levels).
COMMON_GRID = np.array(
    [
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
    ]
)

#: Finite-sample conformal quantile helper (matches ConformalM0).
_ALPHA_DEFAULT = 0.2


def normalize_to_unit_interval(series: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max normalize a series to [0, 1] for TSFM wrappers.

    Args:
        series: 1-D array of real values (no NaN).

    Returns:
        (normalized, lo, hi) with normalized = (series - lo) / (hi - lo).
        Constant series map to 0.5 with a synthetic ±1.0 range.

    """
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("series must be a non-empty 1-D array")
    if bool(np.isnan(arr).any()):
        raise ValueError("series must not contain NaN")
    lo = float(arr.min())
    hi = float(arr.max())
    if hi == lo:
        return np.full_like(arr, 0.5), lo - 1.0, lo + 1.0
    return (arr - lo) / (hi - lo), lo, hi


def denormalize_quantiles(q_values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Invert :func:`normalize_to_unit_interval` on quantile forecasts."""
    return np.asarray(q_values, dtype=float) * (hi - lo) + lo


def interp_quantiles_to_grid(
    levels: np.ndarray, values: np.ndarray, grid: np.ndarray = COMMON_GRID
) -> np.ndarray:
    """Interpolate a quantile lattice onto a common grid.

    Args:
        levels: Ascending quantile levels in (0, 1), shape (K,).
        values: Quantile values, shape (K,) or (K, H).
        grid: Target levels.

    Returns:
        Interpolated values, shape (len(grid),) or (len(grid), H),
        monotone non-decreasing in the level axis.

    """
    lv = np.asarray(levels, dtype=float)
    vals = np.asarray(values, dtype=float)
    g = np.asarray(grid, dtype=float)
    if lv.ndim != 1 or bool((np.diff(lv) <= 0).any()):
        raise ValueError("levels must be strictly ascending 1-D")
    if vals.shape[0] != lv.shape[0]:
        raise ValueError("values first axis must match levels")
    # Enforce monotonicity (TSFM lattices can have tiny crossings).
    vals = np.maximum.accumulate(vals, axis=0)
    if vals.ndim == 1:
        return np.asarray(np.interp(g, lv, vals))
    out = np.stack([np.interp(g, lv, vals[:, h]) for h in range(vals.shape[1])], axis=1)
    return np.asarray(np.maximum.accumulate(out, axis=0))


def crps_from_quantiles(
    levels: np.ndarray,
    quantile_values: np.ndarray,
    actual: np.ndarray | float,
) -> float:
    """CRPS from a discrete quantile forecast (pinball-loss estimator).

    CRPS(F, y) = 2 * mean_k QS_{tau_k}(q_k, y), where
    QS_tau(q, y) = (1{y <= q} - tau) * (q - y) >= 0.

    Args:
        levels: Quantile levels, shape (K,).
        quantile_values: Forecast quantiles, shape (K,) or (K, H).
        actual: Realized value(s), scalar or shape (H,).

    Returns:
        Mean CRPS over the horizon (lower is better).

    """
    lv = np.asarray(levels, dtype=float)
    qv = np.asarray(quantile_values, dtype=float)
    y = np.asarray(actual, dtype=float)
    if qv.ndim == 1:
        qv = qv[:, None]
    if y.ndim == 0:
        y = y.reshape(1)
    if qv.shape[1] != y.shape[0]:
        raise ValueError(f"horizon mismatch: quantiles {qv.shape[1]} vs actual {y.shape[0]}")
    # QS_tau(q, y) = (1{y <= q} - tau) * (q - y); broadcast over (K, H).
    indicator = (y[None, :] <= qv).astype(float)
    qs = (indicator - lv[:, None]) * (qv - y[None, :])
    return float(2.0 * np.mean(qs))


def binary_crps(p: float, y: int) -> float:
    """CRPS for a Bernoulli(p) forecast against a binary outcome.

    Closed form: CRPS = (1 - y) * p^2 + y * (1 - p)^2.
    Strictly proper; lower is better.

    Args:
        p: Predicted P(Y = 1), in [0, 1].
        y: Realized outcome, 0 or 1.

    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1]; got {p}")
    if y not in (0, 1):
        raise ValueError(f"y must be 0 or 1; got {y}")
    return float((1 - y) * p * p + y * (1 - p) * (1 - p))


def binary_target_up(values: np.ndarray, origin: int, horizon: int = 6) -> int:
    """Binary direction target: 1 if values[origin + horizon] > values[origin].

    Args:
        values: Full series (train + test concatenated).
        origin: Index of the forecast origin.
        horizon: Steps ahead.

    """
    v = np.asarray(values, dtype=float)
    if not 0 <= origin < v.shape[0] - horizon:
        raise ValueError(
            f"origin {origin} needs values[{origin + horizon}] but series has length {v.shape[0]}"
        )
    return int(v[origin + horizon] > v[origin])


def framing_b_origins(
    train_len: int,
    test_len: int,
    n_origins: int = 24,
    horizon: int = 6,
    m0_trailing: int = 12,
) -> list[int]:
    """List valid forecast origins for framing B (no leakage).

    Origins are the last ``n_origins`` train points. At origin t the M0
    fits on ``m0_trailing`` binary outcomes from origins [t-h-11, t-h]
    (the latest fully-observed outcome uses y[t] vs y[t-h]), so the
    earliest origin needs t - horizon - m0_trailing + 1 >= 0.

    Args:
        train_len: Length of the training history.
        test_len: Length of the held-out test block (targets may extend
            into it for the last ``horizon`` origins — that is the OOT
            evaluation, not leakage, since models only fit on <= t).
        n_origins: Number of origins (fewer if the series is short).
        horizon: Steps ahead for the direction target.
        m0_trailing: Trailing binary outcomes pooled by M0.

    Returns:
        Ascending list of origin indices (into train+test concatenation).

    """
    if train_len < 1 or test_len < 0:
        raise ValueError("train_len >= 1 and test_len >= 0 required")
    # Latest origin: last train point (never fit on test data).
    # Earliest: need m0_trailing outcomes, each needing horizon lookahead.
    earliest = horizon + m0_trailing - 1
    latest = train_len - 1
    if latest < earliest:
        return []
    start = max(earliest, latest - n_origins + 1)
    return list(range(start, latest + 1))


def m0_origin_window(origin: int, horizon: int = 6, m0_trailing: int = 12) -> list[int]:
    """Origin indices whose binary outcomes are fully observed at ``origin``.

    Outcome at origin s needs y[s + horizon] <= y[origin], i.e. s <= origin
    - horizon. Returns the ``m0_trailing`` most recent such origins.

    """
    if origin < horizon:
        raise ValueError("origin must be >= horizon")
    return list(range(origin - horizon - m0_trailing + 1, origin - horizon + 1))


def trailing_residual_quantiles(
    history: np.ndarray,
    levels: np.ndarray,
    horizon: int,
    kind: str = "naive",
    max_origins: int = 36,
) -> dict[int, np.ndarray]:
    """Empirical h-step residual quantiles for naive-style baselines.

    For step h, residuals are y[t] - y[t-h] (naive) or y[t] - y[t-12]
    (seasonal naive, m=12) over trailing origins. The predictive
    distribution at step h is point_h + residual_quantiles[h].

    Args:
        history: 1-D training history (no NaN).
        levels: Quantile levels for the residual distribution.
        horizon: Forecast steps; returns entries for h = 1..horizon.
        kind: "naive" or "snaive".
        max_origins: Max trailing origins for the residual pool.

    Returns:
        Mapping h -> residual quantiles on ``levels``, shape (len(levels),).

    """
    y = np.asarray(history, dtype=float)
    if y.ndim != 1 or y.size < 14:
        raise ValueError("history must be 1-D with >= 14 values")
    if kind not in ("naive", "snaive"):
        raise ValueError("kind must be 'naive' or 'snaive'")
    lv = np.asarray(levels, dtype=float)
    out: dict[int, np.ndarray] = {}
    n = y.size
    for h in range(1, horizon + 1):
        lag = h if kind == "naive" else 12
        # Origins t with t - lag >= 0; take the most recent max_origins.
        t_end = n - 1
        t_start = max(lag, t_end - max_origins + 1)
        resid = np.array([y[t] - y[t - lag] for t in range(t_start, t_end + 1)])
        out[h] = np.quantile(resid, lv)
    return out


def finite_sample_quantile(sorted_scores: np.ndarray, level: float) -> float:
    """Conformal finite-sample quantile: ceil((m+1)*level)/m."""
    s = np.asarray(sorted_scores, dtype=float)
    m = s.size
    if m == 0:
        raise ValueError("no scores")
    k = int(np.ceil((m + 1) * level)) - 1
    return float(s[min(max(k, 0), m - 1)])


def conformal_naive_quantiles(
    history: np.ndarray,
    levels: np.ndarray,
    horizon: int,
    cal_months: int = 12,
    alpha: float = _ALPHA_DEFAULT,
) -> np.ndarray:
    """Split-conformal naive quantile forecasts.

    Fit block: history[:-cal_months]; calibration block: last cal_months.
    The residual distribution is estimated on the fit block; the final
    forecast is anchored at history[-1] (the true origin). Base quantiles
    come from trailing residuals on the fit block; the 0.1/0.9 levels are
    widened by the finite-sample (1-alpha) quantile of the calibration
    nonconformity scores (CQR-style, mirroring ConformalM0).
    Returns shape (len(levels), horizon).

    Methodological note: nonconformity scores pool 1- to H-step errors
    into a single qhat applied uniformly across horizons. Step-1 intervals
    are therefore likely over-widened relative to a per-horizon calibration.
    Empirically the pooled coverage hits nominal (0.800 on M4), but
    per-horizon calibration would be sharper.

    Args:
        history: 1-D training history (no NaN), length > cal_months + 14.
        levels: Quantile levels.
        horizon: Forecast steps.
        cal_months: Calibration block size.
        alpha: Miscoverage target.

    """
    y = np.asarray(history, dtype=float)
    lv = np.asarray(levels, dtype=float)
    if y.size <= cal_months + 14:
        raise ValueError("history too short for fit + calibration blocks")
    fit, cal = y[:-cal_months], y[-cal_months:]
    h = horizon
    resid_q = trailing_residual_quantiles(fit, lv, h, kind="naive")
    # Anchor the forecast at the true last value (not the fit block end).
    point = float(y[-1])
    k = len(lv)
    base = np.empty((k, h))
    for step in range(1, h + 1):
        base[:, step - 1] = point + resid_q[step]
    # Calibration nonconformity on the cal block using base intervals.
    i10 = int(np.argmin(np.abs(lv - 0.1)))
    i90 = int(np.argmin(np.abs(lv - 0.9)))
    scores: list[float] = []
    for t in range(cal.size):
        # Naive h-step forecasts from the fit block's endpoint evolving
        # through the calibration block (rolling origin within cal).
        for step in range(1, min(h, cal.size - t) + 1):
            origin_val = cal[t - 1] if t > 0 else fit[-1]
            lo_c = origin_val + resid_q[step][i10]
            hi_c = origin_val + resid_q[step][i90]
            actual_c = cal[t + step - 1]
            scores.append(max(lo_c - actual_c, actual_c - hi_c, 0.0))
    if not scores:
        return base
    qhat = finite_sample_quantile(np.sort(np.asarray(scores)), 1.0 - alpha)
    widened = base.copy()
    widened[i10, :] -= qhat
    widened[i90, :] += qhat
    return np.maximum.accumulate(widened, axis=0)


def prob_up_from_quantiles(levels: np.ndarray, q_values: np.ndarray, current: float) -> float:
    """P(forecast > current) from a quantile lattice via CDF interpolation.

    Linearly interpolates the implied CDF at ``current`` (flat
    extrapolation outside the lattice), returning 1 - F(current),
    clipped to [0.02, 0.98] for Brier stability.

    Args:
        levels: Ascending quantile levels in (0, 1).
        q_values: Quantile values at one horizon step, ascending.
        current: Reference value (series value at the origin).

    """
    lv = np.asarray(levels, dtype=float)
    qv = np.asarray(q_values, dtype=float)
    if lv.shape != qv.shape or lv.ndim != 1:
        raise ValueError("levels and q_values must be matching 1-D arrays")
    if bool((np.diff(lv) <= 0).any()):
        raise ValueError("levels must be strictly ascending")
    qv = np.maximum.accumulate(qv)  # monotone safety
    if current <= qv[0]:
        cdf = float(lv[0])
    elif current >= qv[-1]:
        cdf = float(lv[-1])
    else:
        cdf = float(np.interp(current, qv, lv))
    return float(np.clip(1.0 - cdf, 0.02, 0.98))


def conformal_shrink_probability(p_m0: float, qhat: float, strength: float = 20.0) -> float:
    """Shrink an M0 probability toward 0.5 by conformal miscalibration.

    p_conf = w * p_m0 + (1 - w) * 0.5 with w = 1 / (1 + strength * qhat).
    When the conformal correction is zero (well-calibrated), this reduces
    to plain M0; large qhat (miscalibrated) pulls toward maximum entropy.

    Args:
        p_m0: M0 posterior mean probability in [0, 1].
        qhat: Conformal widening in rate units (>= 0).
        strength: Shrinkage strength.

    """
    if not 0.0 <= p_m0 <= 1.0:
        raise ValueError("p_m0 must be in [0, 1]")
    if qhat < 0.0:
        raise ValueError("qhat must be >= 0")
    w = 1.0 / (1.0 + strength * qhat)
    return float(np.clip(w * p_m0 + (1.0 - w) * 0.5, 0.0, 1.0))
