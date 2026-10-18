"""Regime tools: changepoint detection and event-impact backtesting.

``detect_changepoints`` finds structural breaks where models typically fail.
``backtest_event_impact`` quantifies whether Phase-1 events actually move a
series, using a pre-event baseline as the synthetic control.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from td_one.agent.tool_tracker import tracked


class ChangepointResult(BaseModel):
    """Structured output of :func:`detect_changepoints`."""

    changepoints: list[int] = Field(description="0-based indices of segment ends")
    segment_means: list[float]
    n_segments: int
    bic: float = Field(description="BIC of the segmented model (lower is better)")


def _segment_cost(x: np.ndarray, start: int, end: int) -> float:
    """Gaussian negative log-likelihood cost of segment [start, end)."""
    seg = x[start:end]
    n = seg.size
    if n < 2:
        return 0.0
    var = float(np.var(seg))
    if var <= 0:
        return 0.0
    return float(n * (np.log(var) + 1.0))


def _binary_segmentation(
    x: np.ndarray, start: int, end: int, penalty: float, out: list[int]
) -> None:
    """Recursive binary segmentation for changepoint detection."""
    n = end - start
    if n < 12:  # minimum segment length guard (2 * min_distance)
        return
    base_cost = _segment_cost(x, start, end)
    best_gain = 0.0
    best_cp = -1
    # Candidate changepoints respect the minimum distance on both sides.
    for cp in range(start + 6, end - 6):
        gain = base_cost - (_segment_cost(x, start, cp) + _segment_cost(x, cp, end))
        if gain > best_gain:
            best_gain = gain
            best_cp = cp
    if best_cp > 0 and best_gain > penalty:
        out.append(best_cp)
        _binary_segmentation(x, start, best_cp, penalty, out)
        _binary_segmentation(x, best_cp, end, penalty, out)


@tracked()
def detect_changepoints(series: Any, min_distance: int = 6) -> ChangepointResult:
    """Detect mean-shifts via binary segmentation with a BIC penalty.

    A lightweight PELT-style detector (no ruptures dependency): recursively
    splits where the Gaussian log-likelihood gain exceeds ``penalty =
    log(n)`` (BIC), respecting ``min_distance`` between changepoints.

    Args:
        series: 1-D numeric series (min 24 observations).
        min_distance: Minimum observations between changepoints.

    Returns:
        ChangepointResult with changepoint indices and segment means.

    """
    x = np.asarray(series, dtype=float).ravel()
    if x.size < 24:
        raise ValueError(f"series needs at least 24 observations, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or inf")
    if min_distance < 2:
        raise ValueError("min_distance must be >= 2")

    penalty = float(np.log(x.size))  # BIC penalty per added changepoint
    cps: list[int] = []
    _binary_segmentation(x, 0, x.size, penalty, cps)
    cps.sort()
    # Enforce min_distance post-hoc (binary segmentation uses 6 internally).
    filtered: list[int] = []
    for cp in cps:
        if not filtered or cp - filtered[-1] >= min_distance:
            filtered.append(cp)

    bounds = [0, *filtered, x.size]
    means = [float(np.mean(x[bounds[i] : bounds[i + 1]])) for i in range(len(bounds) - 1)]
    # BIC of the fitted segmented model.
    total_cost = sum(_segment_cost(x, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1))
    n_params = len(means) + len(filtered)  # segment means + changepoint locations
    bic = float(total_cost + n_params * np.log(x.size))
    return ChangepointResult(
        changepoints=filtered,
        segment_means=means,
        n_segments=len(means),
        bic=bic,
    )


class EventImpact(BaseModel):
    """Per-event impact estimate from :func:`backtest_event_impact`."""

    event_index: int
    event_date: str | None = None
    pre_mean: float
    post_mean: float
    ate: float = Field(description="post_mean - pre_mean (average treatment effect)")
    n_pre: int
    n_post: int


class EventImpactResult(BaseModel):
    """Structured output of :func:`backtest_event_impact`."""

    impacts: list[EventImpact]
    mean_ate: float
    share_positive: float = Field(description="Fraction of events with ate > 0")


@tracked()
def backtest_event_impact(
    event_dates: list[Any],
    series: Any,
    dates: list[Any] | None = None,
    window: int = 6,
) -> EventImpactResult:
    """Average treatment effect of events on a series.

    For each event, compares the mean of the ``window`` observations after
    the event to the mean of the ``window`` observations before it (the
    pre-event baseline acts as the synthetic control). Events too close to
    the series edges are skipped.

    Args:
        event_dates: Dates of events (date/datetime/str, or integer indices
            into the series when ``dates`` is None).
        series: 1-D numeric series.
        dates: Optional date index aligned with the series. When given,
            event_dates are matched to the nearest date <= event.
        window: Observations before/after each event to compare.

    Returns:
        EventImpactResult with per-event ATEs and the mean ATE.

    """
    x = np.asarray(series, dtype=float).ravel()
    if x.size < 2 * window + 1:
        raise ValueError(f"series too short for window={window}")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or inf")
    if window < 1:
        raise ValueError("window must be >= 1")

    idx: list[int] = []
    if dates is None:
        for e in event_dates:
            i = int(e)
            if window <= i < x.size - window:
                idx.append(i)
    else:
        d = pd.to_datetime(pd.Series(list(dates)))
        for e in event_dates:
            ev = pd.Timestamp(e)
            # Nearest date <= event (PIT-safe: never looks ahead).
            mask = d <= ev
            if not bool(mask.any()):
                continue
            i = int(np.flatnonzero(mask.to_numpy())[-1])
            if window <= i < x.size - window:
                idx.append(i)

    impacts: list[EventImpact] = []
    for k, i in enumerate(idx):
        pre = x[i - window : i]
        post = x[i : i + window]
        ev_date = None
        if dates is not None:
            ev_date = str(pd.Timestamp(event_dates[k]).date())
        impacts.append(
            EventImpact(
                event_index=i,
                event_date=ev_date,
                pre_mean=float(np.mean(pre)),
                post_mean=float(np.mean(post)),
                ate=float(np.mean(post) - np.mean(pre)),
                n_pre=window,
                n_post=window,
            )
        )
    ates = [im.ate for im in impacts]
    return EventImpactResult(
        impacts=impacts,
        mean_ate=float(np.mean(ates)) if ates else 0.0,
        share_positive=float(np.mean([a > 0 for a in ates])) if ates else 0.0,
    )


# ---------------------------------------------------------------------------
# regime_probabilities
# ---------------------------------------------------------------------------


class RegimeProbs(BaseModel):
    """Structured output of :func:`regime_probabilities`."""

    regimes: list[str] = Field(description="Regime names for this spec")
    probabilities: dict[str, float] = Field(description="P(regime), sums to 1")
    label: str = Field(description="Argmax regime (hard label)")
    spec: str = Field(description="Which hand-rule spec was applied")


def _label_m4(feats: dict[str, float]) -> str:
    """exp43 M4 regimes: seasonal / trending / noisy."""
    if feats.get("seasonal_strength", 0.0) > 0.5:
        return "seasonal"
    if feats.get("trend_strength", 0.0) > 0.7:
        return "trending"
    return "noisy"


def _label_generic(feats: dict[str, float]) -> str:
    """exp43 SMM-style regimes: calm / stressed."""
    if feats.get("n_recent_cps", 0.0) >= 1 or feats.get("vol_ratio", 0.0) > 1.0:
        return "stressed"
    return "calm"


def _implied(value: float, lo: float, hi: float) -> int:
    return -1 if value < lo else (1 if value > hi else 0)


def _label_boc(feats: dict[str, float]) -> str:
    """exp32 BoC regimes: EASY (all macro signals agree) / HARD.

    Signals: rate_momentum (cut/hold/hike), inflation_gap, yield_spread.
    """
    implied = {
        _implied(feats.get("rate_momentum", 0.0), -0.25, 0.25),
        _implied(feats.get("inflation_gap", 0.0), -1.0, 1.0),
        _implied(feats.get("yield_spread", 0.0), -0.5, 0.5),
    }
    return "EASY" if len(implied) == 1 else "HARD"


_REGIME_SPECS: dict[str, tuple[list[str], Any]] = {
    "m4": (["seasonal", "trending", "noisy"], _label_m4),
    "generic": (["calm", "stressed"], _label_generic),
    "boc": (["EASY", "HARD"], _label_boc),
}


@tracked()
def regime_probabilities(
    features: dict[str, float],
    spec: str = "m4",
    smooth: float = 0.1,
) -> RegimeProbs:
    """Map diagnostic features to soft regime probabilities.

    This is exp43's winning hand-rule router as a reusable tool
    (kept per exp45: hand rules beat unsupervised discovery).
    The hard label comes from the rule; probabilities are eps-smoothed
    so downstream combination never puts zero mass on a regime.

    Args:
        features: Feature dict, e.g. from :func:`series_features`.
            m4 spec needs ``seasonal_strength`` / ``trend_strength``;
            generic needs ``n_recent_cps`` / ``vol_ratio``;
            boc needs ``rate_momentum`` / ``inflation_gap`` / ``yield_spread``.
        spec: One of ``"m4"``, ``"generic"``, ``"boc"``.
        smooth: Mass reserved for non-winning regimes (default 0.1).

    Returns:
        RegimeProbs with regimes, smoothed probabilities, and hard label.

    """
    if spec not in _REGIME_SPECS:
        raise ValueError(f"spec must be one of {sorted(_REGIME_SPECS)}, got {spec!r}")
    regimes, label_fn = _REGIME_SPECS[spec]
    label = label_fn(dict(features))
    n = len(regimes)
    probs = {r: (1.0 - smooth) if r == label else smooth / (n - 1) for r in regimes}
    return RegimeProbs(regimes=regimes, probabilities=probs, label=label, spec=spec)


# ---------------------------------------------------------------------------
# exp50 extension 3: semi-supervised regimes (cluster LOSS, not features)
# ---------------------------------------------------------------------------


def loss_regime_spec(
    records: list[dict],
    models: list[str],
    n_regimes: int = 3,
    seed: int = 42,
) -> dict:
    """Learn regimes by clustering per-model LOSS vectors (exp50).

    exp45 failed because GMM clustered *feature* density — statistically
    coherent groups that weren't *model-relevant*. This clusters the loss
    vectors instead: regime = "the set of series where ARIMA wins".

    Args:
        records: each {"crps": {model: float}}.
        models: ordered model names (defines loss-vector columns).
        n_regimes: number of loss clusters.
        seed: random seed.

    Returns:
        {"labels": [...], "centers": (n_regimes, n_models),
         "winner_per_regime": [...], "model": KMeans}
        where winner_per_regime[r] is the model with lowest center loss.

    """
    from sklearn.cluster import KMeans

    loss_mat = np.array([[float(r["crps"][m]) for m in models] for r in records], dtype=float)
    # normalize per-record so clustering sees *relative* model strength,
    # not absolute difficulty
    row_min = loss_mat.min(axis=1, keepdims=True)
    row_max = loss_mat.max(axis=1, keepdims=True)
    denom = np.maximum(row_max - row_min, 1e-12)
    normed = (loss_mat - row_min) / denom

    km = KMeans(n_clusters=n_regimes, n_init=10, random_state=seed)
    labels = km.fit_predict(normed)
    centers = km.cluster_centers_
    winner_per_regime = [models[int(np.argmin(centers[r]))] for r in range(n_regimes)]
    return {
        "labels": labels.tolist(),
        "centers": centers,
        "winner_per_regime": winner_per_regime,
        "model": km,
        "models": list(models),
    }


def loss_regime_classifier(
    records: list[dict],
    feature_names: list[str],
    loss_spec: dict,
    seed: int = 42,
) -> Any:
    """Train features -> loss-regime classifier (exp50).

    Maps diagnostic features to the loss-cluster labels from
    :func:`loss_regime_spec`, so at predict time we can assign a new
    series to a model-relevant regime from features alone.
    """
    from sklearn.linear_model import LogisticRegression

    x_mat = np.array([r["features"] for r in records], dtype=float)
    y = np.array(loss_spec["labels"])
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x_mat, y)
    return clf
