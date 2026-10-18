"""Evaluation tools: PIT-safe time-series CV and model comparison.

``cross_validate`` runs rolling-origin backtests so the agent can honestly
evaluate any proposed change. ``compare_models`` runs several models on the
same splits and tests pairwise differences with the Diebold-Mariano test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from td_one.agent.tool_tracker import tracked


class FoldResult(BaseModel):
    """One rolling-origin fold."""

    origin: int
    train_size: int
    horizon: int
    crps: float


class CVResult(BaseModel):
    """Structured output of :func:`cross_validate`."""

    model_name: str
    n_folds: int
    mean_crps: float
    se_crps: float
    folds: list[FoldResult]


def _crps_from_quantiles(y: float, quantiles: dict[float, float]) -> float:
    """CRPS via the quantile (pinball) representation.

    CRPS = 2 * mean_l pinball_l(y, q_l) over quantile levels.
    """
    levels = sorted(quantiles.keys())
    if len(levels) < 2:
        raise ValueError("need at least 2 quantile levels for CRPS")
    total = 0.0
    for lvl in levels:
        q = float(quantiles[lvl])
        total += (y - q) * (lvl - float(y < q))
    return 2.0 * total / len(levels)


@tracked()
def cross_validate(
    model_fn: Callable[[np.ndarray, int], dict[float, float]],
    series: Any,
    n_splits: int = 5,
    horizon: int = 1,
    min_train: int = 24,
    model_name: str = "model",
) -> CVResult:
    """PIT-safe rolling-origin cross-validation.

    Each fold trains on ``series[:origin]`` and evaluates the 1-step (or
    ``horizon``-step) forecast at ``origin``. Origins are spaced so folds
    do not overlap in their test points.

    Args:
        model_fn: ``(train_array, horizon) -> {level: quantile}``. Must only
            use the training array (PIT safety is the caller's contract, but
            the rolling split guarantees no future leakage by construction).
        series: 1-D numeric series.
        n_splits: Number of rolling origins.
        horizon: Forecast horizon in steps.
        min_train: Minimum training observations for the first fold.
        model_name: Label for the result.

    Returns:
        CVResult with mean CRPS ± standard error across folds.

    """
    x = np.asarray(series, dtype=float).ravel()
    if x.size < min_train + n_splits * horizon:
        raise ValueError(
            f"series length {x.size} too short for min_train={min_train}, "
            f"n_splits={n_splits}, horizon={horizon}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains NaN or inf")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    # Origins: last n_splits test points, spaced by horizon.
    last_test = x.size - horizon
    origins = [last_test - (n_splits - 1 - k) * horizon for k in range(n_splits)]
    folds: list[FoldResult] = []
    for origin in origins:
        if origin < min_train:
            raise ValueError(f"origin {origin} < min_train {min_train}")
        train = x[:origin]
        y_test = x[origin : origin + horizon]
        quantiles = model_fn(train, horizon)
        # Score each horizon step against its own quantile forecast.
        # model_fn returns one quantile dict; for horizon > 1 we score the
        # mean of per-step CRPS reusing the same quantiles (documented limit).
        crps_vals = [_crps_from_quantiles(float(y), quantiles) for y in y_test]
        folds.append(
            FoldResult(
                origin=origin,
                train_size=origin,
                horizon=horizon,
                crps=float(np.mean(crps_vals)),
            )
        )
    crps = np.array([f.crps for f in folds])
    return CVResult(
        model_name=model_name,
        n_folds=len(folds),
        mean_crps=float(np.mean(crps)),
        se_crps=float(np.std(crps, ddof=1) / np.sqrt(len(crps))),
        folds=folds,
    )


class PairwiseComparison(BaseModel):
    """Diebold-Mariano comparison of two models."""

    model_a: str
    model_b: str
    mean_loss_diff: float = Field(description="mean(loss_a - loss_b); negative favors A")
    dm_stat: float
    p_value: float
    significant: bool
    winner: str | None


class ModelComparisonResult(BaseModel):
    """Structured output of :func:`compare_models`."""

    models: list[CVResult]
    pairwise: list[PairwiseComparison]
    ranking: list[str] = Field(description="Model names sorted by mean CRPS ascending")


def _diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """DM test with Harvey-Leybourne-Newbold small-sample correction."""
    from scipy import stats

    d = loss_a - loss_b
    n = d.size
    if n < 3:
        raise ValueError("need at least 3 folds for the DM test")
    mean_d = float(np.mean(d))
    # HAC variance with lag truncation floor(sqrt(n)).
    h = max(1, int(np.floor(np.sqrt(n))))
    gamma0 = float(np.var(d, ddof=1))
    acov = gamma0
    for lag in range(1, h + 1):
        acov += 2.0 * (1.0 - lag / (h + 1)) * float(np.cov(d[:-lag], d[lag:], bias=True)[0, 1])
    var_d = acov / n
    if var_d <= 0 or not np.isfinite(var_d):
        return 0.0, 1.0
    dm = mean_d / np.sqrt(var_d)
    # HLN correction.
    dm_adj = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = float(2.0 * stats.t.sf(abs(dm_adj), df=n - 1))
    return float(dm_adj), p


@tracked()
def compare_models(
    model_fns: dict[str, Callable[[np.ndarray, int], dict[float, float]]],
    series: Any,
    n_splits: int = 5,
    horizon: int = 1,
    min_train: int = 24,
) -> ModelComparisonResult:
    """Run several models on identical CV splits and compare pairwise.

    Args:
        model_fns: Mapping of model name -> model_fn (same contract as
            :func:`cross_validate`).
        series: 1-D numeric series.
        n_splits: Rolling origins per model.
        horizon: Forecast horizon.
        min_train: Minimum training size.

    Returns:
        ModelComparisonResult with per-model CV, DM pairwise tests, and a
        CRPS ranking.

    """
    if len(model_fns) < 2:
        raise ValueError("need at least 2 models to compare")
    results = [
        cross_validate(fn, series, n_splits, horizon, min_train, model_name=name)
        for name, fn in model_fns.items()
    ]
    ranking = sorted(results, key=lambda r: r.mean_crps)
    pairwise: list[PairwiseComparison] = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            la = np.array([f.crps for f in a.folds])
            lb = np.array([f.crps for f in b.folds])
            dm, p = _diebold_mariano(la, lb)
            sig = bool(p < 0.05)
            winner: str | None = None
            if sig:
                winner = a.model_name if float(np.mean(la - lb)) < 0 else b.model_name
            pairwise.append(
                PairwiseComparison(
                    model_a=a.model_name,
                    model_b=b.model_name,
                    mean_loss_diff=float(np.mean(la - lb)),
                    dm_stat=dm,
                    p_value=p,
                    significant=sig,
                    winner=winner,
                )
            )
    return ModelComparisonResult(
        models=results,
        pairwise=pairwise,
        ranking=[r.model_name for r in ranking],
    )


# ---------------------------------------------------------------------------
# combine_quantiles
# ---------------------------------------------------------------------------


class CombinedQuantiles(BaseModel):
    """Structured output of :func:`combine_quantiles`."""

    levels: list[float] = Field(description="Quantile levels, ascending")
    values: list[float] = Field(description="Vincentized combined quantiles")
    weights_used: dict[str, float] = Field(description="Normalized weights")


@tracked()
def combine_quantiles(
    model_quantiles: dict[str, Any],
    weights: dict[str, float] | None = None,
    levels: list[float] | None = None,
) -> CombinedQuantiles:
    """Vincentized combination of per-model quantile forecasts.

    Each model's forecast is a set of quantiles; the combined forecast
    averages quantiles level-by-level (Vincentization). This is what
    exp42 implemented manually — now one canonical implementation.

    Args:
        model_quantiles: Mapping model name -> quantiles, where each
            value is either a ``{level: value}`` dict or a list aligned
            with ``levels``.
        weights: Optional mapping model name -> weight. Defaults to
            equal weights. Normalized internally.
        levels: Quantile levels, required when values are lists.

    Returns:
        CombinedQuantiles with the averaged lattice and weights used.

    """
    if not model_quantiles:
        raise ValueError("model_quantiles is empty")

    parsed: dict[str, dict[float, float]] = {}
    ref_levels: list[float] | None = None
    for name, qs in model_quantiles.items():
        if isinstance(qs, dict):
            d = {float(k): float(v) for k, v in qs.items()}
        elif isinstance(qs, (list, tuple)):
            if levels is None:
                raise ValueError(
                    f"levels required when quantiles for {name!r} are a list"
                )
            d = {float(k): float(v) for k, v in zip(levels, qs, strict=True)}
        else:
            raise ValueError(f"quantiles for {name!r} must be dict or list")
        lvls = sorted(d)
        if ref_levels is None:
            ref_levels = lvls
        elif lvls != ref_levels:
            raise ValueError(
                f"level mismatch for {name!r}: {lvls} vs {ref_levels}"
            )
        parsed[name] = d

    assert ref_levels is not None
    if weights is None:
        w = {m: 1.0 / len(parsed) for m in parsed}
    else:
        missing = set(parsed) - set(weights)
        if missing:
            raise ValueError(f"weights missing for models: {sorted(missing)}")
        total = float(sum(weights[m] for m in parsed))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = {m: float(weights[m]) / total for m in parsed}

    combined = [
        sum(w[m] * parsed[m][q] for m in parsed) for q in ref_levels
    ]
    return CombinedQuantiles(
        levels=ref_levels, values=combined, weights_used=w
    )
