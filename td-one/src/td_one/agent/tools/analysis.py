"""Statistical analysis tools: residuals, importance, stationarity, decomposition.

These are the "data scientist" primitives the agent uses to understand
what a model is getting wrong and why.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from td_one.agent.tool_tracker import tracked


class ResidualAnalysis(BaseModel):
    """Structured output of :func:`analyze_residuals`."""

    n: int
    rmse: float
    mae: float
    mean_residual: float
    breusch_pagan_lm: float = Field(description="LM statistic for heteroskedasticity")
    breusch_pagan_pvalue: float
    heteroskedastic: bool = Field(description="True if BP p-value < 0.05")
    top_features: list[dict[str, Any]] = Field(
        description="Top-3 features by |corr| with |residual|, each {name, corr}"
    )


def _as_1d_float(values: Any, name: str, min_len: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} observations, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


@tracked()
def analyze_residuals(
    y_true: Any,
    y_pred: Any,
    features: dict[str, Any] | pd.DataFrame | None = None,
) -> ResidualAnalysis:
    """Analyze forecast residuals for bias and heteroskedasticity.

    Runs the Breusch-Pagan test and correlates ``|residual|`` against each
    supplied feature to suggest where the model is systematically wrong.

    Args:
        y_true: Observed values.
        y_pred: Predicted values (same length as y_true).
        features: Optional mapping of feature name -> values (same length),
            or a DataFrame. Used to find residual drivers.

    Returns:
        ResidualAnalysis with error metrics, BP test, and top-3 features.

    """
    yt = _as_1d_float(y_true, "y_true")
    yp = _as_1d_float(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} and y_pred {yp.shape} must match")

    resid = yt - yp
    abs_resid = np.abs(resid)

    # Breusch-Pagan: regress squared residuals on a constant + time trend.
    # (Full auxiliary regression on features would need the design matrix;
    # the trend version detects the common "error grows over horizon" case.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan  # type: ignore[import-untyped]

            t = np.arange(yt.size, dtype=float)
            x_aux = np.column_stack([np.ones(yt.size), t])
            lm, lm_p, _, _ = het_breuschpagan(resid, x_aux)
        except Exception:  # noqa: BLE001 - degenerate input, fall back to NaN
            lm, lm_p = float("nan"), float("nan")

    top: list[dict[str, Any]] = []
    if features is not None:
        frame = features if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
        corrs: list[tuple[str, float]] = []
        for col in frame.columns:
            try:
                f = _as_1d_float(frame[col].to_numpy(), f"features[{col}]", min_len=yt.size)
            except ValueError:
                continue
            if f.shape[0] != yt.shape[0] or np.std(f) == 0:
                continue
            c = float(np.corrcoef(f, abs_resid)[0, 1])
            if np.isfinite(c):
                corrs.append((str(col), c))
        corrs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        top = [{"name": n, "corr": c} for n, c in corrs[:3]]

    return ResidualAnalysis(
        n=int(yt.size),
        rmse=float(np.sqrt(np.mean(resid**2))),
        mae=float(np.mean(abs_resid)),
        mean_residual=float(np.mean(resid)),
        breusch_pagan_lm=float(lm),
        breusch_pagan_pvalue=float(lm_p),
        heteroskedastic=bool(np.isfinite(lm_p) and lm_p < 0.05),
        top_features=top,
    )


class ImportanceResult(BaseModel):
    """Structured output of :func:`feature_importance`."""

    features: list[dict[str, Any]] = Field(
        description="Ranked [{name, importance}] by permutation importance"
    )
    baseline_score: float
    n_repeats: int


@tracked()
def feature_importance(
    predict_fn: Any,
    X: Any,  # noqa: N803 - X is the universal ML convention for feature matrices
    y: Any,
    feature_names: list[str] | None = None,
    n_repeats: int = 5,
    seed: int = 0,
    score_fn: Any | None = None,
) -> ImportanceResult:
    """Permutation importance for any callable predictor.

    Model-agnostic (no sklearn/SHAP dependency): shuffles each column and
    measures the drop in score. Works with any ``predict_fn(X) -> y_pred``.

    Args:
        predict_fn: Callable mapping a 2-D array to predictions.
        X: Feature matrix (n, p).
        y: Targets (n,).
        feature_names: Optional column names; defaults to f0..fp-1.
        n_repeats: Permutation repeats per feature.
        seed: RNG seed for reproducibility.
        score_fn: ``(y_true, y_pred) -> float`` where higher is better.
            Defaults to negative MAE.

    Returns:
        ImportanceResult ranked by mean importance (score drop).

    """
    xa = np.asarray(X, dtype=float)
    if xa.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {xa.shape}")
    ya = _as_1d_float(y, "y", min_len=10)
    if xa.shape[0] != ya.shape[0]:
        raise ValueError("X and y row counts must match")
    if not np.all(np.isfinite(xa)):
        raise ValueError("X contains NaN or inf")

    n, p = xa.shape
    names = feature_names or [f"f{i}" for i in range(p)]
    if len(names) != p:
        raise ValueError(f"{len(names)} names for {p} features")

    score = score_fn or (lambda a, b: -float(np.mean(np.abs(a - b))))
    rng = np.random.default_rng(seed)

    base_pred = np.asarray(predict_fn(xa), dtype=float).ravel()
    if base_pred.shape[0] != n:
        raise ValueError("predict_fn must return one prediction per row")
    baseline = float(score(ya, base_pred))

    importances: list[tuple[str, float]] = []
    for j in range(p):
        drops: list[float] = []
        for _ in range(n_repeats):
            xp = xa.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            pred = np.asarray(predict_fn(xp), dtype=float).ravel()
            drops.append(baseline - float(score(ya, pred)))
        importances.append((names[j], float(np.mean(drops))))
    importances.sort(key=lambda kv: kv[1], reverse=True)

    return ImportanceResult(
        features=[{"name": n_, "importance": v} for n_, v in importances],
        baseline_score=baseline,
        n_repeats=n_repeats,
    )


class StationarityResult(BaseModel):
    """Structured output of :func:`test_stationarity`."""

    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float
    verdict: str = Field(
        description="one of: stationary, trend_stationary, non_stationary, inconclusive"
    )


@tracked()
def test_stationarity(series: Any, fast: bool = False) -> StationarityResult:
    """ADF + KPSS stationarity tests.

    The two tests have opposite nulls (ADF: unit root; KPSS: stationary),
    so together they distinguish level-stationary, trend-stationary, and
    non-stationary series.

    Args:
        series: 1-D numeric series (min 20 observations).
        fast: If True, run ADF only and skip KPSS (KPSS is the slow half;
            ~40ms vs ~150ms). Verdict is ADF-based. Default False.

    Returns:
        StationarityResult with both statistics and a combined verdict.
        In fast mode the KPSS fields are NaN.

    """
    x = _as_1d_float(series, "series", min_len=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from statsmodels.tsa.stattools import adfuller  # type: ignore[import-untyped]

            adf_stat, adf_p = adfuller(x, autolag="AIC")[:2]
        except Exception as exc:  # pragma: no cover - degenerate input
            raise ValueError(f"stationarity tests failed: {exc}") from exc
        if fast:
            kpss_stat, kpss_p = float("nan"), float("nan")
        else:
            try:
                from statsmodels.tsa.stattools import kpss  # type: ignore[import-untyped]

                kpss_stat, kpss_p, _, _ = kpss(x, regression="c", nlags="auto")
            except Exception as exc:  # pragma: no cover - degenerate input
                raise ValueError(f"stationarity tests failed: {exc}") from exc

    adf_p = float(adf_p)
    kpss_p = float(kpss_p)
    if fast:
        verdict = "stationary" if adf_p < 0.05 else "non_stationary"
    elif adf_p < 0.05 and kpss_p >= 0.05:
        verdict = "stationary"
    elif adf_p >= 0.05 and kpss_p < 0.05:
        verdict = "non_stationary"
    elif adf_p < 0.05 and kpss_p < 0.05:
        verdict = "trend_stationary"
    else:
        verdict = "inconclusive"
    return StationarityResult(
        adf_stat=float(adf_stat),
        adf_pvalue=adf_p,
        kpss_stat=float(kpss_stat),
        kpss_pvalue=kpss_p,
        verdict=verdict,
    )


class DecompositionResult(BaseModel):
    """Structured output of :func:`decompose_series`."""

    trend: list[float]
    seasonal: list[float]
    resid: list[float]
    trend_strength: float = Field(ge=0.0, le=1.0)
    seasonal_strength: float = Field(ge=0.0, le=1.0)
    period: int


@tracked()
def decompose_series(series: Any, period: int = 12) -> DecompositionResult:
    """STL decomposition into trend / seasonal / residual.

    Also reports strength metrics in [0, 1] (Wang, Smith & Hyndman):
    strength = max(0, 1 - Var(resid) / Var(deseasonalised or detrended)).

    Args:
        series: 1-D numeric series (min 2*period observations).
        period: Seasonal period (default 12 for monthly data).

    Returns:
        DecompositionResult with components and strength metrics.

    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    # _as_1d_float enforces len >= 2*period with a clear message.
    x = _as_1d_float(series, "series", min_len=2 * period)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from statsmodels.tsa.seasonal import STL  # type: ignore[import-untyped]

        res = STL(x, period=period, robust=True).fit()
    trend = np.asarray(res.trend, dtype=float)
    seasonal = np.asarray(res.seasonal, dtype=float)
    resid = np.asarray(res.resid, dtype=float)

    def _strength(component: np.ndarray, combined: np.ndarray) -> float:
        v_res = float(np.nanvar(resid))
        v_comb = float(np.nanvar(combined))
        if v_comb <= 0 or not np.isfinite(v_comb):
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - v_res / v_comb)))

    deseason = x - seasonal
    detrend = x - trend
    return DecompositionResult(
        trend=[float(v) for v in trend],
        seasonal=[float(v) for v in seasonal],
        resid=[float(v) for v in resid],
        trend_strength=_strength(trend, deseason),
        seasonal_strength=_strength(seasonal, detrend),
        period=period,
    )
