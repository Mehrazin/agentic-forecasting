"""Diagnostic plotting tool: 4-panel forecast diagnostics.

Returns a PNG path the agent (or a human reviewer) can inspect visually.
"""

from __future__ import annotations

from td_one.agent.tool_tracker import tracked

from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field


class DiagnosticPlot(BaseModel):
    """Structured output of :func:`plot_diagnostics`."""

    path: str = Field(description="Absolute path to the PNG file")
    n: int


@tracked()
def plot_diagnostics(
    y_true: Any,
    y_pred: Any,
    dates: Any | None = None,
    path: str | None = None,
    title: str = "Forecast diagnostics",
) -> DiagnosticPlot:
    """Generate a 4-panel diagnostic plot.

    Panels: residuals vs time, QQ plot of residuals, ACF of residuals,
    predicted vs actual scatter.

    Args:
        y_true: Observed values.
        y_pred: Predicted values (same length).
        dates: Optional x-axis labels for the residuals-vs-time panel.
        path: Output PNG path. Defaults to a file under /tmp.
        title: Figure suptitle.

    Returns:
        DiagnosticPlot with the PNG path.

    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.size < 10:
        raise ValueError(f"need at least 10 observations, got {yt.size}")
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if not np.all(np.isfinite(yt)) or not np.all(np.isfinite(yp)):
        raise ValueError("inputs contain NaN or inf")

    resid = yt - yp
    x_axis: Any = np.arange(yt.size)
    if dates is not None:
        try:
            x_axis = np.asarray(list(dates))
            if x_axis.shape[0] != yt.size:
                x_axis = np.arange(yt.size)
        except Exception:  # noqa: BLE001 - fallback to numeric axis
            x_axis = np.arange(yt.size)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(title, fontsize=14)

    # 1. Residuals vs time.
    ax = axes[0, 0]
    ax.plot(x_axis, resid, marker="o", ms=3, lw=0.8)
    ax.axhline(0, color="red", lw=1)
    ax.set_title("Residuals vs time")
    ax.set_xlabel("time")
    ax.set_ylabel("residual")

    # 2. QQ plot.
    ax = axes[0, 1]
    from scipy import stats

    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("QQ plot of residuals")

    # 3. ACF of residuals.
    ax = axes[1, 0]
    from statsmodels.graphics.tsaplots import plot_acf  # type: ignore[import-untyped]

    plot_acf(resid, ax=ax, lags=min(40, yt.size // 2 - 1))
    ax.set_title("ACF of residuals")

    # 4. Predicted vs actual.
    ax = axes[1, 1]
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    ax.scatter(yp, yt, s=12, alpha=0.6)
    ax.plot([lo, hi], [lo, hi], color="red", lw=1)
    ax.set_title("Predicted vs actual")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    import tempfile

    out = path or tempfile.mktemp(prefix="td1_diag_", suffix=".png")  # noqa: S306
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return DiagnosticPlot(path=out, n=int(yt.size))
