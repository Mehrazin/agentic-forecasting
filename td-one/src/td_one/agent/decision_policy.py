"""Decision policy for code-executing forecast agents (Phase 3).

exp22's diagnosis: v2 detects shifts well (+6.6% CRPS when its analysis fires)
but is high-variance — 3/9 adjustments hurt badly (-59% to -93%). Two causes:

1. Overconfident intervals: quantile width from 3-point sd is too noisy.
2. No corroboration gating: v2 adjusted on the t-test alone (p<0.10).

This module provides the three Phase-3 policy fixes as pure, testable
functions:

- :func:`shrink_shift` — James-Stein-style shrinkage of the level-shift
  estimate toward zero when the post-break sample is small (n < 10).
- :func:`calibrated_quantiles` — quantile width floored at historical
  volatility so tight recent months can't produce degenerate intervals.
- :func:`corroboration_gate` — only adjust when (a) a news article about a
  relevant event appeared within 30 days, OR (b) the shift t-statistic
  exceeds 3.0 (stricter than v2's p<0.10).

All functions are PIT-safe by construction: they only consume statistics
computed on data with ``asof <= origin``.
"""

from __future__ import annotations

import math

#: Minimum |t| for a statistics-only adjustment (no news corroboration).
T_STAT_STRICT = 3.0
#: v2's old threshold (kept for reference / ablations).
T_STAT_LENIENT = 1.65  # ≈ p < 0.10 two-sided
#: News corroboration window (days before origin).
NEWS_WINDOW_DAYS = 30
#: Below this post-break n, shrinkage pulls the shift hard toward zero.
SHRINK_N_REF = 10


def shrink_shift(
    shift: float,
    n_post: int,
    prior_var: float = 1.0,
    n_ref: int = SHRINK_N_REF,
) -> float:
    """Shrink a level-shift estimate toward zero (James-Stein style).

    The shrinkage factor is ``n_post / (n_post + n_ref * prior_var_ratio)``;
    with the default ``prior_var=1.0`` this is ``n/(n+10)``: n=3 → 0.23,
    n=10 → 0.50, n=30 → 0.75. Small post-break samples get pulled hard
    toward zero (no adjustment); large samples are trusted.

    Args:
        shift: Raw shift estimate (recent_mean - trailing_mean).
        n_post: Number of post-break observations the estimate rests on.
        prior_var: Relative prior variance of the shift (larger → more shrinkage).
        n_ref: Reference sample size at which factor = 1/(1+prior_var).

    Returns:
        Shrunken shift estimate.

    """
    if n_post <= 0:
        return 0.0
    factor = n_post / (n_post + n_ref * prior_var)
    return shift * factor


def calibrated_quantiles(
    level_mean: float,
    level_sd: float,
    level_n: int,
    hist_vol: float,
    z: float = 1.2815515655446004,  # 0.9 quantile of N(0,1)
    min_width_factor: float = 0.5,
) -> dict[float, float]:
    """Build (0.1, 0.5, 0.9) quantiles with volatility-floored width.

    exp22's failure mode: 3-point sd could be tiny (e.g. 0.005), producing a
    degenerate interval (width 0.03 vs m0's 0.23). Right → excellent CRPS;
    wrong → catastrophic CRPS. This floors the interval half-width at
    ``min_width_factor * hist_vol * z`` and additionally inflates the sd by
    ``sqrt(1 + 1/n)`` to account for level-estimation uncertainty.

    Args:
        level_mean: Estimated post-break level.
        level_sd: Sd of the post-break sample.
        level_n: Post-break sample size.
        hist_vol: Historical volatility (sd of the full training series).
        z: Normal quantile for the outer bands (default 0.9).
        min_width_factor: Floor multiplier on historical volatility.

    Returns:
        Dict {0.1: q10, 0.5: q50, 0.9: q90}, clipped to [0.001, 0.99].

    """
    sd_adj = level_sd * math.sqrt(1.0 + 1.0 / max(level_n, 1))
    half_width = max(z * sd_adj, min_width_factor * hist_vol * z)
    q10 = min(0.99, max(0.001, level_mean - half_width))
    q50 = min(0.99, max(0.001, level_mean))
    q90 = min(0.99, max(0.001, level_mean + half_width))
    return {0.1: q10, 0.5: q50, 0.9: q90}


def corroboration_gate(
    t_stat: float,
    shift: float,
    n_news_30d: int,
    min_shift: float = 0.02,
    t_strict: float = T_STAT_STRICT,
    news_window_days: int = NEWS_WINDOW_DAYS,
) -> tuple[bool, str]:
    """Decide whether a detected shift justifies adjusting the forecast.

    Gate (Phase 3, stricter than v2):
    - |shift| must exceed ``min_shift`` (else noise), AND
    - EITHER ``n_news_30d > 0`` (a relevant article within the window —
      the information-asymmetry channel), OR ``|t_stat| > t_strict``
      (overwhelming statistical evidence on its own).

    Args:
        t_stat: Welch t-statistic for recent-vs-trailing means.
        shift: Estimated shift (recent_mean - trailing_mean).
        n_news_30d: News articles about relevant events in the last
            ``news_window_days`` days before origin.
        min_shift: Minimum |shift| to act on (2pp default).
        t_strict: Strict |t| threshold for statistics-only adjustment.
        news_window_days: Corroboration window (informational; the caller
            counts the articles).

    Returns:
        (adjust: bool, reason: str) — reason is "news", "strong_stat",
        "weak" (below thresholds), or "tiny" (|shift| too small).

    """
    _ = news_window_days  # informational; caller does the counting
    if abs(shift) < min_shift:
        return False, "tiny"
    if n_news_30d > 0:
        return True, "news"
    if abs(t_stat) > t_strict:
        return True, "strong_stat"
    return False, "weak"
