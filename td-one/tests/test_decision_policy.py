"""Tests for the Phase-3 decision policy (TDD)."""

from td_one.agent.decision_policy import (
    NEWS_WINDOW_DAYS,
    T_STAT_STRICT,
    calibrated_quantiles,
    corroboration_gate,
    shrink_shift,
)


def test_shrink_shift_zero_n() -> None:
    assert shrink_shift(0.05, 0) == 0.0


def test_shrink_shift_monotone_in_n() -> None:
    s3 = shrink_shift(0.05, 3)
    s10 = shrink_shift(0.05, 10)
    s30 = shrink_shift(0.05, 30)
    assert 0 < s3 < s10 < s30 < 0.05
    # n=3 -> factor 3/13 ≈ 0.23
    assert abs(s3 - 0.05 * 3 / 13) < 1e-9


def test_shrink_shift_more_prior_var_more_shrinkage() -> None:
    assert shrink_shift(0.05, 5, prior_var=4.0) < shrink_shift(0.05, 5, prior_var=1.0)


def test_calibrated_quantiles_floor() -> None:
    # Degenerate 3-point sd must not produce a degenerate interval.
    q = calibrated_quantiles(
        level_mean=0.10, level_sd=0.005, level_n=3, hist_vol=0.04
    )
    half = q[0.9] - q[0.5]
    # floor: 0.5 * 0.04 * 1.2816 ≈ 0.0256
    assert half >= 0.5 * 0.04 * 1.2815515655446004 - 1e-9
    assert q[0.1] < q[0.5] < q[0.9]


def test_calibrated_quantiles_respects_large_sd() -> None:
    q = calibrated_quantiles(
        level_mean=0.10, level_sd=0.10, level_n=30, hist_vol=0.04
    )
    # sd term: 1.2816 * 0.10 * sqrt(31/30) ≈ 0.1303 > floor 0.0256
    assert q[0.9] - q[0.5] > 0.10


def test_calibrated_quantiles_clipped() -> None:
    q = calibrated_quantiles(
        level_mean=0.995, level_sd=0.05, level_n=10, hist_vol=0.04
    )
    assert q[0.9] <= 0.99
    assert q[0.1] >= 0.001


def test_gate_tiny_shift_rejected() -> None:
    adjust, reason = corroboration_gate(
        t_stat=5.0, shift=0.005, n_news_30d=10
    )
    assert (adjust, reason) == (False, "tiny")


def test_gate_news_allows_lenient_t() -> None:
    adjust, reason = corroboration_gate(
        t_stat=1.8, shift=0.03, n_news_30d=2
    )
    assert (adjust, reason) == (True, "news")


def test_gate_strong_stat_without_news() -> None:
    adjust, reason = corroboration_gate(
        t_stat=T_STAT_STRICT + 0.5, shift=-0.04, n_news_30d=0
    )
    assert (adjust, reason) == (True, "strong_stat")


def test_gate_weak_stat_no_news_rejected() -> None:
    # v2's old p<0.10 threshold (|t|≈1.65) must NOT pass without news.
    adjust, reason = corroboration_gate(
        t_stat=2.0, shift=0.03, n_news_30d=0
    )
    assert (adjust, reason) == (False, "weak")


def test_gate_constants_sane() -> None:
    # Tightened 2026-10-17 per WTI lesson (commit eda6ab8): t>3.5, FP cost 4-7%.
    assert T_STAT_STRICT == 3.5
    assert NEWS_WINDOW_DAYS == 30
