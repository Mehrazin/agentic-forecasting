"""Tests for the rolling-origin backtest runner."""

from datetime import date

import pandas as pd
import pytest

from td_one.backtest.runner import (
    RESULT_COLUMNS,
    _check_cutoff,
    run_backtest,
)


def _panel(spec):
    """Build a panel from [(y, m, d, n_renewals, n_switches), ...]."""
    return pd.DataFrame(
        {
            "month": [date(y, m, d) for y, m, d, _, _ in spec],
            "n_renewals": [n for _, _, _, n, _ in spec],
            "n_switches": [a for _, _, _, _, a in spec],
        }
    )


def _const_rate_forecast(rate):
    """Build a dummy forecast_fn returning a flat rate lattice."""

    def fn(train, origin):
        # Contract check: the runner must never leak future months.
        assert train["month"].max() <= pd.Timestamp(origin), "cutoff leak to forecast_fn"
        return {0.10: rate, 0.50: rate, 0.90: rate}

    return fn


@pytest.fixture()
def two_segment_panels():
    a = _panel(
        [
            (2026, 1, 31, 100, 10),
            (2026, 2, 28, 120, 12),
            (2026, 3, 31, 110, 11),
            (2026, 4, 30, 130, 13),
        ]
    )
    b = _panel(
        [
            (2026, 2, 28, 50, 5),
            (2026, 3, 31, 60, 6),
            (2026, 4, 30, 55, 5),
            (2026, 5, 31, 65, 7),
        ]
    )
    return {"seg_a": a, "seg_b": b}


def test_tidy_schema_and_cell_skips(two_segment_panels):
    origins = [date(2026, 2, 28), date(2026, 3, 31), date(2026, 5, 31)]
    out = run_backtest(two_segment_panels, _const_rate_forecast(0.1), origins, seed=1)
    assert list(out.columns) == RESULT_COLUMNS
    # 2026-05-31 has no future month in either panel -> skipped.
    assert len(out) == 4
    assert set(out["segment"]) == {"seg_a", "seg_b"}
    # n_renewals is the *next-month* renewal count.
    row = out[(out["segment"] == "seg_a") & (out["origin"] == date(2026, 2, 28))].iloc[0]
    assert row["n_renewals"] == 110


def test_mae_matches_hand_computation(two_segment_panels):
    origins = [date(2026, 2, 28), date(2026, 3, 31)]
    out = run_backtest(two_segment_panels, _const_rate_forecast(0.1), origins, seed=1)
    # seg_a @ 2026-02-28 -> next month Mar: n=110, actual=11, median rate 0.1.
    row = out[(out["segment"] == "seg_a") & (out["origin"] == date(2026, 2, 28))].iloc[0]
    assert row["mae"] == pytest.approx(abs(0.1 * 110 - 11))
    # seg_b @ 2026-03-31 -> next month Apr: n=55, actual=5.
    row = out[(out["segment"] == "seg_b") & (out["origin"] == date(2026, 3, 31))].iloc[0]
    assert row["mae"] == pytest.approx(abs(0.1 * 55 - 5))


def test_crps_matches_bernoulli_closed_form():
    # N=1: Binomial(1, r) is Bernoulli(r); CRPS vs y=1 is (1-r)^2 exactly.
    panels = {"seg_c": _panel([(2026, 1, 31, 10, 1), (2026, 2, 28, 1, 1)])}
    out = run_backtest(panels, _const_rate_forecast(0.3), [date(2026, 1, 31)], seed=3)
    assert out["crps"].iloc[0] == pytest.approx(0.7**2, abs=0.05)


def test_crps_is_deterministic():
    panels = {"seg_c": _panel([(2026, 1, 31, 10, 1), (2026, 2, 28, 40, 4)])}
    kw = {"forecast_fn": _const_rate_forecast(0.12), "origins": [date(2026, 1, 31)], "seed": 9}
    assert run_backtest(panels, **kw)["crps"].iloc[0] == run_backtest(panels, **kw)["crps"].iloc[0]


def test_coverage_degenerate_forecast():
    # Flat 0.0 rate -> all count samples are 0.
    panels = {
        "hit": _panel([(2026, 1, 31, 20, 0), (2026, 2, 28, 20, 0)]),
        "miss": _panel([(2026, 1, 31, 20, 0), (2026, 2, 28, 20, 5)]),
    }
    out = run_backtest(panels, _const_rate_forecast(0.0), [date(2026, 1, 31)], seed=1)
    assert out.set_index("segment")["coverage_80"]["hit"] == 1.0
    assert out.set_index("segment")["coverage_80"]["miss"] == 0.0


def test_origin_before_panel_start_scores_nothing(two_segment_panels):
    out = run_backtest(two_segment_panels, _const_rate_forecast(0.1), [date(2025, 1, 31)])
    assert list(out.columns) == RESULT_COLUMNS
    assert len(out) == 0


def test_check_cutoff_raises_on_leak():
    train = pd.DataFrame({"month": pd.to_datetime(["2026-03-31"])})
    with pytest.raises(ValueError, match="cutoff violation"):
        _check_cutoff(train, pd.Timestamp("2026-02-28"), "seg_x")
    # At-or-before the origin is fine.
    _check_cutoff(train, pd.Timestamp("2026-03-31"), "seg_x")


def test_rejects_bad_inputs(two_segment_panels):
    with pytest.raises(ValueError, match="panels is empty"):
        run_backtest({}, _const_rate_forecast(0.1), [date(2026, 2, 28)])
    with pytest.raises(ValueError, match="origins is empty"):
        run_backtest(two_segment_panels, _const_rate_forecast(0.1), [])

    bad_counts = {"s": _panel([(2026, 1, 31, 10, 11)])}  # A > N
    with pytest.raises(ValueError, match="0 <= switches <= renewals"):
        run_backtest(bad_counts, _const_rate_forecast(0.1), [date(2026, 1, 31)])

    dup_months = {"s": _panel([(2026, 1, 31, 10, 1), (2026, 1, 31, 5, 0)])}
    with pytest.raises(ValueError, match="duplicate months"):
        run_backtest(dup_months, _const_rate_forecast(0.1), [date(2026, 1, 31)])


def test_rejects_bad_forecast_lattices(two_segment_panels):
    origins = [date(2026, 2, 28)]
    bad_fns = {
        "empty": lambda train, origin: {},
        "level_zero": lambda train, origin: {0.0: 0.1, 0.5: 0.1},
        "negative_rate": lambda train, origin: {0.5: -0.1},
        "rate_above_one": lambda train, origin: {0.5: 1.5},
        "non_monotonic": lambda train, origin: {0.1: 0.5, 0.5: 0.2, 0.9: 0.8},
    }
    for fn in bad_fns.values():
        with pytest.raises(ValueError, match="forecast"):
            run_backtest(two_segment_panels, fn, origins)
