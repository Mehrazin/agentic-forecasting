"""Tests for segment discovery (td_one.segments.discovery).

Uses a small simulated population (300 customers x 66 months) so maturity
coverage is realistic (~76% of mortgages mature in-window).
"""

import pandas as pd
import pytest

from td_one.data_gen.simulator import MortgageSimulator
from td_one.segments.discovery import build_segment_panels, discover_segments


@pytest.fixture(scope="module")
def sim_data():
    """Small simulated histories + outcomes (module-scoped: ~2s to build)."""
    sim = MortgageSimulator(n_customers=300, n_months=66, seed=7)
    histories = sim.generate_history()
    outcomes = sim.generate_outcomes(histories)
    return histories, outcomes


@pytest.fixture(scope="module")
def segments(sim_data):
    histories, outcomes = sim_data
    return discover_segments(histories, outcomes)


def test_discovers_expected_dimensions(segments):
    keys = set(segments)
    assert any(k.startswith("payment_shock_q") for k in keys)
    assert {"rate_type_fixed", "rate_type_variable"} <= keys
    assert any(k.startswith("region_") for k in keys)
    assert {"term_short", "term_medium", "term_long"} <= keys
    # 5 shock quintiles + 2 rate types + 5 provinces + 3 term buckets
    assert len(segments) == 15


def test_panel_schema(segments):
    for key, panel in segments.items():
        assert list(panel.columns) == ["month", "n_renewals", "n_switches"], key
        assert panel["month"].is_monotonic_increasing, key
        assert not panel["month"].duplicated().any(), key
        assert (panel["n_switches"] >= 0).all(), key
        assert (panel["n_switches"] <= panel["n_renewals"]).all(), key


def test_each_dimension_partitions_all_renewals(segments, sim_data):
    _, outcomes = sim_data
    total = len(outcomes)
    for prefix in ("payment_shock", "rate_type", "region", "term"):
        dim_total = sum(
            panel["n_renewals"].sum()
            for key, panel in segments.items()
            if key == prefix or key.startswith(prefix + "_")
        )
        assert dim_total == total, prefix


def test_payment_shock_quintiles_balanced(segments, sim_data):
    _, outcomes = sim_data
    quintiles = {k: v for k, v in segments.items() if k.startswith("payment_shock_q")}
    assert len(quintiles) == 5
    counts = [int(p["n_renewals"].sum()) for p in quintiles.values()]
    # qcut => roughly equal bins; each holds a meaningful share, none empty.
    for c in counts:
        assert c >= 0.10 * len(outcomes), counts


def test_empty_inputs_raise(sim_data):
    histories, outcomes = sim_data
    with pytest.raises(ValueError, match="histories is empty"):
        discover_segments([], outcomes)
    with pytest.raises(ValueError, match="outcomes is empty"):
        discover_segments(histories, [])


def test_build_segment_panels_groups_counts():
    renewals = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-31", "2026-01-31", "2026-02-28"]).date,
            "switched_lender_flag": [True, False, True],
            "dim": ["a", "a", "b"],
        }
    )
    panels = build_segment_panels(renewals, "dim", "seg")
    assert set(panels) == {"seg_a", "seg_b"}
    a = panels["seg_a"]
    assert list(a["n_renewals"]) == [2]
    assert list(a["n_switches"]) == [1]
    assert list(panels["seg_b"]["n_renewals"]) == [1]


def test_build_segment_panels_rejects_bad_input():
    with pytest.raises(ValueError, match="renewals is empty"):
        build_segment_panels(pd.DataFrame(), "dim", "seg")
    df = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-31"]).date,
            "switched_lender_flag": [True],
        }
    )
    with pytest.raises(ValueError, match="dim_col"):
        build_segment_panels(df, "nope", "seg")


def test_observed_asof_masks_unobserved_outcomes(sim_data):
    """Regression (leakage).

    With observed_asof=T, past event months count only outcomes observed
    by T; future event months keep actuals as scoring targets.
    """
    histories, outcomes = sim_data
    full = discover_segments(histories, outcomes)
    months = sorted({m for p in full.values() for m in p["month"]})
    asof = months[len(months) // 2]
    masked = discover_segments(histories, outcomes, observed_asof=asof)

    out = pd.DataFrame([r.model_dump() for r in outcomes])
    out = out[out["switched_lender_flag"].notna()]
    obs = pd.to_datetime(out["outcome_observed_at"])
    event_me = pd.to_datetime(out["scheduled_maturity_date"]) + pd.offsets.MonthEnd(0)
    asof_ts = pd.Timestamp(asof)
    # Kept rows: future event months (actuals, scoring targets) + past
    # months observed by asof.
    kept_switch = out["switched_lender_flag"].astype(bool) & (
        (event_me > asof_ts) | (obs.notna() & (obs <= asof_ts))
    )
    # rate_type_* partitions every renewal, so its switch sums are totals.
    masked_total = sum(
        int(p["n_switches"].sum()) for k, p in masked.items() if k.startswith("rate_type_")
    )
    assert masked_total == int(kept_switch.sum())
    full_total = sum(
        int(p["n_switches"].sum()) for k, p in full.items() if k.startswith("rate_type_")
    )
    assert full_total >= masked_total

    # A future event month keeps identical actuals under masking.
    future_month = months[-1]
    assert future_month > asof
    for key in [k for k in full if k.startswith("rate_type_")]:
        row_full = full[key].loc[full[key]["month"] == future_month].iloc[0]
        row_masked = masked[key].loc[masked[key]["month"] == future_month].iloc[0]
        assert row_full["n_renewals"] == row_masked["n_renewals"]
        assert row_full["n_switches"] == row_masked["n_switches"]
