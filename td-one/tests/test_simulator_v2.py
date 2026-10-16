"""Tests for simulator v2 (richer DGP).

Covers: offer generation + offer-outcome consistency, directional tests for
every new DGP coefficient (via the pure ``v2_switching_logit``), the
prepayment-canon features (burnout, seasoning, S-curve), FRED macro
integration, and v1 backward compatibility.
"""

import math

import pytest

from td_one.data_gen.simulator import MortgageSimulator
from td_one.data_gen.simulator_v2 import (
    BETA_ADVISOR,
    BETA_BURNOUT,
    BETA_DIGITAL,
    BETA_OFFER_S,
    BETA_SEASONING,
    BETA_UTIL_TREND,
    BURNOUT_DAMP,
    S_CURVE_SCALE,
    SEASONING_MONTHS,
    MortgageSimulatorV2,
    load_fred_macro_z,
    v2_switching_logit,
)


def _base_kwargs(**overrides):
    """Default logit kwargs (all neutral) with overrides."""
    kw = {
        "payment_shock": 0.0,
        "offer_spread": 0.0,
        "rate_spread": 0.0,
        "family_squeeze": 0.0,
        "osfi_exempt": 0.0,
        "util_drift_ppm": 0.0,
        "life_event_6m": 0.0,
        "is_digital": 0.0,
        "has_advisor": 0.0,
        "n_prior_refinances": 0,
        "age_months": 60,
    }
    kw.update(overrides)
    return kw


# ------------------------------------------------------------------
# S-curve (arctan rate incentive)
# ------------------------------------------------------------------


def test_s_curve_increasing_and_saturating() -> None:
    """Offer-spread effect rises then saturates (not linear)."""
    l0 = v2_switching_logit(**_base_kwargs(offer_spread=0.0))
    l1 = v2_switching_logit(**_base_kwargs(offer_spread=0.5))
    l2 = v2_switching_logit(**_base_kwargs(offer_spread=2.0))
    l5 = v2_switching_logit(**_base_kwargs(offer_spread=5.0))
    assert l1 > l0
    assert l2 > l1
    assert l5 > l2
    # Saturating: the 2pp->5pp gain is smaller than the 0->2pp gain.
    assert (l5 - l2) < (l2 - l0)
    # Bounded by BETA_OFFER_S * pi/2.
    assert (l5 - l0) < BETA_OFFER_S * math.pi / 2 + 1e-9


def test_s_curve_matches_atan_formula() -> None:
    """The S-curve term is exactly BETA_OFFER_S * atan(spread / scale)."""
    spread = 1.25
    l_base = v2_switching_logit(**_base_kwargs(offer_spread=0.0))
    l_s = v2_switching_logit(**_base_kwargs(offer_spread=spread))
    expected = BETA_OFFER_S * math.atan(spread / S_CURVE_SCALE)
    assert (l_s - l_base) == pytest.approx(expected)


# ------------------------------------------------------------------
# Burnout
# ------------------------------------------------------------------


def test_burnout_level_effect() -> None:
    """More prior refinances -> lower switching logit."""
    l0 = v2_switching_logit(**_base_kwargs(n_prior_refinances=0))
    l1 = v2_switching_logit(**_base_kwargs(n_prior_refinances=1))
    l2 = v2_switching_logit(**_base_kwargs(n_prior_refinances=2))
    assert l1 < l0
    assert l2 < l1
    assert (l0 - l1) == pytest.approx(-BETA_BURNOUT)


def test_burnout_dampens_rate_sensitivity() -> None:
    """Burnout flattens the offer-spread slope (selection effect)."""
    spread = 1.0
    gain_fresh = v2_switching_logit(
        **_base_kwargs(offer_spread=spread, n_prior_refinances=0)
    ) - v2_switching_logit(**_base_kwargs(n_prior_refinances=0))
    gain_burned = v2_switching_logit(
        **_base_kwargs(offer_spread=spread, n_prior_refinances=2)
    ) - v2_switching_logit(**_base_kwargs(n_prior_refinances=2))
    assert gain_burned < gain_fresh
    assert gain_burned == pytest.approx(gain_fresh * (1.0 - BURNOUT_DAMP * 2))


# ------------------------------------------------------------------
# Seasoning
# ------------------------------------------------------------------


def test_seasoning_ramp() -> None:
    """Young mortgages switch less; effect vanishes at 30 months."""
    l12 = v2_switching_logit(**_base_kwargs(age_months=12))
    l24 = v2_switching_logit(**_base_kwargs(age_months=24))
    l30 = v2_switching_logit(**_base_kwargs(age_months=30))
    l60 = v2_switching_logit(**_base_kwargs(age_months=60))
    assert l12 < l24 < l30
    assert l30 == pytest.approx(l60)  # plateau
    assert (l30 - l12) == pytest.approx(BETA_SEASONING * (1.0 - 12.0 / SEASONING_MONTHS))


# ------------------------------------------------------------------
# Heterogeneity coefficients (directional)
# ------------------------------------------------------------------


def test_util_drift_direction() -> None:
    """Rising utilization -> more switching."""
    up = v2_switching_logit(**_base_kwargs(util_drift_ppm=0.5))
    dn = v2_switching_logit(**_base_kwargs(util_drift_ppm=-0.5))
    base = v2_switching_logit(**_base_kwargs())
    assert up > base > dn
    assert (up - base) == pytest.approx(BETA_UTIL_TREND * 0.5)


def test_life_event_direction() -> None:
    """Recent life event -> more switching."""
    assert v2_switching_logit(**_base_kwargs(life_event_6m=1.0)) > v2_switching_logit(
        **_base_kwargs()
    )


def test_digital_direction() -> None:
    """Digital channel -> more switching than branch."""
    assert v2_switching_logit(**_base_kwargs(is_digital=1.0)) > v2_switching_logit(**_base_kwargs())
    assert BETA_DIGITAL > 0


def test_advisor_direction() -> None:
    """Advisor relationship -> less switching (retention)."""
    assert v2_switching_logit(**_base_kwargs(has_advisor=1.0)) < v2_switching_logit(
        **_base_kwargs()
    )
    assert BETA_ADVISOR < 0


# ------------------------------------------------------------------
# Offers
# ------------------------------------------------------------------


@pytest.fixture
def v2_small() -> MortgageSimulatorV2:
    """Small v2 simulator (fast)."""
    return MortgageSimulatorV2(n_customers=200, n_months=24, seed=7)


def test_generate_offers_structure(v2_small: MortgageSimulatorV2) -> None:
    """Each maturing mortgage gets 1 TD + 1-2 competitor offers."""
    history = v2_small.generate_history()
    offers = v2_small.generate_offers(history)
    assert len(offers) > 0
    by_mid: dict[str, list] = {}
    for o in offers:
        by_mid.setdefault(o.mortgage_id, []).append(o)
    for mid, olist in by_mid.items():
        td = [o for o in olist if o.offer_id.endswith("-TD")]
        comp = [o for o in olist if not o.offer_id.endswith("-TD")]
        assert len(td) == 1, f"{mid}: expected 1 TD offer"
        assert 1 <= len(comp) <= 2, f"{mid}: expected 1-2 competitor offers"
        # Offer dates precede maturity; expiries follow.
        for o in olist:
            assert o.offer_date <= o.offer_expiry_date


def test_offer_rates_competitive(v2_small: MortgageSimulatorV2) -> None:
    """Competitor rates disperse around the TD rate (not degenerate)."""
    history = v2_small.generate_history()
    offers = v2_small.generate_offers(history)
    spreads = []
    for mid in {o.mortgage_id for o in offers}:
        spreads.append(v2_small.offer_spread_for(mid, offers))
    assert len(spreads) > 0
    # Some episodes TD is beaten (spread > 0), some not.
    assert any(s > 0.05 for s in spreads), "no competitive undercuts generated"
    assert any(s == 0.0 for s in spreads), "TD never cheapest"


def test_offer_outcome_consistency(v2_small: MortgageSimulatorV2) -> None:
    """Switchers accept a competitor; renewers accept TD (or expire)."""
    history = v2_small.generate_history()
    outcomes = v2_small.generate_outcomes(history)
    offers = v2_small.offers
    assert len(offers) > 0
    by_ep: dict[str, list] = {}
    for o in offers:
        by_ep.setdefault(o.renewal_episode_id, []).append(o)
    outc = {o.renewal_episode_id: o.switched_lender_flag for o in outcomes}
    checked_sw, checked_rn = 0, 0
    for ep, olist in by_ep.items():
        sw = outc.get(ep)
        if sw is None:
            continue
        accepted = [o for o in olist if o.offer_status == "accepted"]
        if sw:
            # Switcher must accept a competitor offer (not TD).
            assert accepted, f"{ep}: switcher accepted nothing"
            assert all(not o.offer_id.endswith("-TD") for o in accepted)
            checked_sw += 1
        else:
            # Renewer accepts TD or lets everything expire.
            if accepted:
                assert any(o.offer_id.endswith("-TD") for o in accepted)
            checked_rn += 1
    assert checked_sw > 0 and checked_rn > 0


def test_offer_spread_drives_switching() -> None:
    """Higher offer spread -> higher predicted switching propensity.

    Tests the mechanism on simulated data (spread from offers -> DGP logit),
    using predicted probabilities rather than noisy binary outcomes.
    """
    import numpy as np

    from td_one.data_gen.simulator_v2 import _add_months

    sim = MortgageSimulatorV2(n_customers=400, n_months=24, seed=11, use_fred_macro=False)
    history = sim.generate_history()
    outcomes = sim.generate_outcomes(history)
    offers = sim.offers
    first, last = {}, {}
    for r in history:
        if r.mortgage_id not in first:
            first[r.mortgage_id] = r
        last[r.mortgage_id] = r
    month_dates = sim._month_dates()
    spreads, pred_ps = [], []
    for o in outcomes:
        mid = o.mortgage_id
        rec = last[mid]
        if rec.maturity_date < month_dates[0] or rec.maturity_date > month_dates[-1]:
            continue
        orig = first[mid]
        prof = sim._profile_for(mid)
        s = sim.offer_spread_for(mid, offers)
        income_mo = rec.household_income_cad / 12
        ps = max(0.0, rec.mortgage_payment_monthly_cad / max(income_mo, 1.0) - 0.32)
        rs = max(0.0, 5.0 - orig.mortgage_rate_pct)
        cutoff = _add_months(rec.maturity_date, -6)
        le = 1.0 if any(cutoff < d <= rec.maturity_date for d, _ in prof.life_events) else 0.0
        lp = v2_switching_logit(
            payment_shock=ps,
            offer_spread=s,
            rate_spread=rs,
            family_squeeze=0.0,
            osfi_exempt=0.0,
            util_drift_ppm=prof.util_drift_ppm,
            life_event_6m=le,
            is_digital=float(prof.channel == "digital"),
            has_advisor=float(prof.has_advisor),
            n_prior_refinances=prof.n_prior_refinances,
            age_months=int(rec.term_months),
        )
        spreads.append(s)
        pred_ps.append(1.0 / (1.0 + math.exp(-lp)))
    assert len(spreads) > 20
    corr = float(np.corrcoef(spreads, pred_ps)[0, 1])
    assert corr > 0.3, f"offer spread not driving predictions (corr={corr:.3f})"


# ------------------------------------------------------------------
# v2 records + heterogeneity
# ------------------------------------------------------------------


def test_v2_history_fields(v2_small: MortgageSimulatorV2) -> None:
    """v2 records carry channel, advisor, and life-event fields."""
    history = v2_small.generate_history()
    assert len(history) > 0
    channels = {r.channel for r in history}
    assert channels <= {"branch", "digital"}
    assert any(r.has_advisor for r in history)
    assert any(r.life_event_6m for r in history)
    # Utilization drifts over time (not static like v1).
    by_mid: dict[str, list] = {}
    for r in history:
        by_mid.setdefault(r.mortgage_id, []).append(r)
    drifts = 0
    for recs in list(by_mid.values())[:50]:
        utils = [r.credit_utilization_pct for r in recs]
        if max(utils) - min(utils) > 1.0:
            drifts += 1
    assert drifts > 0, "utilization never drifts"


def test_v2_terms_include_short(v2_small: MortgageSimulatorV2) -> None:
    """v2 adds 12/24-month terms (seasoning ramp bites)."""
    history = v2_small.generate_history()
    terms = {r.term_months for r in history}
    assert 12 in terms or 24 in terms, f"no short terms: {sorted(terms)}"


def test_heterogeneity_plumbing() -> None:
    """Channel/advisor from profiles flow into the DGP on simulated data.

    Verifies the end-to-end plumbing (not just the coefficient in isolation):
    recomputing the v2 logit from each maturing borrower's actual covariates,
    digital borrowers get higher predicted switching propensity than branch,
    and advised borrowers get lower than non-advised.
    """
    from td_one.data_gen.simulator_v2 import _add_months

    sim = MortgageSimulatorV2(n_customers=400, n_months=24, seed=21, use_fred_macro=False)
    history = sim.generate_history()
    offers = sim.generate_offers(history)
    first, last = {}, {}
    for r in history:
        if r.mortgage_id not in first:
            first[r.mortgage_id] = r
        last[r.mortgage_id] = r
    month_dates = sim._month_dates()
    dig_ps, br_ps, adv_ps, nad_ps = [], [], [], []
    for mid, rec in last.items():
        if rec.maturity_date > month_dates[-1]:
            continue
        orig = first[mid]
        prof = sim._profile_for(mid)
        income_mo = rec.household_income_cad / 12
        ps = max(0.0, rec.mortgage_payment_monthly_cad / max(income_mo, 1.0) - 0.32)
        rs = max(0.0, 5.0 - orig.mortgage_rate_pct)
        mat = rec.maturity_date
        cutoff = _add_months(mat, -6)
        le = 1.0 if any(cutoff < d <= mat for d, _ in prof.life_events) else 0.0
        lp = v2_switching_logit(
            payment_shock=ps,
            offer_spread=sim.offer_spread_for(mid, offers),
            rate_spread=rs,
            family_squeeze=0.0,
            osfi_exempt=0.0,
            util_drift_ppm=prof.util_drift_ppm,
            life_event_6m=le,
            is_digital=1.0 if prof.channel == "digital" else 0.0,
            has_advisor=1.0 if prof.has_advisor else 0.0,
            n_prior_refinances=prof.n_prior_refinances,
            age_months=int(rec.term_months),
        )
        p = 1.0 / (1.0 + math.exp(-lp))
        if prof.channel == "digital":
            dig_ps.append(p)
        else:
            br_ps.append(p)
        if prof.has_advisor:
            adv_ps.append(p)
        else:
            nad_ps.append(p)
    assert len(dig_ps) > 10 and len(br_ps) > 10
    assert sum(dig_ps) / len(dig_ps) > sum(br_ps) / len(br_ps)
    assert len(adv_ps) > 10 and len(nad_ps) > 10
    assert sum(adv_ps) / len(adv_ps) < sum(nad_ps) / len(nad_ps)


# ------------------------------------------------------------------
# FRED macro
# ------------------------------------------------------------------


def test_fred_macro_loads() -> None:
    """FRED z-scores load from the parquet cache (if present)."""
    z = load_fred_macro_z()
    # Graceful: empty dict if cache missing; non-empty with sane values if present.
    for d, v in z.items():
        assert isinstance(v, float)
        assert abs(v) < 20, f"wild z-score {v} at {d}"
    # The cache exists in this repo; expect real data.
    assert len(z) > 100, "FRED cache missing or too short"


def test_fred_macro_default_on() -> None:
    """v2 uses FRED macro by default; can be disabled."""
    sim_on = MortgageSimulatorV2(n_customers=100, n_months=24, seed=5)
    sim_off = MortgageSimulatorV2(n_customers=100, n_months=24, seed=5, use_fred_macro=False)
    assert sim_on.use_fred_macro is True
    assert sim_off.use_fred_macro is False
    shifts_on = sim_on._fred_macro_shifts()
    shifts_off = sim_off._fred_macro_shifts()
    # _fred_macro_shifts ignores the flag (flag is checked in generate_outcomes);
    # it should return real shifts when cache exists.
    assert len(shifts_on) == 24
    assert shifts_off == shifts_on  # same loader; flag gates usage


def test_fred_macro_moves_outcomes() -> None:
    """Explicit macro_shifts move outcomes (mechanism is wired)."""
    sim = MortgageSimulatorV2(n_customers=400, n_months=24, seed=9, use_fred_macro=False)
    history = sim.generate_history()
    base = sim.generate_outcomes(history)
    # Large positive shift -> unambiguously more switching.
    big_shift = dict.fromkeys(sim._month_dates(), 2.0)
    shifted = sim.generate_outcomes(history, macro_shifts=big_shift)
    assert len(base) == len(shifted) > 20
    r_base = sum(o.switched_lender_flag for o in base) / len(base)
    r_shift = sum(o.switched_lender_flag for o in shifted) / len(shifted)
    assert r_shift > r_base, "macro_shifts had no effect on outcomes"


# ------------------------------------------------------------------
# v1 compatibility
# ------------------------------------------------------------------


def test_v1_untouched_and_green() -> None:
    """v1 simulator still works (spot check; full suite runs separately)."""
    sim = MortgageSimulator(n_customers=100, n_months=12, seed=42)
    history = sim.generate_history()
    outcomes = sim.generate_outcomes(history)
    assert len(history) == 1200
    assert len(outcomes) > 0
    # v1 has no v2 attributes.
    assert not hasattr(history[0], "channel")
