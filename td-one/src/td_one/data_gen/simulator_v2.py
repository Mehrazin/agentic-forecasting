"""Simulator v2: richer DGP with renewal offers, competition, and heterogeneity.

Extends :class:`td_one.data_gen.simulator.MortgageSimulator` (v1) without
modifying it — v1's class, tests, and behavior are untouched.

New in v2 (all with real DGP effects, documented coefficients):

1. **Renewal offers table** — one TD offer plus 1-2 competitor offers per
   renewal episode, with rates, terms, and statuses consistent with the
   outcome (switchers accept a competitor; renewers accept TD or let
   offers lapse).
2. **Offer-competition DGP** — switching depends on the spread between TD's
   renewal offer and the best competitor offer (the economically central
   mechanism v1 misses; v1 only saw the origination-rate spread).
3. **Richer heterogeneity** — channel (branch/digital), advisor relationship,
   life events (job change / new child), and credit-utilization drift.
4. **FRED macro by default** — v1's ``macro_shifts`` mechanism is now driven
   by the actual MORTGAGE30US series unless disabled.
5. **Prepayment-canon mechanics** (from the literature survey):
   - **Burnout**: borrowers who already refinanced are less likely to switch
     again (level effect) and less rate-sensitive (selection dampening).
   - **Seasoning**: 30-month ramp — young mortgages switch less; v2 adds
     12/24-month terms so the ramp bites.
   - **S-curve**: the rate incentive enters via arctan (saturating), not linear.

The v2 switching DGP (logit scale)::

    logit(p) = BETA_0
             + BETA_PAYMENT_SHOCK * payment_shock      (v1)
             + BETA_OFFER_S * atan(offer_spread / S_CURVE_SCALE) * burnout_damp
                                                       (NEW: S-curved incentive)
             + V2_BETA_RATE_SPREAD * rate_spread       (v2: reduced; offer is primary)
             + BETA_FAMILY_SQUEEZE * family_squeeze    (v1)
             + BETA_OSFI_EXEMPT * osfi_exempt          (v1)
             + BETA_UTIL_TREND * util_drift            (NEW: rising utilization)
             + BETA_LIFE_EVENT * life_event_6m         (NEW)
             + BETA_DIGITAL * is_digital               (NEW)
             + BETA_ADVISOR * has_advisor              (NEW, negative: retention)
             + BETA_BURNOUT * n_prior_refinances       (NEW: level)
             + BETA_SEASONING * (seasoning - 1)        (NEW: ramp, <= 0)
             + cohort_effects[cohort]                  (v1)
             + fred_macro_shift[maturity_month]       (NEW: on by default)

where ``burnout_damp = 1 - BURNOUT_DAMP * min(n_prior_refinances, 2)`` and
``seasoning = min(1, age_months / SEASONING_MONTHS)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from td_one.data_gen.amortization import canadian_monthly_payment
from td_one.data_gen.simulator import (
    BETA_FAMILY_SQUEEZE,
    BETA_OSFI_EXEMPT,
    BETA_PAYMENT_SHOCK,
    COHORT_SD,
    OUTCOME_LAG_MONTHS,
    MortgageSimulator,
    _month_end,
)
from td_one.schemas import (
    MortgageHistoryRecord,
    MortgageHistoryRecordV2,
    RenewalOffer,
    RenewalOutcomeRecord,
)

# --- v2 DGP coefficients (logit scale; documented, tunable) ---

#: Recalibrated intercept (empirically tuned for ~11% base switch rate;
#: the v2 incentive terms add substantial positive logit vs v1).
V2_BETA_0 = -3.7
#: Per pp/month of utilization drift. +0.5pp/mo -> +0.5 logit.
BETA_UTIL_TREND = 1.0
#: Recent life event (job change / new child within 6m of maturity).
BETA_LIFE_EVENT = 0.6
#: Digital channel (shops more than branch).
BETA_DIGITAL = 0.35
#: Advisor relationship (retention; negative).
BETA_ADVISOR = -0.45

# --- Prepayment-canon coefficients ---

#: Burnout level: per prior refinance (already refinanced -> less likely again).
BETA_BURNOUT = -0.45
#: Burnout selection: fractional dampening of rate sensitivity per prior
#: refinance (capped at 2). Sensitive borrowers already left the pool.
BURNOUT_DAMP = 0.25
#: Seasoning ramp: full seasoning at 30 months (PSA-style).
SEASONING_MONTHS = 30.0
#: Logit penalty scale for unseasoned mortgages (0 at full seasoning).
BETA_SEASONING = 1.0
#: S-curve scale for the arctan rate incentive (pp).
S_CURVE_SCALE = 0.75
#: Logit per unit of arctan S-curve response (saturates at pi/2).
BETA_OFFER_S = 1.2

#: v2 rate-spread coefficient (reduced from v1's 0.9: the offer spread is now
#: the primary incentive; this is a secondary "facing a big increase" control).
V2_BETA_RATE_SPREAD = 0.4

#: v2 term distribution: adds 12/24-month terms so the seasoning ramp bites.
#: (v1: {36: 0.25, 60: 0.65, 84: 0.10}; all fully seasoned at maturity.)
V2_TERM_DIST = {12: 0.05, 24: 0.10, 36: 0.20, 60: 0.55, 84: 0.10}

#: FRED series driving the default macro overlay.
FRED_SERIES_ID = "MORTGAGE30US"
#: Scale of the FRED macro logit component. Applied with a NEGATIVE sign:
#: when mortgage rates spike (positive z), switching falls (lock-in and
#: affordability effects); when rates fall, switching rises (refi incentive).
FRED_BETA = 0.3
#: Winsorization bound for FRED z-scores (the 2022 hiking cycle is ~6 sigma;
#: uncapped it would dominate the DGP).
FRED_Z_CAP = 2.5

#: Offer timing: TD offer goes out this many days before maturity.
TD_OFFER_LEAD_DAYS = 90
#: P(2 competitor offers vs 1).
P_TWO_COMPETITORS = 0.6
#: Market 5y-fixed spread over the BoC policy path (pp).
MARKET_SPREAD_PP = 2.0

_LIFE_EVENT_TYPES = ("job_change", "new_child", "relocation")


def _add_months(d: date, months: int) -> date:
    """Shift a date by a whole number of months (day clamped)."""
    total = (d.year * 12 + d.month - 1) + months
    y, m = divmod(total, 12)
    return date(y, m + 1, min(d.day, _month_end(y, m + 1).day))


def _add_days(d: date, days: int) -> date:
    """Shift a date by a whole number of days."""
    return d + timedelta(days=days)


def load_fred_macro_z(fred_id: str = FRED_SERIES_ID) -> dict[date, float]:
    """Load a FRED series; return z-scored 3m changes keyed by month-end.

    Returns an empty dict if the parquet cache is missing, so v2 degrades
    gracefully to "no macro" in environments without FRED data.
    """
    fred_path = Path(__file__).resolve().parents[3] / "data" / "fred" / f"{fred_id}.parquet"
    if not fred_path.exists():
        return {}
    df = pd.read_parquet(fred_path)
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date")["value"].resample("ME").last().dropna()
    mom3 = monthly.pct_change(3)
    std = float(mom3.std())
    if std <= 0 or np.isnan(std):
        return {}
    z = (mom3 - mom3.mean()) / std
    out: dict[date, float] = {}
    for i in range(len(z)):
        ts = z.index[i]
        val = float(z.iloc[i])
        if not np.isnan(val):
            out[ts.date()] = val
    return out


@dataclass
class _V2Profile:
    """Per-customer v2 attributes (internal; not all are in the record schema)."""

    channel: str = "branch"
    has_advisor: bool = False
    util_drift_ppm: float = 0.0
    life_events: list[tuple[date, str]] = field(default_factory=list)
    n_prior_refinances: int = 0
    td_renewal_premium: float = 0.10


def v2_switching_logit(
    *,
    payment_shock: float,
    offer_spread: float,
    rate_spread: float,
    family_squeeze: float,
    osfi_exempt: float,
    util_drift_ppm: float,
    life_event_6m: float,
    is_digital: float,
    has_advisor: float,
    n_prior_refinances: int,
    age_months: int,
    cohort_effect: float = 0.0,
    macro_shift: float = 0.0,
) -> float:
    """Compute the v2 structural switching logit.

    Pure function (no RNG, no I/O) so the DGP components — S-curve shape,
    burnout level + dampening, seasoning ramp — are unit-testable without
    simulation noise.

    Args:
        payment_shock: Payment-to-income ratio above 0.32 (v1).
        offer_spread: TD-over-best-competitor spread in pp (>= 0).
        rate_spread: Market-minus-origination spread in pp (v1).
        family_squeeze: 1.0 if family-squeeze cohort else 0.0.
        osfi_exempt: 1.0 if OSFI-exempt cohort else 0.0.
        util_drift_ppm: Credit-utilization drift in pp/month.
        life_event_6m: 1.0 if a life event occurred within 6m of maturity.
        is_digital: 1.0 if digital channel else 0.0.
        has_advisor: 1.0 if advisor relationship else 0.0.
        n_prior_refinances: Burnout state (refinances before this episode).
        age_months: Mortgage age at maturity in months (for seasoning).
        cohort_effect: Cohort random effect (v1).
        macro_shift: FRED macro logit shift for the maturity month.

    Returns:
        Logit of the switching probability.

    """
    # Burnout dampens rate sensitivity (selection: sensitive borrowers left).
    damp = 1.0 - BURNOUT_DAMP * min(n_prior_refinances, 2)
    # S-curve: arctan saturates the rate incentive (not linear).
    s_curve = math.atan(max(0.0, offer_spread) / S_CURVE_SCALE)
    # Seasoning ramp: unseasoned mortgages switch less.
    seasoning = min(1.0, age_months / SEASONING_MONTHS)
    return (
        V2_BETA_0
        + BETA_PAYMENT_SHOCK * payment_shock
        + BETA_OFFER_S * s_curve * damp
        + V2_BETA_RATE_SPREAD * rate_spread
        + BETA_FAMILY_SQUEEZE * family_squeeze
        + BETA_OSFI_EXEMPT * osfi_exempt
        + BETA_UTIL_TREND * util_drift_ppm
        + BETA_LIFE_EVENT * life_event_6m
        + BETA_DIGITAL * is_digital
        + BETA_ADVISOR * has_advisor
        + BETA_BURNOUT * n_prior_refinances
        + BETA_SEASONING * (seasoning - 1.0)
        + cohort_effect
        + macro_shift
    )


class MortgageSimulatorV2(MortgageSimulator):
    """Richer DGP: offers, competition, heterogeneity, FRED macro by default."""

    def __init__(
        self,
        n_customers: int = 10000,
        n_months: int = 66,
        start_date: date = date(2021, 4, 30),
        seed: int = 42,
        use_fred_macro: bool = True,
        fred_beta: float = FRED_BETA,
    ) -> None:
        """Initialize the v2 simulator.

        Args:
            n_customers: Number of active mortgage customers.
            n_months: Number of monthly snapshots.
            start_date: First snapshot date (month-end).
            seed: Random seed for reproducibility.
            use_fred_macro: Drive ``macro_shifts`` from FRED by default.
            fred_beta: Scale of the FRED macro logit component.

        """
        super().__init__(
            n_customers=n_customers,
            n_months=n_months,
            start_date=start_date,
            seed=seed,
        )
        self.use_fred_macro = use_fred_macro
        self.fred_beta = fred_beta
        self._v2_profiles: dict[str, _V2Profile] = {}
        self._offers: list[RenewalOffer] = []

    # ------------------------------------------------------------------
    # Customer heterogeneity
    # ------------------------------------------------------------------

    def _generate_customers(self) -> pd.DataFrame:
        """Extend v1 customers with v2 heterogeneity.

        Adds channel, advisor, util drift, burnout state, renewal premium,
        and life events. Also overrides the term distribution with v2's
        (adds 12/24-month terms so the seasoning ramp bites).
        """
        df = super()._generate_customers()
        rng = self.rng
        n = len(df)

        # v2 term distribution (12/24-month terms for seasoning).
        v2_terms = np.array(list(V2_TERM_DIST.keys()))
        v2_term_p = np.array(list(V2_TERM_DIST.values()))
        df["term_months"] = rng.choice(v2_terms, size=n, p=v2_term_p)

        # Channel: younger borrowers skew digital.
        age = df["primary_borrower_age"].to_numpy(dtype=float)
        p_digital = np.clip(0.6 - 0.008 * (age - 25.0), 0.15, 0.6)
        df["channel"] = np.where(rng.random(n) < p_digital, "digital", "branch")

        # Advisor: tenure + income.
        p_adv = np.clip(
            0.25
            + 0.30 * (df["td_tenure_months"].to_numpy() > 60)
            + 0.10 * (df["household_income_cad"].to_numpy() > 150000),
            0.0,
            0.8,
        )
        df["has_advisor"] = rng.random(n) < p_adv

        # Utilization drift (pp/month), symmetric around 0.
        df["util_drift_ppm"] = np.clip(rng.normal(0.0, 0.3, size=n), -1.0, 1.0)

        # Burnout: prior refinances before the window (older originations).
        orig_year = df["origination_year"].to_numpy()
        n_prior = np.zeros(n, dtype=int)
        for i in range(n):
            oy = int(orig_year[i])
            if oy <= 2019:
                n_prior[i] = int(rng.choice([0, 1, 2], p=[0.50, 0.35, 0.15]))
            elif oy == 2020:
                n_prior[i] = int(rng.choice([0, 1], p=[0.70, 0.30]))
            else:
                n_prior[i] = int(rng.choice([0, 1], p=[0.90, 0.10]))
        df["n_prior_refinances"] = n_prior

        # TD renewal premium over market (for market-consistent repricing).
        df["td_renewal_premium"] = rng.uniform(0.0, 0.25, size=n)

        # Life events with absolute dates (need maturity per customer).
        profiles: dict[str, _V2Profile] = {}
        for _, row in df.iterrows():
            mid = str(row["mortgage_id"])
            orig = date(int(row["origination_year"]), int(row["origination_month"]), 15)
            total = (int(row["origination_month"]) - 1) + int(row["term_months"])
            mat = date(
                int(row["origination_year"]) + total // 12,
                total % 12 + 1,
                15,
            )
            n_events = int(rng.poisson(0.8))
            events: list[tuple[date, str]] = []
            span_days = max(1, (mat - orig).days)
            for _ in range(n_events):
                ev_date = _add_days(orig, int(rng.integers(0, span_days)))
                ev_type = str(rng.choice(_LIFE_EVENT_TYPES))
                events.append((ev_date, ev_type))
            events.sort()
            profiles[mid] = _V2Profile(
                channel=str(row["channel"]),
                has_advisor=bool(row["has_advisor"]),
                util_drift_ppm=float(row["util_drift_ppm"]),
                life_events=events,
                n_prior_refinances=int(row["n_prior_refinances"]),
                td_renewal_premium=float(row["td_renewal_premium"]),
            )
        self._v2_profiles = profiles
        return df

    # ------------------------------------------------------------------
    # History (v2 records)
    # ------------------------------------------------------------------

    def generate_history(self) -> list[MortgageHistoryRecord]:
        """Generate v1 history, then apply v2 enrichments.

        - Credit utilization drifts by the customer's ``util_drift_ppm``.
        - ``channel`` / ``has_advisor`` are threaded through.
        - ``life_event_6m`` is computed PIT-correctly from event dates.
        - Episode-02 fixed rates are repriced to market-consistent levels
          (v1 hardcodes 4.5-5.5% regardless of maturity year; v2 uses the
          BoC-path market rate at maturity plus the TD premium, so the
          12/24-month terms maturing in 2022-2023 are priced correctly).

        Returns:
            List of validated records (runtime type MortgageHistoryRecordV2;
            annotated as the parent type for Liskov compatibility).

        """
        v1_records = super().generate_history()
        # Annotated as the parent type for Liskov; runtime type is V2.
        out: list[MortgageHistoryRecord] = []
        for r in v1_records:
            prof = self._profile_for(r.mortgage_id)
            # Utilization drift from origination.
            months_out = (r.snapshot_date.year - r.origination_date.year) * 12 + (
                r.snapshot_date.month - r.origination_date.month
            )
            base_util = r.credit_utilization_pct if r.credit_utilization_pct is not None else 30.0
            new_util = float(np.clip(base_util + prof.util_drift_ppm * months_out, 1.0, 95.0))
            # Life event in the trailing 6 months (PIT: event_date <= snapshot).
            cutoff = _add_months(r.snapshot_date, -6)
            recent = [e for e in prof.life_events if cutoff < e[0] <= r.snapshot_date]
            life_6m = len(recent) > 0
            life_type = recent[-1][1] if recent else "none"
            # Market-consistent renewal pricing (fixed, episode 02).
            new_rate = r.mortgage_rate_pct
            new_payment = r.mortgage_payment_monthly_cad
            if r.renewal_episode_id.endswith("-02") and r.rate_type == "fixed":
                new_rate = float(
                    np.clip(
                        self._market_rate_at(r.maturity_date) + prof.td_renewal_premium,
                        0.5,
                        12.0,
                    )
                )
                new_payment = float(
                    canadian_monthly_payment(
                        float(r.mortgage_balance_cad),
                        new_rate,
                        int(r.remaining_amortization_months),
                    )
                )
            out.append(
                MortgageHistoryRecordV2(
                    **{
                        k: v
                        for k, v in r.model_dump().items()
                        if k
                        not in (
                            "credit_utilization_pct",
                            "mortgage_rate_pct",
                            "mortgage_payment_monthly_cad",
                        )
                    },
                    channel=prof.channel,  # type: ignore[arg-type]
                    has_advisor=prof.has_advisor,
                    life_event_6m=life_6m,
                    life_event_type=life_type,  # type: ignore[arg-type]
                    credit_utilization_pct=round(new_util, 1),
                    mortgage_rate_pct=round(new_rate, 3),
                    mortgage_payment_monthly_cad=round(new_payment, 2),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Market rates + offers
    # ------------------------------------------------------------------

    def _profile_for(self, mortgage_id: str) -> _V2Profile:
        """Return the v2 profile, or defaults if history came from elsewhere."""
        return self._v2_profiles.get(mortgage_id, _V2Profile())

    def _market_rate_at(self, d: date) -> float:
        """Prevailing 5y-fixed market rate at a date (BoC path + spread)."""
        idx = (d.year - self.start_date.year) * 12 + (d.month - self.start_date.month)
        idx = int(np.clip(idx, 0, self.n_months - 1))
        return float(self._boc_path[idx] + MARKET_SPREAD_PP)

    def _assign_offer_statuses(
        self,
        offers: list[RenewalOffer],
        outcomes: list[RenewalOutcomeRecord],
    ) -> list[RenewalOffer]:
        """Return copies of ``offers`` with statuses consistent with ``outcomes``.

        Switchers accept the best competitor offer (TD declined); renewers
        accept the TD offer (or let it expire through inertia). Rates and
        dates are unchanged — only statuses are assigned.
        """
        rng = self.rng
        outcome_by_episode: dict[str, bool] = {}
        for outc in outcomes:
            if outc.switched_lender_flag is not None:
                outcome_by_episode[outc.renewal_episode_id] = bool(outc.switched_lender_flag)
        # Best competitor rate per episode (for "accepted" selection).
        best_comp: dict[str, float] = {}
        for off in offers:
            if not off.offer_id.endswith("-TD"):
                ep = off.renewal_episode_id
                best_comp[ep] = min(best_comp.get(ep, 9e9), off.offered_rate_pct)

        finalized: list[RenewalOffer] = []
        for off in offers:
            ep = off.renewal_episode_id
            switched = outcome_by_episode.get(ep)
            is_td = off.offer_id.endswith("-TD")
            maturity_d = _add_days(off.offer_expiry_date, -15)
            if switched is None:
                status = "unknown"
            elif is_td:
                if switched:
                    status = "declined" if rng.random() < 0.8 else "expired"
                else:
                    status = "accepted" if rng.random() < 0.85 else "expired"
            else:
                if switched:
                    status = (
                        "accepted" if off.offered_rate_pct == best_comp.get(ep, 9e9) else "declined"
                    )
                else:
                    status = "declined" if rng.random() < 0.7 else "expired"
            if status == "accepted":
                sed = _add_days(maturity_d, -5)
            elif status == "declined":
                sed = _add_days(maturity_d, -10)
            elif status == "expired":
                sed = off.offer_expiry_date
            else:  # unknown
                sed = off.status_effective_date
            soa = _add_days(sed, int(rng.integers(1, 8)))
            rao = off.record_asof
            if soa > rao:
                rao = soa
            finalized.append(
                off.model_copy(
                    update={
                        "offer_status": status,
                        "status_effective_date": sed,
                        "status_observed_at": soa,
                        "record_asof": rao,
                    }
                )
            )
        return finalized

    def generate_offers(
        self,
        history: list[MortgageHistoryRecord] | None = None,
        outcomes: list[RenewalOutcomeRecord] | None = None,
    ) -> list[RenewalOffer]:
        """Generate the renewal-offers table (1 TD + 1-2 competitor offers/episode).

        Offer *rates* are generated here; *statuses* are assigned from
        ``outcomes`` when provided (otherwise "unknown").

        Args:
            history: v2 history (generated if None).
            outcomes: Optional outcomes for status assignment.

        Returns:
            List of validated RenewalOffer.

        """
        if history is None:
            history = self.generate_history()
        rng = self.rng
        month_dates = self._month_dates()

        # Last record per mortgage -> maturity + repriced (TD offer) rate.
        last: dict[str, MortgageHistoryRecord] = {}
        for r in history:
            last[r.mortgage_id] = r

        offers: list[RenewalOffer] = []
        for mid, rec in last.items():
            # Only in-window maturities (v2's 12/24-month terms can mature
            # before the window starts; those have no history row to link).
            if rec.maturity_date < month_dates[0] or rec.maturity_date > month_dates[-1]:
                continue
            prof = self._profile_for(mid)
            episode = rec.renewal_episode_id
            maturity = rec.maturity_date
            # TD renewal offer: market fixed rate at maturity + TD premium.
            # (Not the history rate, which is stale for early maturities.)
            td_rate = float(
                np.clip(
                    self._market_rate_at(maturity) + prof.td_renewal_premium,
                    0.5,
                    12.0,
                )
            )
            rate_type = rec.rate_type
            pay_struct = rec.payment_structure
            term = int(rec.term_months)

            # --- TD offer ---
            td_offer_date = _add_days(maturity, -TD_OFFER_LEAD_DAYS)
            market_at_offer = self._market_rate_at(td_offer_date)

            # --- Competitor offers ---
            n_comp = 2 if rng.random() < P_TWO_COMPETITORS else 1
            comp_rates: list[float] = []
            for _ in range(n_comp):
                cr = float(market_at_offer + rng.normal(0, 0.30))
                comp_rates.append(float(np.clip(cr, 0.5, 12.0)))

            offer_specs: list[tuple[str, float, date]] = [("TD", td_rate, td_offer_date)]
            for i, cr in enumerate(comp_rates):
                lead = int(rng.integers(30, 61))
                c_date = _add_days(maturity, -lead)
                offer_specs.append((f"C{i + 1}", cr, c_date))

            for tag, orate, odate in offer_specs:
                expiry_d = _add_days(maturity, 15)
                available = _add_days(odate, int(rng.integers(1, 4)))
                soa = _add_days(odate, int(rng.integers(1, 8)))
                rao = _add_months(maturity, OUTCOME_LAG_MONTHS)
                if soa > rao:
                    rao = soa
                offers.append(
                    RenewalOffer(
                        offer_id=f"O{mid[1:]}-{tag}",
                        customer_id=rec.customer_id,
                        mortgage_id=mid,
                        renewal_episode_id=episode,
                        offer_date=odate,
                        offer_available_at=available,
                        offer_expiry_date=expiry_d,
                        offered_rate_pct=round(orate, 3),
                        offered_rate_type=rate_type,
                        offered_payment_structure=pay_struct,
                        offered_term_months=term,  # type: ignore[arg-type]
                        offer_status="unknown",
                        status_effective_date=odate,
                        status_observed_at=soa,
                        record_asof=rao,
                    )
                )
        if outcomes is not None:
            offers = self._assign_offer_statuses(offers, outcomes)
        self._offers = offers
        return offers

    # ------------------------------------------------------------------
    # Outcomes (v2 DGP)
    # ------------------------------------------------------------------

    def _fred_macro_shifts(self) -> dict[date, float]:
        """Default macro overlay from FRED (empty if cache missing).

        Rising mortgage rates REDUCE switching (lock-in/affordability), so
        the z-scored 3m change enters with a negative sign. Winsorized to
        keep the 2022 hiking cycle from dominating the DGP.
        """
        z = load_fred_macro_z()
        if not z:
            return {}
        out: dict[date, float] = {}
        for d in self._month_dates():
            zval = max(-FRED_Z_CAP, min(FRED_Z_CAP, z.get(d, 0.0)))
            out[d] = -self.fred_beta * zval
        return out

    def offer_spread_for(self, mortgage_id: str, offers: list[RenewalOffer] | None = None) -> float:
        """TD-over-best-competitor spread (pp) for a mortgage's episode.

        Positive when TD is more expensive than the best competitor.
        Returns 0.0 when no offers exist.
        """
        offers = self._offers if offers is None else offers
        mine = [o for o in offers if o.mortgage_id == mortgage_id]
        td = [o for o in mine if o.offer_id.endswith("-TD")]
        comp = [o for o in mine if not o.offer_id.endswith("-TD")]
        if not td or not comp:
            return 0.0
        return max(0.0, td[0].offered_rate_pct - min(o.offered_rate_pct for o in comp))

    def generate_outcomes(
        self,
        history: list[MortgageHistoryRecord] | None = None,
        macro_shifts: dict[date, float] | None = None,
        offers: list[RenewalOffer] | None = None,
        use_fred_macro: bool | None = None,
    ) -> list[RenewalOutcomeRecord]:
        """Generate renewal outcomes from the v2 structural switching DGP.

        Offer rates are generated first (they feed the DGP); statuses are
        assigned after outcomes are drawn, guaranteeing offer-outcome
        consistency.

        Args:
            history: v2 history (generated if None).
            macro_shifts: Explicit overlay; if None and FRED is enabled,
                driven by the actual MORTGAGE30US series.
            offers: Pre-generated offers (generated if None).
            use_fred_macro: Override the instance default.

        Returns:
            List of validated RenewalOutcomeRecord.

        """
        if history is None:
            history = self.generate_history()
        if offers is None:
            # Rates first (statuses "unknown"); finalized below.
            offers = self.generate_offers(history)
        fred_on = self.use_fred_macro if use_fred_macro is None else use_fred_macro
        if macro_shifts is None and fred_on:
            macro_shifts = self._fred_macro_shifts()

        rng = self.rng
        first: dict[str, MortgageHistoryRecord] = {}
        last: dict[str, MortgageHistoryRecord] = {}
        for r in history:
            if r.mortgage_id not in first:
                first[r.mortgage_id] = r
            last[r.mortgage_id] = r

        # Offer lookup: episode -> (td_rate, best_competitor_rate).
        td_rate_by_ep: dict[str, float] = {}
        comp_rate_by_ep: dict[str, float] = {}
        for o in offers:
            ep = o.renewal_episode_id
            if o.offer_id.endswith("-TD"):
                td_rate_by_ep[ep] = o.offered_rate_pct
            else:
                comp_rate_by_ep[ep] = min(comp_rate_by_ep.get(ep, 9e9), o.offered_rate_pct)

        month_dates = self._month_dates()
        outcomes: list[RenewalOutcomeRecord] = []
        cohort_effects: dict[str, float] = {}
        for mid, rec in last.items():
            # Only in-window maturities (v2's 12/24-month terms can mature
            # before the window starts; those have no history row to link).
            if rec.maturity_date < month_dates[0] or rec.maturity_date > month_dates[-1]:
                continue
            orig = first[mid]
            prof = self._profile_for(mid)
            # v1 cohort + covariates.
            if (
                orig.rate_type == "fixed"
                and orig.origination_date.year == 2021
                and orig.mortgage_rate_pct <= 3.0
            ):
                cohort = "low_rate"
            elif rec.insured_flag is False and rec.credit_score_band == "720_plus":
                cohort = "osfi_exempt"
            else:
                n_kids0 = orig.children_under6 + orig.children_6to17
                cohort = (
                    "family_squeeze"
                    if (orig.household_income_cad < 75000 and n_kids0 >= 3)
                    else "control"
                )
            income_mo = rec.household_income_cad / 12
            payment_shock = max(0.0, rec.mortgage_payment_monthly_cad / max(income_mo, 1.0) - 0.32)
            rate_spread = max(0.0, 5.0 - orig.mortgage_rate_pct)
            family_squeeze = 1.0 if cohort == "family_squeeze" else 0.0
            osfi_exempt = 1.0 if cohort == "osfi_exempt" else 0.0
            # v2: offer competition.
            ep = rec.renewal_episode_id
            td_r = td_rate_by_ep.get(ep)
            comp_r = comp_rate_by_ep.get(ep)
            offer_spread = (
                max(0.0, td_r - comp_r) if td_r is not None and comp_r is not None else 0.0
            )
            # v2: heterogeneity.
            mat = rec.maturity_date
            cutoff = _add_months(mat, -6)
            life_event_6m = 1.0 if any(cutoff < d <= mat for d, _ in prof.life_events) else 0.0
            is_digital = 1.0 if prof.channel == "digital" else 0.0
            has_advisor = 1.0 if prof.has_advisor else 0.0
            if cohort not in cohort_effects:
                cohort_effects[cohort] = float(rng.normal(0, COHORT_SD))
            macro_shift = 0.0
            if macro_shifts:
                mkey = date(mat.year, mat.month, 1)
                shift = macro_shifts.get(mkey)
                if shift is None:
                    shift = macro_shifts.get(_month_end(mat.year, mat.month), 0.0)
                macro_shift = float(shift)
            logit_p = v2_switching_logit(
                payment_shock=payment_shock,
                offer_spread=offer_spread,
                rate_spread=rate_spread,
                family_squeeze=family_squeeze,
                osfi_exempt=osfi_exempt,
                util_drift_ppm=prof.util_drift_ppm,
                life_event_6m=life_event_6m,
                is_digital=is_digital,
                has_advisor=has_advisor,
                n_prior_refinances=prof.n_prior_refinances,
                age_months=int(rec.term_months),
                cohort_effect=cohort_effects[cohort],
                macro_shift=macro_shift,
            )
            p_switch = float(1 / (1 + np.exp(-logit_p)))
            switched = bool(rng.random() < p_switch)
            event_date = mat
            observed_at = _add_months(
                date(event_date.year, event_date.month, min(event_date.day, 28)),
                OUTCOME_LAG_MONTHS,
            )
            outcome = "switched_lender" if switched else "renewed_td"
            outcomes.append(
                RenewalOutcomeRecord(
                    customer_id=rec.customer_id,
                    mortgage_id=mid,
                    renewal_episode_id=ep,
                    scheduled_maturity_date=mat,
                    outcome=outcome,  # type: ignore[arg-type]
                    switched_lender_flag=switched,
                    event_date=event_date,
                    outcome_observed_at=observed_at,
                    record_asof=observed_at,
                )
            )

        # Finalize offer statuses for consistency with outcomes, then store.
        # (Rates are NOT regenerated: _assign_offer_statuses only sets statuses.)
        self._offers = self._assign_offer_statuses(offers, outcomes)
        return outcomes

    @property
    def offers(self) -> list[RenewalOffer]:
        """Renewal offers from the last generate_offers/generate_outcomes call."""
        return list(self._offers)
