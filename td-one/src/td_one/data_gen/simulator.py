"""Synthetic mortgage data generator with structural outcome DGP.

Generates customers × months of mortgage history matching the Excel schema
(46 fields), plus renewal outcomes from a structural switching DGP.

Planted shocks (measurable in outcomes, not decorative):
1. Rate shock: low-rate 2021 borrowers face payment shock at 2026 maturity
   (~25% switch vs ~12% control).
2. OSFI shock: uninsured 720+ borrowers exempt from the stress test at
   renewal (~+8pp switch uplift).
3. Family squeeze: low-income families with 3+ kids (~15% switch).

Realism features:
- Lognormal household income (not uniform), sticky delinquency (Markov),
  autocorrelated deposit changes linked to payment shock.
- ~30% variable-rate share with rates tied to a simulated BoC policy path.
- Term heterogeneity {36, 60, 84} months; maturities spread across months.
- Amortization decrements monthly; episode "-02" emitted at maturity.
- Switching DGP: logit(p) = β0 + β1·payment_shock + β2·rate_spread
  + β3·family_squeeze + β4·osfi_exempt + cohort random effect.
- Outcomes observed with a settlement lag (outcome_observed_at).

Uses Canadian semi-annual compounding for all mortgage math.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from td_one.data_gen.amortization import (
    canadian_balance_after_payment,
    canadian_monthly_payment,
)
from td_one.schemas import MortgageHistoryRecord, RenewalOutcomeRecord

# --- DGP parameters (documented, tunable) ---

#: Variable-rate share (Canada ~30%).
VARIABLE_SHARE = 0.30
#: Term distribution in months.
TERM_DIST = {36: 0.25, 60: 0.65, 84: 0.10}
#: Monthly delinquency transition probs: P(miss | clean), P(miss | missed).
DELINQ_P01 = 0.003
DELINQ_P11 = 0.35
#: Deposit AR(1) persistence.
DEPOSIT_RHO = 0.7
#: Settlement lag for outcomes (months).
OUTCOME_LAG_MONTHS = 2

#: Structural switching DGP coefficients (logit scale).
BETA_0 = -2.2  # baseline ~10% switch
BETA_PAYMENT_SHOCK = 1.8  # per unit payment-shock ratio
BETA_RATE_SPREAD = 0.9  # per pp of market-vs-current spread
BETA_FAMILY_SQUEEZE = 0.7
BETA_OSFI_EXEMPT = 0.6
COHORT_SD = 0.25  # cohort random effect


def _boc_policy_path(n_months: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate the BoC policy rate path (drives variable rates).

    Starts at 0.25% (2021), rises to ~5% by 2023, eases to ~3% by 2026,
    plus small noise. Shape: (n_months,).
    """
    t = np.arange(n_months, dtype=float)
    # Piecewise: hike 2022-2023, hold, ease 2024-2026
    path = np.piecewise(
        t,
        [t < 12, (t >= 12) & (t < 24), t >= 24],
        [
            lambda t: 0.25 + 0.02 * t,  # 2021-2022 gentle
            lambda t: 0.5 + 0.19 * (t - 12),  # hiking cycle
            lambda t: 5.0 - 0.06 * (t - 24),  # easing
        ],
    )
    path += rng.normal(0, 0.05, size=n_months)
    return np.clip(path, 0.1, 6.0)


class MortgageSimulator:
    """Generates synthetic mortgage history + renewal outcomes."""

    def __init__(
        self,
        n_customers: int = 10000,
        n_months: int = 66,
        start_date: date = date(2021, 4, 30),
        seed: int = 42,
    ) -> None:
        """Initialize the simulator.

        Args:
            n_customers: Number of active mortgage customers.
            n_months: Number of monthly snapshots (60 train + 6 OOT).
            start_date: First snapshot date (month-end).
            seed: Random seed for reproducibility.

        """
        self.n_customers = n_customers
        self.n_months = n_months
        self.start_date = start_date
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._boc_path = _boc_policy_path(n_months, self.rng)

    def _month_dates(self) -> list[date]:
        """Generate true month-end dates."""
        dates = []
        y, m = self.start_date.year, self.start_date.month
        for _ in range(self.n_months):
            # Last day of month
            if m == 12:
                last_day = 31
                next_y, next_m = y + 1, 1
            elif m in (4, 6, 9, 11):
                last_day = 30
                next_y, next_m = y, m + 1
            elif m == 2:
                last_day = 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28
                next_y, next_m = y, m + 1
            else:
                last_day = 31
                next_y, next_m = y, m + 1
            dates.append(date(y, m, last_day))
            y, m = next_y, next_m
        return dates

    def _generate_customers(self) -> pd.DataFrame:
        """Generate base customer demographics with joint structure."""
        n = self.n_customers
        rng = self.rng
        df = pd.DataFrame(
            {
                "customer_id": [f"C{i:05d}" for i in range(n)],
                "household_id": [f"H{i:05d}" for i in range(n)],
                "mortgage_id": [f"M{i:05d}" for i in range(n)],
                "province": rng.choice(
                    ["ON", "BC", "AB", "QC", "MB"],
                    size=n,
                    p=[0.4, 0.2, 0.15, 0.15, 0.1],
                ),
                "primary_borrower_age": rng.integers(25, 65, size=n),
                "household_size": rng.integers(1, 6, size=n),
                # Lognormal income (median ~95k, realistic right skew)
                "household_income_cad": np.clip(
                    rng.lognormal(mean=11.45, sigma=0.55, size=n), 20000, 500000
                ),
                "td_tenure_months": rng.integers(0, 120, size=n),
                "origination_year": rng.choice(
                    [2019, 2020, 2021, 2022], size=n, p=[0.2, 0.2, 0.4, 0.2]
                ),
            }
        )
        df["children_under6"] = rng.integers(0, 3, size=n)
        df["children_6to17"] = rng.integers(0, 3, size=n)
        # Employment (correlated with income lightly via age)
        df["employment_status"] = rng.choice(
            ["employed", "self_employed", "retired", "unemployed"],
            size=n,
            p=[0.72, 0.12, 0.10, 0.06],
        )
        # Rate type: ~30% variable
        df["rate_type"] = np.where(
            rng.random(size=n) < VARIABLE_SHARE, "variable", "fixed"
        )
        # Fixed rates: 2021 originations get <=3.0% (rate-shock cohort)
        is_low = df["origination_year"] == 2021
        fixed_rate = np.where(
            is_low,
            rng.uniform(2.0, 3.0, size=n),
            rng.uniform(3.5, 5.5, size=n),
        )
        # Variable rates: BoC + ~1.5% spread at origination
        df["mortgage_rate_pct"] = np.where(
            df["rate_type"] == "variable",
            1.75 + rng.normal(0, 0.2, size=n),  # BoC 0.25 + spread
            fixed_rate,
        )
        # Term heterogeneity
        terms = np.array(list(TERM_DIST.keys()))
        term_p = np.array(list(TERM_DIST.values()))
        df["term_months"] = rng.choice(terms, size=n, p=term_p)
        # Maturity spread across the maturity month
        df["maturity_day"] = rng.integers(1, 29, size=n)
        # Originations happen year-round (realistic renewal calendar);
        # previously hardcoded to June, which collapsed all maturities
        # into a handful of months.
        df["origination_month"] = rng.integers(1, 13, size=n)
        # Insured flag (OSFI shock): 70% insured
        df["insured_flag"] = rng.random(size=n) < 0.7
        df["credit_score_band"] = rng.choice(
            ["below_600", "600_659", "660_719", "720_plus"],
            size=n,
            p=[0.05, 0.15, 0.35, 0.45],
        )
        # Property value → LTV structure
        df["property_value_cad"] = np.clip(
            rng.lognormal(mean=13.2, sigma=0.4, size=n), 150000, 2500000
        )
        # Credit utilization correlated with income (lower income → higher util)
        income_z = (np.log(df["household_income_cad"]) - 11.45) / 0.55
        df["credit_utilization_pct"] = np.clip(
            35 - 10 * income_z + rng.normal(0, 8, size=n), 1, 95
        )
        df["other_debt_payment_monthly_cad"] = np.clip(
            rng.lognormal(mean=5.5, sigma=0.8, size=n), 0, 3000
        )
        df["td_liquid_deposits_cad"] = np.clip(
            rng.lognormal(mean=8.5, sigma=1.0, size=n), 500, 200000
        )
        df["occupancy_type"] = rng.choice(
            ["owner_occupied", "rental", "second_home"],
            size=n,
            p=[0.85, 0.10, 0.05],
        )
        return df

    def _cohort(self, row: pd.Series) -> str:
        """Assign planted-shock cohort label."""
        n_kids = int(row["children_under6"]) + int(row["children_6to17"])
        if row["origination_year"] == 2021 and row["mortgage_rate_pct"] <= 3.0:
            return "low_rate"
        if not bool(row["insured_flag"]) and row["credit_score_band"] == "720_plus":
            return "osfi_exempt"
        if float(row["household_income_cad"]) < 75000 and n_kids >= 3:
            return "family_squeeze"
        return "control"

    def generate_history(self) -> list[MortgageHistoryRecord]:
        """Generate monthly mortgage history records.

        Returns:
            List of validated MortgageHistoryRecord (n_customers × n_months).

        """
        customers = self._generate_customers()
        dates = self._month_dates()
        records: list[MortgageHistoryRecord] = []
        rng = self.rng

        for _, cust in customers.iterrows():
            balance = float(rng.uniform(300000, 600000))
            rate = float(cust["mortgage_rate_pct"])
            is_variable = cust["rate_type"] == "variable"
            amort_months = 300
            payment = canadian_monthly_payment(balance, rate, amort_months)

            orig_year = int(cust["origination_year"])
            orig_month = int(cust["origination_month"])
            term_months = int(cust["term_months"])
            # Maturity: term months after origination (calendar-correct).
            total = (orig_month - 1) + term_months
            mat_year = orig_year + total // 12
            mat_month = total % 12 + 1
            maturity = date(mat_year, mat_month, min(int(cust["maturity_day"]), 28))
            orig_date = date(orig_year, orig_month, 15)

            # Persistent per-customer state
            delinquent = False
            deposit_change = 0.0
            deposits = float(cust["td_liquid_deposits_cad"])
            episode = 1

            for t, snap_date in enumerate(dates):
                # Variable rate follows BoC path
                if is_variable:
                    rate = float(self._boc_path[t] + 1.5 + rng.normal(0, 0.05))
                    rate = float(np.clip(rate, 0.5, 12.0))
                    payment = canadian_monthly_payment(balance, rate, amort_months)
                balance = canadian_balance_after_payment(balance, rate, payment)
                amort_months = max(1, amort_months - 1)

                # Sticky delinquency (Markov)
                p_miss = DELINQ_P11 if delinquent else DELINQ_P01
                delinquent = bool(rng.random() < p_miss)
                missed_12m = 1 if delinquent else 0

                # Autocorrelated deposits, drained by payment shock
                payment_ratio = payment / (float(cust["household_income_cad"]) / 12)
                shock_drag = -500 * max(0.0, payment_ratio - 0.32)
                deposit_change = (
                    DEPOSIT_RHO * deposit_change
                    + rng.normal(0, 300)
                    + shock_drag
                )
                deposits = max(0.0, deposits + deposit_change)

                # New episode at maturity (renewal repricing for fixed)
                ep_id = f"E{str(cust['mortgage_id'])[1:]}-{episode:02d}"
                if snap_date >= maturity and episode == 1:
                    episode = 2
                    ep_id = f"E{str(cust['mortgage_id'])[1:]}-02"
                    if not is_variable:
                        # Renew into prevailing rates (~5% in 2026)
                        rate = float(rng.uniform(4.5, 5.5))
                        payment = canadian_monthly_payment(
                            balance, rate, amort_months
                        )

                borrower_income = float(cust["household_income_cad"]) * 0.6
                prop_value = float(cust["property_value_cad"])

                rec = MortgageHistoryRecord(
                    customer_id=str(cust["customer_id"]),
                    household_id=str(cust["household_id"]),
                    mortgage_id=str(cust["mortgage_id"]),
                    renewal_episode_id=ep_id,
                    snapshot_date=snap_date,
                    province=str(cust["province"]),
                    primary_borrower_age=int(cust["primary_borrower_age"]),
                    household_size=int(cust["household_size"]),
                    children_under6=int(cust["children_under6"]),
                    children_6to17=int(cust["children_6to17"]),
                    household_income_cad=float(cust["household_income_cad"]),
                    borrower_income_cad=borrower_income,
                    income_reference_year=snap_date.year - 1,
                    td_tenure_months=int(cust["td_tenure_months"]),
                    origination_date=orig_date,
                    term_start_date=orig_date,
                    maturity_date=maturity,
                    term_months=term_months,  # type: ignore[arg-type]
                    mortgage_balance_cad=round(balance, 2),
                    mortgage_rate_pct=round(rate, 3),
                    rate_type=str(cust["rate_type"]),  # type: ignore[arg-type]
                    payment_structure=(
                        "adjustable_payment" if is_variable else "fixed_payment"
                    ),
                    remaining_amortization_months=amort_months,
                    mortgage_payment_monthly_cad=round(payment, 2),
                    td_liquid_deposits_cad=round(deposits, 2),
                    td_deposits_3m_change_cad=round(deposit_change * 3, 2),
                    has_chequing=True,
                    has_savings=bool(rng.random() < 0.8),
                    has_credit_card=bool(rng.random() < 0.9),
                    has_investments=bool(rng.random() < 0.4),
                    credit_score_band=str(cust["credit_score_band"]),  # type: ignore[arg-type]
                    credit_score_asof=snap_date,
                    missed_payment_count_12m=missed_12m,
                    max_days_past_due_12m=30 if delinquent else 0,
                    marketing_contacts_90d=int(rng.integers(0, 5)),
                    campaign_member_90d=False,
                    marketing_channel_last="email",
                    # Optional fields (threaded through, not dropped)
                    employment_status=str(cust["employment_status"]),  # type: ignore[arg-type]
                    adjusted_family_net_income_cad=round(
                        float(cust["household_income_cad"]) * 0.75, 2
                    ),
                    government_benefit_income_monthly_cad=round(
                        150.0 * (int(cust["children_under6"]) + int(cust["children_6to17"])), 2
                    ),
                    property_value_cad=round(prop_value, 2),
                    property_value_asof=snap_date,
                    insured_flag=bool(cust["insured_flag"]),
                    occupancy_type=str(cust["occupancy_type"]),  # type: ignore[arg-type]
                    other_debt_payment_monthly_cad=round(
                        float(cust["other_debt_payment_monthly_cad"]), 2
                    ),
                    credit_utilization_pct=round(
                        float(cust["credit_utilization_pct"]), 1
                    ),
                )
                records.append(rec)

        return records

    def generate_outcomes(
        self,
        history: list[MortgageHistoryRecord] | None = None,
    ) -> list[RenewalOutcomeRecord]:
        """Generate renewal outcomes from the structural switching DGP.

        For each mortgage whose maturity falls inside the snapshot window,
        draws a switching outcome from:
            logit(p) = β0 + β1·payment_shock + β2·rate_spread
                     + β3·family_squeeze + β4·osfi_exempt + ε_cohort

        Outcomes carry an observation lag (outcome_observed_at = event + 2mo)
        for PIT masking.

        Args:
            history: Pre-generated history (regenerated if None).

        Returns:
            List of validated RenewalOutcomeRecord.

        """
        if history is None:
            history = self.generate_history()
        rng = self.rng
        # First snapshot per mortgage → origination characteristics (cohort);
        # last snapshot per mortgage → covariates at maturity.
        first: dict[str, MortgageHistoryRecord] = {}
        last: dict[str, MortgageHistoryRecord] = {}
        for r in history:
            if r.mortgage_id not in first:
                first[r.mortgage_id] = r
            last[r.mortgage_id] = r  # chronological, last wins

        month_dates = self._month_dates()
        outcomes: list[RenewalOutcomeRecord] = []
        cohort_effects: dict[str, float] = {}
        for mid, rec in last.items():
            # Only maturities inside the window produce outcomes
            if rec.maturity_date > month_dates[-1]:
                continue
            orig = first[mid]
            # Cohort from ORIGINATION characteristics (not repriced rate)
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
            # Covariates at maturity (from last snapshot)
            income_mo = rec.household_income_cad / 12
            payment_shock = max(
                0.0, rec.mortgage_payment_monthly_cad / max(income_mo, 1.0) - 0.32
            )
            rate_spread = max(0.0, 5.0 - orig.mortgage_rate_pct)  # vs ORIGINATION rate
            family_squeeze = 1.0 if cohort == "family_squeeze" else 0.0
            osfi_exempt = 1.0 if cohort == "osfi_exempt" else 0.0
            if cohort not in cohort_effects:
                cohort_effects[cohort] = float(rng.normal(0, COHORT_SD))
            logit_p = (
                BETA_0
                + BETA_PAYMENT_SHOCK * payment_shock
                + BETA_RATE_SPREAD * rate_spread
                + BETA_FAMILY_SQUEEZE * family_squeeze
                + BETA_OSFI_EXEMPT * osfi_exempt
                + cohort_effects[cohort]
            )
            p_switch = float(1 / (1 + np.exp(-logit_p)))
            switched = bool(rng.random() < p_switch)
            # Event at maturity; observed after settlement lag
            event_date = rec.maturity_date
            observed_at = date(
                event_date.year + (event_date.month + OUTCOME_LAG_MONTHS - 1) // 12,
                (event_date.month + OUTCOME_LAG_MONTHS - 1) % 12 + 1,
                min(event_date.day, 28),
            )
            outcome = "switched_lender" if switched else "renewed_td"
            outcomes.append(
                RenewalOutcomeRecord(
                    customer_id=rec.customer_id,
                    mortgage_id=rec.mortgage_id,
                    renewal_episode_id=rec.renewal_episode_id,
                    scheduled_maturity_date=rec.maturity_date,
                    outcome=outcome,  # type: ignore[arg-type]
                    switched_lender_flag=switched,
                    event_date=event_date,
                    outcome_observed_at=observed_at,
                    record_asof=observed_at,
                )
            )
        return outcomes

    def cohort_of(self, rec: MortgageHistoryRecord) -> str:
        """Cohort label for a record (for uplift tests)."""
        n_kids = rec.children_under6 + rec.children_6to17
        if rec.mortgage_rate_pct <= 3.0 and rec.term_months == 60:
            return "low_rate"
        if rec.insured_flag is False and rec.credit_score_band == "720_plus":
            return "osfi_exempt"
        if rec.household_income_cad < 75000 and n_kids >= 3:
            return "family_squeeze"
        return "control"

    def to_parquet(self, path: str) -> None:
        """Export history to Parquet."""
        records = self.generate_history()
        df = pd.DataFrame([r.model_dump() for r in records])
        df.to_parquet(path, index=False)
