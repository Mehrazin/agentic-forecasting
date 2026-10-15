"""Synthetic mortgage data generator with planted regime shocks.

Generates 10k customers × 66 months of mortgage history matching the Excel
schema. Plants three verifiable shocks (plan §3.2):
1. Rate shock: low-rate 2021 borrowers face payment shock in 2026 (~25% switch)
2. OSFI shock: uninsured high-credit borrowers exempt from MQR stress test
3. Family squeeze: low-income families with 3+ kids (~15% switch)

Uses Canadian semi-annual compounding for all mortgage math.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from td_one.data_gen.amortization import (
    canadian_balance_after_payment,
    canadian_monthly_payment,
)
from td_one.schemas import MortgageHistoryRecord


class MortgageSimulator:
    """Generates synthetic mortgage history with structural DGP."""

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
        self.rng = np.random.default_rng(seed)

    def _month_dates(self) -> list[date]:
        """Generate month-end dates."""
        dates = []
        y, m = self.start_date.year, self.start_date.month
        for _ in range(self.n_months):
            # Last day of month
            if m == 12:
                next_y, next_m = y + 1, 1
            else:
                next_y, next_m = y, m + 1
            # Approximate month-end (use 28th to avoid month-length issues)
            dates.append(date(y, m, 28))
            y, m = next_y, next_m
        return dates

    def _generate_customers(self) -> pd.DataFrame:
        """Generate base customer demographics (static)."""
        n = self.n_customers
        df = pd.DataFrame(
            {
                "customer_id": [f"C{i:05d}" for i in range(n)],
                "household_id": [f"H{i:05d}" for i in range(n)],
                "mortgage_id": [f"M{i:05d}" for i in range(n)],
                "renewal_episode_id": [f"E{i:05d}-01" for i in range(n)],
                "province": self.rng.choice(
                    ["ON", "BC", "AB", "QC", "MB"],
                    size=n,
                    p=[0.4, 0.2, 0.15, 0.15, 0.1],
                ),
                "primary_borrower_age": self.rng.integers(25, 65, size=n),
                "household_size": self.rng.integers(1, 6, size=n),
                "household_income_cad": self.rng.uniform(50000, 200000, size=n),
                "td_tenure_months": self.rng.integers(0, 120, size=n),
                # Mortgage terms: 2021 originations get low rates (for shock)
                "origination_year": self.rng.choice(
                    [2019, 2020, 2021, 2022], size=n, p=[0.2, 0.2, 0.4, 0.2]
                ),
            }
        )
        # Children (for family shock)
        df["children_under6"] = self.rng.integers(0, 3, size=n)
        df["children_6to17"] = self.rng.integers(0, 3, size=n)
        # Rate: 2021 originations get <=3.0% (planted for rate shock)
        df["mortgage_rate_pct"] = np.where(
            df["origination_year"] == 2021,
            self.rng.uniform(2.0, 3.0, size=n),  # Low-rate cohort
            self.rng.uniform(3.5, 5.5, size=n),
        )
        df["rate_type"] = "fixed"
        df["term_months"] = 60  # 5-year fixed (matures in 2026 for 2021 cohort)
        # Insured flag (for OSFI shock): 70% insured
        df["insured_flag"] = self.rng.random(size=n) < 0.7
        # Credit score (for OSFI shock: need 720_plus)
        df["credit_score_band"] = self.rng.choice(
            ["below_600", "600_659", "660_719", "720_plus"],
            size=n,
            p=[0.05, 0.15, 0.35, 0.45],
        )
        # Deposits (for rate shock: <5000)
        df["td_liquid_deposits_cad"] = self.rng.uniform(1000, 50000, size=n)
        return df

    def generate_history(self) -> list[MortgageHistoryRecord]:
        """Generate monthly mortgage history records.

        Returns:
            List of validated MortgageHistoryRecord (n_customers × n_months).
        """
        customers = self._generate_customers()
        dates = self._month_dates()
        records: list[MortgageHistoryRecord] = []

        for _, cust in customers.iterrows():
            # Initial mortgage setup
            balance = float(self.rng.uniform(300000, 600000))
            rate = float(cust["mortgage_rate_pct"])
            amort_months = 300  # 25-year amortization
            payment = canadian_monthly_payment(balance, rate, amort_months)

            # Maturity: 5-year term from origination
            orig_year = int(cust["origination_year"])
            maturity = date(orig_year + 5, 6, 30)

            for snap_date in dates:
                # Update balance (simplified: one payment per month)
                balance = canadian_balance_after_payment(balance, rate, payment)

                # Borrower income (<= household)
                borrower_income = float(cust["household_income_cad"]) * 0.6

                rec = MortgageHistoryRecord(
                    customer_id=str(cust["customer_id"]),
                    household_id=str(cust["household_id"]),
                    mortgage_id=str(cust["mortgage_id"]),
                    renewal_episode_id=str(cust["renewal_episode_id"]),
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
                    origination_date=date(orig_year, 6, 15),
                    term_start_date=date(orig_year, 6, 15),
                    maturity_date=maturity,
                    term_months=60,
                    mortgage_balance_cad=round(balance, 2),
                    mortgage_rate_pct=rate,
                    rate_type="fixed",
                    payment_structure="fixed_payment",
                    remaining_amortization_months=max(1, amort_months - 12),
                    mortgage_payment_monthly_cad=round(payment, 2),
                    td_liquid_deposits_cad=float(cust["td_liquid_deposits_cad"]),
                    td_deposits_3m_change_cad=float(
                        self.rng.uniform(-1000, 1000)
                    ),
                    has_chequing=True,
                    has_savings=bool(self.rng.random() < 0.8),
                    has_credit_card=bool(self.rng.random() < 0.9),
                    has_investments=bool(self.rng.random() < 0.4),
                    credit_score_band=str(cust["credit_score_band"]),  # type: ignore[arg-type]
                    credit_score_asof=snap_date,
                    missed_payment_count_12m=int(
                        self.rng.integers(0, 2)
                    ),
                    max_days_past_due_12m=0,
                    marketing_contacts_90d=int(self.rng.integers(0, 5)),
                    campaign_member_90d=False,
                    marketing_channel_last="email",
                )
                records.append(rec)

        return records

    def to_parquet(self, path: str) -> None:
        """Export history to Parquet."""
        records = self.generate_history()
        df = pd.DataFrame([r.model_dump() for r in records])
        df.to_parquet(path, index=False)
