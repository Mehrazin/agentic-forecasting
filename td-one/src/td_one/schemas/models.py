"""Pydantic schemas for TD-One mortgage data (from Excel spec).

Source: docs/Mortgage_Base_Data_and_Resources_2_nar3.xlsx
- Mortgage History Schema: 46 fields (37 core + 9 optional)
- Renewal Outcomes Schema: 9 fields
- Renewal Offers Schema: 15 fields (MVP2)
- Propensity Scores Schema: 7 fields (MVP2)
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MortgageHistoryRecord(BaseModel):
    """Single monthly snapshot of a mortgage (37 core + 9 optional fields).

    Key: (mortgage_id, snapshot_date).
    """

    customer_id: str = Field(min_length=1)
    household_id: str = Field(min_length=1)
    mortgage_id: str = Field(min_length=1)
    renewal_episode_id: str = Field(min_length=1)
    snapshot_date: date
    province: str = Field(min_length=2, max_length=2)
    primary_borrower_age: int = Field(ge=18, le=90)
    household_size: int = Field(ge=1, le=10)
    children_under6: int = Field(ge=0, le=6)
    children_6to17: int = Field(ge=0, le=6)
    household_income_cad: float = Field(ge=0, le=500000)
    borrower_income_cad: float = Field(ge=0)
    income_reference_year: int = Field(ge=2000, le=2030)
    td_tenure_months: int = Field(ge=0, le=600)
    origination_date: date
    term_start_date: date
    maturity_date: date
    term_months: Literal[12, 24, 36, 48, 60, 84, 120]
    mortgage_balance_cad: float = Field(gt=0)
    mortgage_rate_pct: float = Field(ge=0.5, le=12.0)
    rate_type: Literal["fixed", "variable"]
    payment_structure: Literal["fixed_payment", "adjustable_payment"]
    remaining_amortization_months: int = Field(ge=1, le=360)
    mortgage_payment_monthly_cad: float = Field(gt=0)
    td_liquid_deposits_cad: float = Field(ge=0, le=500000)
    td_deposits_3m_change_cad: float
    has_chequing: bool
    has_savings: bool
    has_credit_card: bool
    has_investments: bool
    credit_score_band: Literal["below_600", "600_659", "660_719", "720_plus", "unknown"]
    credit_score_asof: date
    missed_payment_count_12m: int = Field(ge=0, le=12)
    max_days_past_due_12m: int = Field(ge=0, le=365)
    marketing_contacts_90d: int = Field(ge=0, le=30)
    campaign_member_90d: bool
    marketing_channel_last: str
    # --- 9 optional fields (Excel spec; load-bearing for attrition) ---
    employment_status: Literal["employed", "self_employed", "retired", "unemployed", "unknown"] = (
        "unknown"
    )
    adjusted_family_net_income_cad: float | None = Field(default=None, ge=0)
    government_benefit_income_monthly_cad: float | None = Field(default=None, ge=0)
    property_value_cad: float | None = Field(default=None, gt=0)
    property_value_asof: date | None = None
    insured_flag: bool | None = None  # OSFI stress-test exemption driver
    occupancy_type: Literal["owner_occupied", "rental", "second_home", "unknown"] = "unknown"
    other_debt_payment_monthly_cad: float | None = Field(default=None, ge=0)
    credit_utilization_pct: float | None = Field(default=None, ge=0, le=100)


class RenewalOutcomeRecord(BaseModel):
    """Renewal outcome for a mortgage episode (9 fields).

    Key: (mortgage_id, renewal_episode_id, record_asof).
    Point-in-time: query with record_asof <= run_date.
    """

    customer_id: str = Field(min_length=1)
    mortgage_id: str = Field(min_length=1)
    renewal_episode_id: str = Field(min_length=1)
    scheduled_maturity_date: date
    outcome: Literal[
        "renewed_td",
        "switched_lender",
        "paid_off",
        "property_sale",
        "default_exit",
        "lender_declined",
        "other_exit",
        "pending",
        "unknown",
    ]
    switched_lender_flag: bool | None = Field(
        default=None,
        description="1/0 for resolved, None (NaN) for pending/unknown",
    )
    event_date: date | None = None
    outcome_observed_at: date | None = None
    record_asof: date

    @model_validator(mode="after")
    def _resolved_requires_flag(self) -> "RenewalOutcomeRecord":
        """Enforce: resolved outcomes carry a non-null switched_lender_flag."""
        resolved = self.outcome not in ("pending", "unknown")
        if resolved and self.switched_lender_flag is None:
            raise ValueError(
                f"outcome={self.outcome!r} is resolved but switched_lender_flag is None"
            )
        if not resolved and self.switched_lender_flag is not None:
            raise ValueError(
                f"outcome={self.outcome!r} is unresolved but switched_lender_flag is not None"
            )
        return self
