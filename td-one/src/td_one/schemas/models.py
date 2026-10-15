"""Pydantic schemas for TD-One mortgage data (from Excel spec).

Source: docs/Mortgage_Base_Data_and_Resources_2_nar3.xlsx
- Mortgage History Schema: 46 fields (37 core + 9 optional)
- Renewal Outcomes Schema: 9 fields
- Renewal Offers Schema: 15 fields (MVP2)
- Propensity Scores Schema: 7 fields (MVP2)
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class MortgageHistoryRecord(BaseModel):
    """Single monthly snapshot of a mortgage (37 core fields).

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
    credit_score_band: Literal[
        "below_600", "600_659", "660_719", "720_plus", "unknown"
    ]
    credit_score_asof: date
    missed_payment_count_12m: int = Field(ge=0, le=12)
    max_days_past_due_12m: int = Field(ge=0, le=365)
    marketing_contacts_90d: int = Field(ge=0, le=30)
    campaign_member_90d: bool
    marketing_channel_last: str


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
