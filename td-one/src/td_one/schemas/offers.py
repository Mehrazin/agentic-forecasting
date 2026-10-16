"""Pydantic schemas for renewal offers (Excel spec, MVP2).

Source: docs/Mortgage_Base_Data_and_Resources_2_nar3.xlsx
- Renewal Offers Schema: 15 fields. Key: offer_id.
  Links to the same mortgage and renewal episode as the history records.

Point-in-time: offer_available_at and status_observed_at are the PIT guards —
query with record_asof <= run_date and never backfill a status into earlier
snapshots.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RenewalOffer(BaseModel):
    """A single renewal offer issued to a borrower (15 fields).

    Key: offer_id. Each revised offer receives a new offer_id.
    """

    offer_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    mortgage_id: str = Field(min_length=1)
    renewal_episode_id: str = Field(min_length=1)
    offer_date: date
    offer_available_at: date
    offer_expiry_date: date
    offered_rate_pct: float = Field(ge=0.5, le=12.0)
    offered_rate_type: Literal["fixed", "variable"]
    offered_payment_structure: Literal["fixed_payment", "adjustable_payment"]
    offered_term_months: Literal[12, 24, 36, 48, 60, 84, 120]
    offer_status: Literal[
        "open",
        "accepted",
        "declined",
        "expired",
        "superseded",
        "withdrawn",
        "unknown",
    ]
    status_effective_date: date
    status_observed_at: date
    record_asof: date

    @model_validator(mode="after")
    def _temporal_ordering(self) -> "RenewalOffer":
        """Enforce the Excel spec's date ordering constraints."""
        if self.offer_available_at < self.offer_date:
            raise ValueError("offer_available_at must be on or after offer_date")
        if self.offer_expiry_date < self.offer_date:
            raise ValueError("offer_expiry_date must be on or after offer_date")
        if not (self.offer_date <= self.status_effective_date <= self.record_asof):
            raise ValueError("status_effective_date must satisfy offer_date <= date <= record_asof")
        if not (self.status_effective_date <= self.status_observed_at <= self.record_asof):
            raise ValueError(
                "status_observed_at must satisfy status_effective_date <= date <= record_asof"
            )
        return self
