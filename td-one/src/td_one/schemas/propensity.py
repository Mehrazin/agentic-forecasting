"""Pydantic schemas for propensity scores (Excel spec, MVP2).

Source: docs/Mortgage_Base_Data_and_Resources_2_nar3.xlsx
- Propensity Scores Schema: 7 fields.
  Key: (customer_id, score_name, score_date, model_version).

Scores are illustrative/fictional (not calibrated); they describe named
account-opening events, not mortgage attrition.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PropensityScore(BaseModel):
    """Illustrative propensity score for a named account-opening event.

    Key: (customer_id, score_name, score_date, model_version).
    """

    customer_id: str = Field(min_length=1)
    score_date: date
    score_available_at: date
    score_name: Literal[
        "open_td_investment_account",
        "open_td_savings_account",
        "open_td_chequing_account",
    ]
    prediction_horizon_months: int = Field(gt=0)
    model_version: str = Field(min_length=1)
    score_value: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _available_after_scoring(self) -> "PropensityScore":
        """Enforce: score_available_at is on or after score_date."""
        if self.score_available_at < self.score_date:
            raise ValueError("score_available_at must be on or after score_date")
        return self
