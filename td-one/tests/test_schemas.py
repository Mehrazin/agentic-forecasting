"""Tests for TD-One Pydantic schemas (from Excel spec)."""

from datetime import date

import pytest
from pydantic import ValidationError

from td_one.schemas import (
    MortgageHistoryRecord,
    RenewalOutcomeRecord,
)


def _history_kwargs(**overrides):  # type: ignore[no-untyped-def]
    """Minimal valid history kwargs (helper, not a test)."""
    kw = {
        "customer_id": "C001",
        "household_id": "H001",
        "mortgage_id": "M001",
        "renewal_episode_id": "E001",
        "snapshot_date": date(2021, 4, 30),
        "province": "ON",
        "primary_borrower_age": 35,
        "household_size": 4,
        "children_under6": 1,
        "children_6to17": 1,
        "household_income_cad": 95000.0,
        "borrower_income_cad": 60000.0,
        "income_reference_year": 2020,
        "td_tenure_months": 24,
        "origination_date": date(2019, 5, 15),
        "term_start_date": date(2021, 5, 1),
        "maturity_date": date(2026, 5, 1),
        "term_months": 60,
        "mortgage_balance_cad": 420000.0,
        "mortgage_rate_pct": 3.0,
        "rate_type": "fixed",
        "payment_structure": "fixed_payment",
        "remaining_amortization_months": 264,
        "mortgage_payment_monthly_cad": 2171.20,
        "td_liquid_deposits_cad": 8000.0,
        "td_deposits_3m_change_cad": 500.0,
        "has_chequing": True,
        "has_savings": True,
        "has_credit_card": True,
        "has_investments": False,
        "credit_score_band": "720_plus",
        "credit_score_asof": date(2021, 4, 15),
        "missed_payment_count_12m": 0,
        "max_days_past_due_12m": 0,
        "marketing_contacts_90d": 2,
        "campaign_member_90d": False,
        "marketing_channel_last": "email",
    }
    kw.update(overrides)
    return kw


def test_mortgage_history_optional_fields_default() -> None:
    """The 9 optional Excel fields default sensibly and accept values."""
    rec = MortgageHistoryRecord(**_history_kwargs())
    assert rec.insured_flag is None
    assert rec.property_value_cad is None
    rec2 = MortgageHistoryRecord(
        **_history_kwargs(insured_flag=True, property_value_cad=650000.0)
    )
    assert rec2.insured_flag is True
    assert rec2.property_value_cad == 650000.0


def test_mortgage_history_invalid_term() -> None:
    """Term months must be in allowed set."""
    with pytest.raises(ValidationError):
        MortgageHistoryRecord(**_history_kwargs(term_months=13))  # type: ignore[arg-type]


def _outcome_kwargs(**overrides):  # type: ignore[no-untyped-def]
    """Minimal valid outcome kwargs (helper, not a test)."""
    kw = {
        "customer_id": "C001",
        "mortgage_id": "M001",
        "renewal_episode_id": "E001",
        "scheduled_maturity_date": date(2026, 10, 31),
        "outcome": "pending",
        "switched_lender_flag": None,
        "event_date": None,
        "outcome_observed_at": None,
        "record_asof": date(2026, 10, 15),
    }
    kw.update(overrides)
    return kw


def test_renewal_outcome_resolved_requires_flag() -> None:
    """Cross-field invariant: resolved outcome + None flag is rejected."""
    with pytest.raises(ValidationError, match="switched_lender_flag"):
        RenewalOutcomeRecord(**_outcome_kwargs(outcome="switched_lender"))  # type: ignore[arg-type]


def test_renewal_outcome_pending_rejects_flag() -> None:
    """Cross-field invariant: pending outcome + non-None flag is rejected."""
    with pytest.raises(ValidationError, match="switched_lender_flag"):
        RenewalOutcomeRecord(
            **_outcome_kwargs(outcome="pending", switched_lender_flag=True)  # type: ignore[arg-type]
        )


def test_renewal_outcome_resolved_valid() -> None:
    """Resolved outcome with a flag validates."""
    rec = RenewalOutcomeRecord(
        **_outcome_kwargs(
            outcome="switched_lender",
            switched_lender_flag=True,
            event_date=date(2026, 10, 31),
            outcome_observed_at=date(2026, 12, 31),
        )  # type: ignore[arg-type]
    )
    assert rec.switched_lender_flag is True
