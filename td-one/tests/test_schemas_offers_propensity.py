"""Tests for the RenewalOffer and PropensityScore schemas (Excel spec, MVP2).

Covers the temporal ordering contracts and range/category validation —
the non-obvious logic that is easy to get wrong. Plain construction is
not tested beyond the fixtures.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from td_one.schemas import PropensityScore, RenewalOffer


def _offer(**overrides) -> dict:
    base = {
        "offer_id": "O-001",
        "customer_id": "C001",
        "mortgage_id": "M-001",
        "renewal_episode_id": "M-001-02",
        "offer_date": date(2024, 5, 1),
        "offer_available_at": date(2024, 5, 2),
        "offer_expiry_date": date(2024, 6, 30),
        "offered_rate_pct": 4.3,
        "offered_rate_type": "fixed",
        "offered_payment_structure": "fixed_payment",
        "offered_term_months": 60,
        "offer_status": "open",
        "status_effective_date": date(2024, 5, 1),
        "status_observed_at": date(2024, 5, 2),
        "record_asof": date(2024, 5, 31),
    }
    base.update(overrides)
    return base


def _score(**overrides) -> dict:
    base = {
        "customer_id": "C001",
        "score_date": date(2024, 5, 1),
        "score_available_at": date(2024, 5, 3),
        "score_name": "open_td_savings_account",
        "prediction_horizon_months": 6,
        "model_version": "v1-synthetic",
        "score_value": 0.18,
    }
    base.update(overrides)
    return base


def test_offer_valid_row() -> None:
    offer = RenewalOffer(**_offer())
    assert offer.offered_term_months == 60
    assert offer.offer_status == "open"


def test_offer_available_before_issued_rejected() -> None:
    with pytest.raises(ValidationError, match="offer_available_at"):
        RenewalOffer(**_offer(offer_available_at=date(2024, 4, 30)))


def test_offer_expiry_before_issued_rejected() -> None:
    with pytest.raises(ValidationError, match="offer_expiry_date"):
        RenewalOffer(**_offer(offer_expiry_date=date(2024, 4, 30)))


def test_offer_status_effective_window_rejected() -> None:
    # status cannot take effect before the offer exists
    with pytest.raises(ValidationError, match="status_effective_date"):
        RenewalOffer(**_offer(status_effective_date=date(2024, 4, 30)))
    # status cannot be observed before it took effect
    with pytest.raises(ValidationError, match="status_observed_at"):
        RenewalOffer(
            **_offer(
                status_effective_date=date(2024, 5, 10),
                status_observed_at=date(2024, 5, 5),
            )
        )
    # never backfill a status past the extract cutoff
    with pytest.raises(ValidationError, match="status_observed_at"):
        RenewalOffer(**_offer(status_observed_at=date(2024, 6, 15)))


def test_offer_rejects_bad_categories_and_rate() -> None:
    with pytest.raises(ValidationError):
        RenewalOffer(**_offer(offered_rate_type="hybrid"))
    with pytest.raises(ValidationError):
        RenewalOffer(**_offer(offered_term_months=18))
    with pytest.raises(ValidationError):
        RenewalOffer(**_offer(offered_rate_pct=25.0))


def test_score_valid_row() -> None:
    score = PropensityScore(**_score())
    assert score.score_value == pytest.approx(0.18)


def test_score_available_before_scoring_rejected() -> None:
    with pytest.raises(ValidationError, match="score_available_at"):
        PropensityScore(**_score(score_available_at=date(2024, 4, 30)))


def test_score_value_bounded() -> None:
    with pytest.raises(ValidationError):
        PropensityScore(**_score(score_value=1.5))
    with pytest.raises(ValidationError):
        PropensityScore(**_score(score_value=-0.1))
    # boundaries are inclusive per spec ("0-1 inclusive")
    PropensityScore(**_score(score_value=0.0))
    PropensityScore(**_score(score_value=1.0))


def test_score_rejects_unknown_event_name() -> None:
    with pytest.raises(ValidationError):
        PropensityScore(**_score(score_name="open_td_mortgage"))
