"""Tests for Canadian mortgage amortization (semi-annual compounding).

Verification values from plan §3.1 (Excel C001):
- B_t = $420,000, rate = 3.0%, n = 264 months
- PMT = $2,171.20, B_{t+1} = $418,872.30 (to the cent)
"""


from td_one.data_gen.amortization import (
    canadian_balance_after_payment,
    canadian_monthly_payment,
)


def test_c001_payment_exact() -> None:
    """C001: $420k, 3.0%, 264 months -> $2,171.20."""
    pmt = canadian_monthly_payment(
        principal=420000.0,
        annual_rate_pct=3.0,
        amortization_months=264,
    )
    assert abs(pmt - 2171.20) < 0.01, f"got {pmt}"


def test_c001_balance_after_payment_exact() -> None:
    """C001: after one $2,171.20 payment, balance = $418,872.30."""
    pmt = 2171.20
    new_balance = canadian_balance_after_payment(
        balance=420000.0,
        annual_rate_pct=3.0,
        payment=pmt,
    )
    assert abs(new_balance - 418872.30) < 0.01, f"got {new_balance}"


def test_payment_increases_with_rate() -> None:
    """Higher rate -> higher payment (sanity)."""
    pmt_low = canadian_monthly_payment(400000.0, 2.5, 300)
    pmt_high = canadian_monthly_payment(400000.0, 5.0, 300)
    assert pmt_high > pmt_low


def test_balance_decreases_with_payment() -> None:
    """Balance decreases after payment (sanity)."""
    new_bal = canadian_balance_after_payment(400000.0, 3.0, 2000.0)
    assert new_bal < 400000.0
    assert new_bal > 0
