"""Canadian mortgage amortization with semi-annual compounding.

Canadian fixed-rate mortgages compound semi-annually, not monthly.
The equivalent monthly rate is: i_m = (1 + i_annual/2)^(1/6) - 1

Reference: plan §3.1, verified against Excel C001/C002/C003.
"""

from __future__ import annotations


def _monthly_rate(annual_rate_pct: float) -> float:
    """Convert annual rate (semi-annual compounding) to monthly rate."""
    i_semi = annual_rate_pct / 100.0 / 2.0
    return float((1.0 + i_semi) ** (1.0 / 6.0) - 1.0)


def canadian_monthly_payment(
    principal: float,
    annual_rate_pct: float,
    amortization_months: int,
) -> float:
    """Calculate fixed monthly payment with Canadian semi-annual compounding.

    Args:
        principal: Initial loan amount.
        annual_rate_pct: Annual interest rate (percent, e.g. 3.0).
        amortization_months: Total amortization period in months.

    Returns:
        Monthly payment amount.

    """
    if principal <= 0:
        raise ValueError("principal must be positive")
    if amortization_months <= 0:
        raise ValueError("amortization_months must be positive")
    i_m = _monthly_rate(annual_rate_pct)
    if i_m == 0:
        return principal / amortization_months
    # Standard amortization: PMT = P * i / (1 - (1+i)^-n)
    return principal * i_m / (1.0 - (1.0 + i_m) ** (-amortization_months))


def canadian_balance_after_payment(
    balance: float,
    annual_rate_pct: float,
    payment: float,
) -> float:
    """Calculate remaining balance after one monthly payment.

    Args:
        balance: Current balance before payment.
        annual_rate_pct: Annual interest rate (percent).
        payment: Monthly payment amount.

    Returns:
        New balance (never negative; floors at 0).

    """
    i_m = _monthly_rate(annual_rate_pct)
    interest = balance * i_m
    new_balance = balance + interest - payment
    return max(0.0, new_balance)
