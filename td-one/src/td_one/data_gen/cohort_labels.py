"""Ground-truth cohort labels for the loan simulator DGP.

``MortgageSimulatorV2.generate_outcomes`` plants four borrower cohorts
that drive heterogeneous switching propensity (see
``data_gen/simulator_v2.py``). This module re-implements the cohort
assignment as a pure function so experiments can measure whether
discovered groups recover the DGP's true structure — without touching
the simulator.
"""

from __future__ import annotations

COHORTS = ("low_rate", "osfi_exempt", "family_squeeze", "control")


def cohort_label(
    *,
    rate_type: str,
    origination_year: int,
    mortgage_rate_pct: float,
    insured_flag: bool,
    credit_score_band: str,
    household_income_cad: float,
    n_children: int,
) -> str:
    """Assign the planted cohort, mirroring ``generate_outcomes``.

    Check order matters (``low_rate`` first, then ``osfi_exempt``),
    exactly as in the simulator. ``rate_type`` / ``mortgage_rate_pct``
    / ``household_income_cad`` / ``n_children`` are ORIGINATION-period
    values; ``insured_flag`` / ``credit_score_band`` are from the last
    pre-maturity snapshot.
    """
    if rate_type == "fixed" and origination_year == 2021 and mortgage_rate_pct <= 3.0:
        return "low_rate"
    if not insured_flag and credit_score_band == "720_plus":
        return "osfi_exempt"
    if household_income_cad < 75000 and n_children >= 3:
        return "family_squeeze"
    return "control"
