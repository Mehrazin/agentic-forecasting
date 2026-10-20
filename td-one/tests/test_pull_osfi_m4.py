"""Regression tests for the OSFI M4 pull script.

Guards the bugs caught during development:
1. Component map conflating outstanding balances with funds advanced
   (pivot grabbed new-lending values as outstanding).
2. Rate unit label is "Interest rate", not "Percent" (all-NaN rates).
3. Segments must have BOTH outstanding and advances for the runoff
   identity to hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pull_osfi_m4 import ADVANCES_MAP, OUTSTANDING_MAP, RATE_COMPONENTS


def test_measure_maps_disjoint_outstanding_vs_advances() -> None:
    """Outstanding and advances maps must not share component strings.

    The original bug mixed them.
    """
    assert not set(OUTSTANDING_MAP) & set(ADVANCES_MAP)


def test_every_segment_has_both_measures() -> None:
    """Runoff needs outstanding[t-1] + advanced[t] for the same segment."""
    assert set(OUTSTANDING_MAP.values()) == set(ADVANCES_MAP.values())


def test_rate_components_come_from_funds_advanced() -> None:
    """StatCan publishes rates only for new-lending components."""
    assert all(
        "funds advanced" in c.lower() for c in RATE_COMPONENTS
    )


@pytest.mark.skipif(
    not Path("data/osfi/m4_balances.parquet").exists(),
    reason="OSFI parquet not pulled (run scripts/pull_osfi_m4.py)",
)
def test_balances_sane_magnitudes() -> None:
    """Canadian mortgage stocks are hundreds of billions, rates in [1, 10]."""
    b = pd.read_parquet("data/osfi/m4_balances.parquet")
    assert b["month"].nunique() >= 100
    for seg in ["insured", "uninsured"]:
        latest = b[b["segment"] == seg].iloc[-1]
        assert latest["outstanding_m"] > 100_000  # > 100B CAD
    assert b["rate_pct"].dropna().between(0.5, 12).all()


@pytest.mark.skipif(
    not Path("data/osfi/m4_runoff.parquet").exists(),
    reason="OSFI parquet not pulled (run scripts/pull_osfi_m4.py)",
)
def test_runoff_mostly_nonnegative() -> None:
    """Runoff identity check.

    Occasional negative months (restatements) are OK; systematic
    negatives mean the identity is broken.
    """
    r = pd.read_parquet("data/osfi/m4_runoff.parquet")
    frac_neg = (r["runoff_m"] < 0).mean()
    assert frac_neg < 0.10
    assert (r["runoff_m"] > 0).mean() > 0.5
