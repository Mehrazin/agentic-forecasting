"""Regression test for the BoC pilot_data.json forward-fill bug (2026-10-19).

The FRED monthly rate series (IRSTCB01CAM156N.parquet) ends in 2023.
Before the fix, pilot.py derived decisions for 2024-2026 meetings from
stale 2023 rates, producing 24 fake rows with decision=0 and
rate_momentum=0.0 (7 real BoC cuts turned into holds, plus 2 future
meetings). This test guards against reintroducing post-cutoff meetings.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

PILOT_DATA = Path(__file__).resolve().parents[1] / "benchmarks" / "boc_rates" / "pilot_data.json"
CUTOFF = date(2023, 12, 31)


def _load() -> list[dict]:
    with open(PILOT_DATA) as f:
        return json.load(f)


def test_no_meetings_after_rate_series_cutoff() -> None:
    """No meeting in pilot_data.json may postdate the FRED rate series (ends 2023)."""
    rows = _load()
    late = [r for r in rows if r["meeting_date"] > CUTOFF.isoformat()]
    assert not late, f"{len(late)} meetings after {CUTOFF}: {[r['meeting_date'] for r in late[:5]]}"


def test_no_stale_forward_fill_block() -> None:
    """Guard against a block of identical fake-hold rows (the 2026-10-19 bug signature).

    The bug produced 24 consecutive post-2023 rows with decision=0 AND
    rate_momentum=0.0 from stale rates. Genuine long hold periods (e.g.
    2011-2014 at 1.0%) are fine — so this checks that no such block occurs
    at or after the rate-series cutoff.
    """
    rows = _load()
    bad = [
        r for r in rows
        if r["meeting_date"] >= "2024-01-01"
        and r["decision"] == 0
        and r["rate_momentum"] == 0.0
    ]
    assert not bad, f"{len(bad)} stale-looking rows from 2024 onward"


def test_decisions_include_cuts_and_hikes() -> None:
    """Sanity: the truncated series must still contain all three decision classes."""
    rows = _load()
    decisions = {r["decision"] for r in rows}
    assert decisions == {-1, 0, 1}, f"expected cuts/holds/hikes, got {decisions}"


def test_meeting_count_matches_truncation() -> None:
    """119 meetings through 2023-12-31 (143 before the fix, incl. 24 stale + 2 future)."""
    rows = _load()
    assert len(rows) == 119, f"expected 119 rows after truncation, got {len(rows)}"
