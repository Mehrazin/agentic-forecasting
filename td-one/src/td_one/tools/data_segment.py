"""Data & Segment Tool: point-in-time segment panel retrieval.

Segment panels live under ``td-one/data/segments/`` as one parquet file per
segment: ``<segment_id>.parquet`` with columns

- ``month``: month start (date or parseable string)
- ``n_renewals``: renewal count (int)
- ``n_switches``: switch count (int)

The file is written by the scale-up pipeline; this tool only reads and
enforces the point-in-time cutoff. Unknown segment ids return an empty
DataFrame with the canonical columns (the agent can distinguish "no data"
from "bad segment" only by absence of rows, which is intentional: segment
discovery is a separate step).
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from td_one._paths import _repo_root

#: Directory holding per-segment panel parquet files.
SEGMENTS_DIR = _repo_root() / "data" / "segments"

PANEL_COLUMNS = ("month", "n_renewals", "n_switches")


def _empty_panel() -> pd.DataFrame:
    """Empty panel with canonical columns."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in PANEL_COLUMNS})


def data_segment_tool(segment_id: str, run_date: date) -> pd.DataFrame:
    """Return the monthly renewal/switch panel for a segment at point-in-time.

    Rows with ``month > run_date`` are masked (TemporalGuard: the agent can
    never see post-run_date outcomes). Months are inclusive of run_date.

    Args:
        segment_id: Segment identifier (matches ``<segment_id>.parquet``).
        run_date: Point-in-time cutoff (inclusive). Must not be in the future.

    Returns:
        DataFrame with columns ``month`` (date), ``n_renewals`` (int),
        ``n_switches`` (int), sorted by month. Empty (with canonical columns)
        when the segment is unknown or has no visible rows.

    Raises:
        ValueError: If ``run_date`` is after today.

    """
    if run_date > datetime.now(UTC).date():
        raise ValueError(
            f"run_date {run_date} is in the future; latest allowed is {datetime.now(UTC).date()}"
        )
    path = SEGMENTS_DIR / f"{segment_id}.parquet"
    if not path.exists():
        return _empty_panel()
    panel = pd.read_parquet(path)
    if panel.empty:
        return _empty_panel()
    panel = panel.copy()
    panel["month"] = pd.to_datetime(panel["month"]).dt.date
    panel = panel[panel["month"] <= run_date].sort_values("month").reset_index(drop=True)
    panel["n_renewals"] = panel["n_renewals"].astype(int)
    panel["n_switches"] = panel["n_switches"].astype(int)
    return panel.loc[:, ["month", "n_renewals", "n_switches"]]
