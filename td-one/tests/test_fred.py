"""Tests for the FRED mock series and file cache.

All deterministic; no network.
"""

from __future__ import annotations

from datetime import date
import os
from pathlib import Path

import pandas as pd
import pytest

from td_one.fred import (
    FIXTURE_END,
    FIXTURE_START,
    SUPPORTED_SERIES,
    FileCache,
    get_series,
    get_series_live,
)


def test_get_series_monthly_and_deterministic() -> None:
    a = get_series("DGS5", date(2021, 3, 15), date(2022, 5, 20))
    b = get_series("DGS5", date(2021, 3, 1), date(2022, 5, 1))
    pd.testing.assert_series_equal(a, b)
    assert a.index[0] == pd.Timestamp("2021-03-01")
    assert a.index[-1] == pd.Timestamp("2022-05-01")
    assert len(a) == 15  # monthly over Mar 2021 .. May 2022
    assert isinstance(a.index, pd.DatetimeIndex)
    assert a.name == "DGS5"


def test_get_series_plausible_ranges() -> None:
    dgs5 = get_series("DGS5", FIXTURE_START, FIXTURE_END)
    assert ((dgs5 >= 0.0) & (dgs5 <= 10.0)).all()
    cpi = get_series("CPI_YOY", FIXTURE_START, FIXTURE_END)
    assert ((cpi >= 0.0) & (cpi <= 12.0)).all()
    unrate = get_series("UNRATE", FIXTURE_START, FIXTURE_END)
    assert ((unrate >= 0.0) & (unrate <= 20.0)).all()
    # COVID shock visible in unemployment fixture
    assert unrate.loc["2020-05-01"] > 10.0
    # 2022 inflation spike visible
    assert cpi.loc["2022-06-01"] > 6.0


def test_get_series_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown series"):
        get_series("GDP", date(2021, 1, 1), date(2021, 6, 1))


def test_get_series_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="after end"):
        get_series("DGS5", date(2022, 1, 1), date(2021, 1, 1))
    with pytest.raises(ValueError, match="outside fixture range"):
        get_series("DGS5", date(2019, 1, 1), date(2021, 1, 1))
    with pytest.raises(ValueError, match="outside fixture range"):
        get_series("DGS5", date(2024, 1, 1), date(2026, 6, 1))


def test_live_client_unavailable_credential() -> None:
    """A bogus vault credential name raises FredUnavailableError, never leaks."""
    from td_one.fred.live import FredClient, FredUnavailableError

    client = FredClient(credential="custom.fred-does-not-exist")
    with pytest.raises(FredUnavailableError, match="unavailable"):
        client.observations("DGS5", start=date(2024, 1, 1), end=date(2024, 1, 10))


@pytest.mark.skipif(
    os.environ.get("TD_ONE_LIVE_FRED") != "1",
    reason="live FRED test: set TD_ONE_LIVE_FRED=1 to run (needs network + vault key)",
)
def test_get_series_live_roundtrip() -> None:
    """Live fetch returns a date-bounded float series."""
    s = get_series_live("DGS5", date(2024, 1, 1), date(2024, 1, 31))
    assert len(s) > 0
    assert s.index.min().date() >= date(2024, 1, 1)
    assert s.index.max().date() <= date(2024, 1, 31)
    assert s.dtype == float


def test_all_supported_series_load() -> None:
    for sid in SUPPORTED_SERIES:
        s = get_series(sid, date(2023, 1, 1), date(2023, 12, 1))
        assert len(s) == 12
        assert s.notna().all()


def test_file_cache_roundtrip(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    assert cache.get("DGS5_2020") is None  # miss
    df = pd.DataFrame({"month": ["2020-01", "2020-02"], "value": [1.0, 2.0]})
    cache.put("DGS5_2020", df)
    out = cache.get("DGS5_2020")
    assert out is not None
    pd.testing.assert_frame_equal(out, df)


def test_file_cache_rejects_traversal(tmp_path: Path) -> None:
    cache = FileCache(tmp_path)
    with pytest.raises(ValueError, match="invalid cache key"):
        cache.put("../evil", pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="invalid cache key"):
        cache.get("../../etc/passwd")
