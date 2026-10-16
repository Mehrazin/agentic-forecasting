"""Tests for scripts/fetch_fred.py (no network; the FRED client is faked)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_script():
    """Import scripts/fetch_fred.py as a module (it has a __main__ guard)."""
    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / "fetch_fred.py"
    spec = importlib.util.spec_from_file_location("fetch_fred", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeClient:
    """FRED client double: empty observations, canned metadata."""

    def __init__(self, *args, **kwargs):
        pass

    def observations(self, sid):
        return pd.DataFrame(columns=["date", "value"])

    def series_info(self, sid):
        return {
            "title": "t",
            "frequency": "M",
            "units": "u",
            "seasonal_adjustment": "sa",
            "last_updated": "2024-02-01 12:00:00",
        }


def test_fetch_all_empty_frame_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Empty frames produce null start/end (not "nan") and carry last_updated."""
    mod = _load_script()
    monkeypatch.setattr(mod, "FredClient", _FakeClient)
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    manifest = mod.fetch_all(["DGS5"], refresh=True)
    entry = manifest["DGS5"]
    assert entry["rows"] == 0
    assert entry["start"] is None
    assert entry["end"] is None
    assert entry["last_updated"] == "2024-02-01 12:00:00"
    assert (tmp_path / "fred_manifest.json").is_file()
    assert (tmp_path / "DGS5.parquet").is_file()


def test_fetch_all_rejects_unknown_series(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unknown series ids raise ValueError (not a raw KeyError on SERIES)."""
    mod = _load_script()
    monkeypatch.setattr(mod, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="unknown series"):
        mod.fetch_all(["NOPE"], refresh=True)


class _Sleepless:
    """Stand-in for the time module: records sleeps, never really waits."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def sleep(self, s: float) -> None:
        self.delays.append(s)


def test_fetch_series_retries_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transport blips wrapped as FredUnavailableError are retried by cause."""
    import urllib.error

    import td_one.agent.retry as retry_mod
    from td_one.fred.live import FredUnavailableError

    mod = _load_script()
    calls = {"n": 0}

    class _BlipClient:
        def observations(self, sid):
            calls["n"] += 1
            raise FredUnavailableError("FRED request failed: blip") from urllib.error.URLError(
                "blip"
            )

        def series_info(self, sid):
            raise AssertionError("unreachable")

    sleepless = _Sleepless()
    monkeypatch.setattr(retry_mod, "time", sleepless)
    with pytest.raises(urllib.error.URLError):
        mod._fetch_series(_BlipClient(), "DGS5")
    assert calls["n"] == 3  # _FETCH_RETRY max_attempts
    assert len(sleepless.delays) == 2


def test_fetch_series_fatal_without_transport_cause() -> None:
    """FredUnavailableError with no transport cause is not retried."""
    from td_one.fred.live import FredUnavailableError

    mod = _load_script()
    calls = {"n": 0}

    class _FatalClient:
        def observations(self, sid):
            calls["n"] += 1
            raise FredUnavailableError("FRED credential 'x' unavailable")

        def series_info(self, sid):
            raise AssertionError("unreachable")

    with pytest.raises(FredUnavailableError, match="unavailable"):
        mod._fetch_series(_FatalClient(), "DGS5")
    assert calls["n"] == 1
