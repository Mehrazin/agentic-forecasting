"""Tests for the agent tools: data_segment, evidence, forecasting.

All deterministic; no network. Filesystem fixtures are written to tmp_path
and wired in via monkeypatching the module-level directory constants.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import td_one.tools.data_segment as ds_mod
import td_one.tools.evidence as ev_mod
from td_one.tools import data_segment_tool, evidence_tool, forecast


def _panel(months: int = 12, n: int = 200, rate: float = 0.12) -> pd.DataFrame:
    """Synthetic monthly panel with a constant switching rate."""
    dates = pd.date_range("2024-01-01", periods=months, freq="MS").date
    switches = [int(round(n * rate))] * months
    return pd.DataFrame({"month": list(dates), "n_renewals": [n] * months, "n_switches": switches})


# --- data_segment_tool -------------------------------------------------------


def test_data_segment_future_run_date_raises() -> None:
    with pytest.raises(ValueError, match="future"):
        data_segment_tool("any", date(2100, 1, 1))


def test_data_segment_unknown_segment_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ds_mod, "SEGMENTS_DIR", tmp_path)
    out = data_segment_tool("nope", date(2025, 6, 1))
    assert list(out.columns) == ["month", "n_renewals", "n_switches"]
    assert out.empty


def test_data_segment_masks_future_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ds_mod, "SEGMENTS_DIR", tmp_path)
    _panel(12).to_parquet(tmp_path / "seg_a.parquet", index=False)
    out = data_segment_tool("seg_a", date(2024, 6, 15))
    # run_date 2024-06-15: months 2024-01..2024-06 visible, later masked
    assert len(out) == 6
    assert out["month"].max() == date(2024, 6, 1)
    assert (out["n_switches"] <= out["n_renewals"]).all()


# --- evidence_tool -----------------------------------------------------------


def _write_corpus(tmp_path, monkeypatch, entries: list[dict]) -> None:
    corpus_dir = tmp_path / "evidence_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "corpus.json").write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(ev_mod, "CORPUS_PATH", corpus_dir / "corpus.json")


def _entry(eid: str, day: str, title: str, text: str) -> dict:
    return {"id": eid, "date": day, "title": title, "text": text, "source": "test"}


def test_evidence_missing_corpus_returns_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ev_mod, "CORPUS_PATH", tmp_path / "nope.json")
    assert evidence_tool("rates", date(2025, 1, 1)) == []


def test_evidence_cutoff_and_ranking(tmp_path, monkeypatch) -> None:
    _write_corpus(
        tmp_path,
        monkeypatch,
        [
            _entry(
                "e1",
                "2024-01-01",
                "Bank of Canada holds rate",
                "policy rate unchanged at five percent",
            ),
            _entry(
                "e2",
                "2024-06-01",
                "Mortgage renewals surge",
                "renewal volumes up as fixed terms mature",
            ),
            _entry("e3", "2025-06-01", "Future rate cut", "this must not leak into a 2025-01 run"),
        ],
    )
    out = evidence_tool("mortgage renewals", date(2025, 1, 1), top_k=5)
    ids = [e["id"] for e in out]
    assert "e3" not in ids  # post-run_date entry masked
    assert ids[0] == "e2"  # title match outranks text-only match
    assert set(out[0]) == {"id", "date", "title", "text", "source"}


def test_evidence_top_k_respected(tmp_path, monkeypatch) -> None:
    _write_corpus(
        tmp_path,
        monkeypatch,
        [_entry(f"e{i}", "2024-01-01", "rate update", f"rate news number {i}") for i in range(10)],
    )
    assert len(evidence_tool("rate", date(2025, 1, 1), top_k=3)) == 3


# --- forecast ----------------------------------------------------------------


def test_forecast_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        forecast("gpt5", _panel(), date(2025, 1, 1))


def test_forecast_empty_panel_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        forecast("m0", pd.DataFrame(), date(2025, 1, 1))


def test_forecast_future_run_date_raises() -> None:
    with pytest.raises(ValueError, match="future"):
        forecast("m0", _panel(), date(2100, 1, 1))


def _assert_valid_forecast(out: dict, levels: list[float]) -> None:
    assert sorted(out.keys()) == sorted(levels)
    vals = [out[lvl] for lvl in sorted(out)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(b >= a for a, b in zip(vals, vals[1:], strict=False))  # monotone


def test_forecast_m0_nine_level_grid() -> None:
    out = forecast("m0", _panel(months=12, n=200, rate=0.12), date(2025, 6, 1))
    _assert_valid_forecast(out, [round(0.1 * i, 1) for i in range(1, 10)])
    # 12% empirical rate: median should sit near it, q10 below q90
    assert out[0.1] < 0.12 < out[0.9] or out[0.1] <= 0.12 <= out[0.9]
    assert abs(out[0.5] - 0.12) < 0.03


def test_forecast_m0_masks_future_months() -> None:
    # Regime change halfway: trailing-6 fit differs by visible window.
    panel = pd.concat(
        [_panel(months=6, n=200, rate=0.10), _panel(months=6, n=200, rate=0.30)],
        ignore_index=True,
    )
    panel["month"] = pd.date_range("2024-01-01", periods=12, freq="MS").date
    full = forecast("m0", panel, date(2025, 12, 31))
    partial = forecast("m0", panel, date(2024, 6, 30))
    # Full window sees the 30% regime; partial window only the 10% regime.
    assert full[0.5] > 0.20 > partial[0.5]


def test_forecast_mocks_deterministic_and_monotone() -> None:
    panel = _panel(months=12, n=200, rate=0.20)
    run = date(2025, 6, 1)
    c1 = forecast("chronos2_mock", panel, run)
    c2 = forecast("chronos2_mock", panel, run)
    assert c1 == c2
    _assert_valid_forecast(
        c1,
        [
            round(x, 2)
            for x in [
                0.01,
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
                0.99,
            ]
        ],
    )
    t1 = forecast("timesfm_mock", panel, run)
    assert t1 == forecast("timesfm_mock", panel, run)
    _assert_valid_forecast(t1, [round(0.1 * i, 1) for i in range(1, 10)])
    # Mocks center near the last observed rate (0.20)
    assert abs(c1[0.50] - 0.20) < 0.05
    assert abs(t1[0.5] - 0.20) < 0.05


def test_forecast_m0_drops_zero_exposure_months() -> None:
    panel = _panel(months=12)
    panel.loc[3:5, "n_renewals"] = 0
    panel.loc[3:5, "n_switches"] = 0
    out = forecast("m0", panel, date(2025, 6, 1))
    _assert_valid_forecast(out, [round(0.1 * i, 1) for i in range(1, 10)])


def _repo_td_one() -> Path:
    """Return the td-one/ directory (one level above this test file)."""
    return Path(__file__).resolve().parents[1]


def test_segments_dir_default_resolves_to_repo_data() -> None:
    """Check SEGMENTS_DIR points at td-one/data/segments.

    Regression: a parents[2]/parents[3] bug silently wrote pilot panels
    into src/data/segments instead of the documented repo data dir.
    """
    import td_one.tools.data_segment as ds_mod

    assert _repo_td_one() / "data" / "segments" == ds_mod.SEGMENTS_DIR


def test_corpus_path_default_points_at_real_corpus() -> None:
    """Check the evidence tool's default corpus path resolves.

    Regression: the same parents bug made the tool read a nonexistent
    path and silently return zero cards.
    """
    import td_one.tools.evidence as ev_mod

    assert (
        _repo_td_one() / "experiments" / "evidence_corpus" / "corpus.json"
    ) == ev_mod.CORPUS_PATH
    assert ev_mod.CORPUS_PATH.is_file()


def test_evidence_tool_reads_real_corpus() -> None:
    """End-to-end against the real frozen corpus (no monkeypatching)."""
    from datetime import date

    from td_one.tools.evidence import evidence_tool

    cards = evidence_tool("Bank of Canada rate decision", date(2025, 1, 1))
    assert len(cards) > 0
    assert all(c["date"] <= "2025-01-01" for c in cards)


def test_fred_cache_default_root_resolves_to_repo_data() -> None:
    """Regression: same parents[2]/parents[3] class of bug in fred/cache."""
    import td_one.fred.cache as cache_mod

    assert _repo_td_one() / "data" / "fred_cache" == cache_mod.DEFAULT_CACHE_ROOT
