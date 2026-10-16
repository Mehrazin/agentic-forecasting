"""Mock-mode tests for exp3 (M2 ReAct agent vs M0).

No real LLM. run_exp3 is tested with injected fake tool callables and
tmp_path; the real-tools wiring is tested against td_one.tools.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from exp3_m2 import (
    build_panels,
    evaluate_m0,
    run_exp3,
    wire_tools,
    write_pilot_segments,
)


def _fake_tools() -> dict:
    """Fake tools honoring the args contract (segment_id, run_date)."""

    def data_segment(args: dict) -> str:
        return json.dumps(
            [
                {
                    "month": "2025-05-31",
                    "n_renewals": 200,
                    "n_switches": 20,
                }
            ]
        )

    def evidence(args: dict) -> str:
        return json.dumps([{"id": "boc-2025-06", "fact": "held at 2.75%"}])

    def forecasting(args: dict) -> str:
        return json.dumps({"0.1": 0.08, "0.5": 0.10, "0.9": 0.12})

    return {
        "data_segment": data_segment,
        "evidence": evidence,
        "forecasting": forecasting,
    }


def test_build_panels_returns_three_panels_of_twelve_batches() -> None:
    panels = build_panels(n_customers=100, n_months=24)
    assert len(panels) == 3
    for p in panels:
        assert len(p["N"]) == 12 and len(p["A"]) == 12
        assert bool((np.asarray(p["N"]) >= np.asarray(p["A"])).all())
        assert len(p["months"]) == 12


def test_write_pilot_segments_writes_parquet(tmp_path: Path) -> None:
    panels = build_panels(n_customers=100, n_months=24)
    write_pilot_segments(panels, segments_dir=tmp_path)
    for i in range(3):
        path = tmp_path / f"seg-{i}.parquet"
        assert path.exists()
        import pandas as pd

        df = pd.read_parquet(path)
        assert list(df.columns) == ["month", "n_renewals", "n_switches"]
        assert len(df) == 12


def test_evaluate_m0_computes_finite_metrics() -> None:
    panels = build_panels(n_customers=100, n_months=24)
    rows = evaluate_m0(panels, seed=3)
    assert len(rows) == 3
    for r in rows:
        assert math.isfinite(r["crps"])
        assert math.isfinite(r["mae"])
        assert 0.0 <= r["coverage"] <= 1.0


def test_wire_tools_imports_real_tools() -> None:
    tools = wire_tools()
    assert set(tools) == {"data_segment", "evidence", "forecasting"}
    assert all(callable(t) for t in tools.values())


def test_wire_tools_skips_cleanly_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "td_one.tools", None)
    with pytest.raises(SystemExit) as exc:
        wire_tools()
    assert exc.value.code == 0


def test_run_exp3_with_injected_tools_writes_json(tmp_path: Path) -> None:
    panels = build_panels(n_customers=100, n_months=24)
    out = run_exp3(out_dir=tmp_path, panels=panels, tools=_fake_tools(), seed=5)
    assert out == tmp_path / "exp3_results.json"
    assert (tmp_path / "exp3_report.md").exists()
    data = json.loads(out.read_text())
    assert data["mode"] == "mock"
    assert len(data["panels"]) == 3
    for p in data["panels"]:
        assert math.isfinite(p["crps"]) and math.isfinite(p["m2_crps"])
        assert p["m2_steps"] <= 6
        assert p["evidence_cards"] == 1
