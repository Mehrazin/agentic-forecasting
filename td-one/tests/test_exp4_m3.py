"""Mock-mode tests for exp4 (M3 hybrid driver).

No real LLM: everything runs on scripted deltas. tmp_path keeps artifacts
out of the repo.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from exp4_m3 import SCENARIOS, apply_m3_forecast, run_exp4


def _panels() -> dict[str, dict[str, np.ndarray]]:
    # Small hand-made (N, A) panels keyed by scenario id; deterministic,
    # no simulator needed. Reused across scenarios.
    panel = {
        "N": np.array([200] * 12, dtype=float),
        "A": np.array([18, 22, 19, 25, 21, 24, 20, 23, 26, 22, 24, 25], dtype=float),
    }
    return {str(s["id"]): {"N": panel["N"].copy(), "A": panel["A"].copy()} for s in SCENARIOS}


def test_scripted_deltas_respect_bounds() -> None:
    """Every scenario's mock delta must lie in the literature bound."""
    assert len(SCENARIOS) >= 6
    for s in SCENARIOS:
        assert -0.75 <= s["mock_delta"] <= 0.75, s["id"]


def test_m3_delta_is_clipped_to_bound() -> None:
    """An out-of-bound agent delta must equal the ±0.75-clipped result."""
    panel = _panels()[str(SCENARIOS[0]["id"])]
    clipped = apply_m3_forecast(panel["N"], panel["A"], n_upcoming=200, delta=2.0)
    bound = apply_m3_forecast(panel["N"], panel["A"], n_upcoming=200, delta=0.75)
    assert clipped["adjusted_rate"] == bound["adjusted_rate"]
    assert clipped["adjusted_rate"] > clipped["base_rate"]


def test_run_exp4_mock_writes_json_and_report(tmp_path: Path) -> None:
    """Mock-mode run writes results JSON with finite metrics per scenario."""
    out = run_exp4(out_dir=tmp_path, use_real_llm=False, seed=7, panels=_panels())
    results_path = tmp_path / "exp4_results.json"
    assert out == results_path
    assert results_path.exists()
    assert (tmp_path / "exp4_report.md").exists()

    data = json.loads(results_path.read_text())
    assert data["mode"] == "mock"
    assert len(data["scenarios"]) == len(SCENARIOS)
    for entry in data["scenarios"]:
        assert -0.75 <= entry["delta"] <= 0.75
        assert math.isfinite(entry["m0_crps"])
        assert math.isfinite(entry["m3_crps"])
        assert 0.0 <= entry["m0_coverage"] <= 1.0
        assert 0.0 <= entry["m3_coverage"] <= 1.0
        assert entry["n_test"] > 0


def test_exp2_grid_reference_hardcoded_with_note(tmp_path: Path) -> None:
    """The exp2 fixed-Δ grid row is carried as historical context, flagged."""
    run_exp4(out_dir=tmp_path, use_real_llm=False, seed=7, panels=_panels())
    data = json.loads((tmp_path / "exp4_results.json").read_text())
    grid = data["exp2_fixed_delta_reference"]
    assert grid["deltas"] == [-2.0, -0.75, -0.25, 0.0, 0.25, 0.75, 2.0]
    assert len(grid["crps"]) == 7
    assert "not comparable" in grid["note"].lower() or "not directly" in grid["note"].lower()
