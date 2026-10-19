"""Tests for the LLM-ladder offline infrastructure (rungs 1-2).

Covers: the dataset registry (resolution, timeless descriptions, the
``dataset`` wiring on ``ProposedForecaster``), the grid-cache round trip
and validation, ``grid_to_crps`` against the pinball formula, LLM-grid
injection into classical records, and the blocked-CV evaluation harness.
"""

import json
import re

import numpy as np
import pytest

from td_one.agent.proposed_method import ProposedForecaster
from td_one.llm_rungs.cache import (
    grid_to_crps,
    load_grid_cache,
    save_grid_cache,
)
from td_one.llm_rungs.combine import blocked_cv_predict, build_records_with_llm
from td_one.llm_rungs.datasets import DATASET_SPECS
from td_one.llm_rungs.prompts import LEVELS_9, LEVELS_11

FEATURE_NAMES = [
    "n_changepoints",
    "n_recent_cps",
    "trend_strength",
    "seasonal_strength",
    "vol_ratio",
    "is_stationary",
    "is_nonstationary",
    "acf1",
    "spectral_entropy",
    "hurst",
]
CLASSICAL = ["naive", "m0", "ets", "arima"]


def _fake_record(seed: int, origin: int, models: list[str] | None = None) -> dict:
    rng = np.random.default_rng(seed)
    feats = rng.random(len(FEATURE_NAMES)).tolist()
    feats[3] = float(rng.choice([0.1, 0.8]))  # seasonal_strength varies regimes
    feats[2] = float(rng.choice([0.2, 0.9]))  # trend_strength varies regimes
    models = models or CLASSICAL
    quantiles = {m: (rng.random((9, 1)) * 0.2 + 0.4).tolist() for m in models}
    crps = {m: float(rng.random()) for m in models}
    return {
        "origin": origin,
        "features": feats,
        "quantiles": quantiles,
        "crps": crps,
        "actual": [0.5],
    }


def _fake_cache(origins: list[int], ok: bool = True) -> dict:
    per_origin = []
    for o in origins:
        grid = (np.arange(len(LEVELS_11)) * 0.1 + 0.4).tolist()
        per_origin.append(
            {
                "origin": o,
                "as_of": f"2021-{o % 12 + 1:02d}-01",
                "actual": 0.5,
                "grid": grid,
                "crps": 0.123,
                "ok": ok,
            }
        )
    return {
        "variant": "test",
        "dataset": "food_cpi",
        "model": "test-model",
        "levels": list(LEVELS_11),
        "per_origin": per_origin,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_resolves_all_datasets():
    assert set(DATASET_SPECS) == {"food_cpi", "m4", "smm", "boc_rates", "synthetic"}
    for name, spec in DATASET_SPECS.items():
        rspec = spec.regime_spec_fn()
        assert rspec.regimes, name
        assert spec.models, name
        assert spec.data_path is None or isinstance(spec.data_path, str), name


def test_registry_descriptions_have_no_digits():
    # Leakage smoke test: descriptions are timeless metadata — a digit would
    # look like a data value (a level, a date, a count) baked into the spec.
    for name, spec in DATASET_SPECS.items():
        assert spec.description, name
        assert not re.search(r"\d", spec.description), (
            f"{name}: digit in description: {spec.description!r}"
        )


def test_dataset_param_selects_registry_entry():
    fc = ProposedForecaster(dataset="food_cpi")
    assert fc.regime_spec.name == "generic"
    assert fc.models == ["naive", "m0", "ets", "arima"]
    fc = ProposedForecaster(dataset="m4")
    assert fc.regime_spec.name == "m4"
    fc = ProposedForecaster(dataset="boc_rates")
    assert fc.regime_spec.name == "boc"
    assert fc.models == ["rule", "climatology"]


def test_dataset_param_default_unchanged():
    # No dataset -> legacy behavior, exactly as before the rung-1 change.
    fc = ProposedForecaster()
    assert fc.dataset is None
    assert fc.regime_spec.name == "m4"
    assert fc.models == ["naive", "m0", "ets", "arima"]


def test_dataset_param_unknown_raises():
    with pytest.raises(KeyError, match="unknown dataset"):
        ProposedForecaster(dataset="not_a_dataset")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_round_trip_identical(tmp_path):
    cache = _fake_cache([10, 11, 12])
    path = tmp_path / "grid.json"
    save_grid_cache(
        path,
        cache["variant"],
        cache["dataset"],
        cache["model"],
        cache["levels"],
        cache["per_origin"],
    )
    loaded = load_grid_cache(path)
    assert loaded == cache
    # and the file itself is the documented schema
    raw = json.loads(path.read_text())
    assert set(raw) == {"variant", "dataset", "model", "levels", "per_origin"}
    assert set(raw["per_origin"][0]) == {
        "origin",
        "as_of",
        "actual",
        "grid",
        "crps",
        "ok",
    }


def test_cache_rejects_bad_as_of(tmp_path):
    cache = _fake_cache([10])
    cache["per_origin"][0]["as_of"] = "10/2021"
    with pytest.raises(ValueError, match="as_of"):
        save_grid_cache(
            tmp_path / "bad.json",
            cache["variant"],
            cache["dataset"],
            cache["model"],
            cache["levels"],
            cache["per_origin"],
        )


def test_cache_rejects_grid_length_mismatch(tmp_path):
    cache = _fake_cache([10])
    cache["per_origin"][0]["grid"] = [0.5] * 9  # levels has 11
    with pytest.raises(ValueError, match="grid"):
        save_grid_cache(
            tmp_path / "bad.json",
            cache["variant"],
            cache["dataset"],
            cache["model"],
            cache["levels"],
            cache["per_origin"],
        )


def test_load_grid_cache_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_grid_cache(tmp_path / "nope.json")


def test_load_grid_cache_corrupt_raises(tmp_path):
    p = tmp_path / "corrupt.json"
    p.write_text('{"variant": "x"}')  # missing keys
    with pytest.raises(ValueError, match="schema validation"):
        load_grid_cache(p)


def test_grid_to_crps_matches_pinball():
    # Independent reimplementation of the exp53 formula.
    rng = np.random.default_rng(0)
    grid = np.sort(rng.random(9)) + 0.2
    levels = np.array(LEVELS_9)
    actual = 0.55
    expected = float(
        np.mean(
            [
                2.0 * ((actual - q) * lv - (actual - q) * (actual < q))
                for q, lv in zip(grid, levels, strict=True)
            ]
        )
    )
    assert grid_to_crps(grid, levels, actual) == pytest.approx(expected, rel=1e-12)


def test_grid_to_crps_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        grid_to_crps([0.5] * 9, [0.1, 0.2], 0.5)


# ---------------------------------------------------------------------------
# Combine: build_records_with_llm
# ---------------------------------------------------------------------------


def test_build_records_injects_llm_expert():
    recs = [_fake_record(s, origin=100 + s) for s in range(4)]
    cache = _fake_cache([100, 101, 102, 103])
    out = build_records_with_llm(recs, cache, FEATURE_NAMES)
    assert len(out) == 4
    for r in out:
        assert "llm" in r["quantiles"]
        assert "llm" in r["crps"]
        q = np.asarray(r["quantiles"]["llm"])
        assert q.shape == (9, 1)
        # 11-level grid subset to our 9: indices of 0.1..0.9 in LEVELS_11
        expected_grid = np.array(cache["per_origin"][0]["grid"])[
            [LEVELS_11.index(lv) for lv in LEVELS_9]
        ]
        assert np.allclose(q[:, 0], expected_grid)
        assert r["crps"]["llm"] == pytest.approx(
            grid_to_crps(expected_grid, LEVELS_9, 0.5), rel=1e-12
        )
    # inputs not mutated
    assert all("llm" not in r["quantiles"] for r in recs)


def test_build_records_drops_failed_origins():
    recs = [_fake_record(s, origin=100 + s) for s in range(4)]
    cache = _fake_cache([100, 101, 102, 103])
    cache["per_origin"][1]["ok"] = False  # origin 101 failed
    cache["per_origin"][3]["ok"] = False  # origin 103 failed
    out = build_records_with_llm(recs, cache, FEATURE_NAMES)
    assert [r["origin"] for r in out] == [100, 102]


def test_build_records_named_expert():
    """Multiple caches can be injected as distinct experts (exp58 hybrid)."""
    recs = [_fake_record(s, origin=100 + s) for s in range(3)]
    cache_a = _fake_cache([100, 101, 102])
    cache_b = _fake_cache([100, 101, 102])
    out_a = build_records_with_llm(recs, cache_a, FEATURE_NAMES, expert_name="llm-V1")
    out_b = build_records_with_llm(recs, cache_b, FEATURE_NAMES, expert_name="llm-V2")
    # same origins kept, distinct expert keys, no cross-contamination
    assert [r["origin"] for r in out_a] == [100, 101, 102]
    for r in out_a:
        assert "llm-V1" in r["quantiles"] and "llm" not in r["quantiles"]
    for r in out_b:
        assert "llm-V2" in r["quantiles"] and "llm" not in r["quantiles"]
    # stacking two named injections keeps both experts on the same origins
    stacked = build_records_with_llm(out_a, cache_b, FEATURE_NAMES, expert_name="llm-V2")
    assert len(stacked) == 3
    for r in stacked:
        assert "llm-V1" in r["quantiles"] and "llm-V2" in r["quantiles"]


def test_build_records_rejects_feature_mismatch():
    recs = [_fake_record(0, origin=100)]
    cache = _fake_cache([100])
    with pytest.raises(ValueError, match="features"):
        build_records_with_llm(recs, cache, FEATURE_NAMES[:5])


# ---------------------------------------------------------------------------
# Combine: router picks the LLM expert when it wins on train
# ---------------------------------------------------------------------------


def _regime_crafted_record(seed: int, origin: int, regime: str) -> dict:
    r = _fake_record(seed, origin)
    if regime == "seasonal":
        r["features"][3] = 0.8  # seasonal_strength > 0.5
    elif regime == "trending":
        r["features"][3] = 0.1
        r["features"][2] = 0.9  # trend_strength > 0.7
    else:  # noisy
        r["features"][3] = 0.1
        r["features"][2] = 0.2
    return r


def test_router_picks_llm_as_expert_when_best_on_train():
    # 15 records per m4 regime (>= MIN_REGIME_N=10); the LLM grid is
    # near-perfect on every origin, classical experts are noise.
    recs, per_origin = [], []
    seed = 0
    for regime in ("seasonal", "trending", "noisy"):
        for _ in range(15):
            r = _regime_crafted_record(seed, origin=200 + seed, regime=regime)
            for m in CLASSICAL:
                r["crps"][m] = 1.0 + seed * 0.001
            recs.append(r)
            grid = (np.array(LEVELS_11) * 0.01 + 0.495).tolist()  # tight around 0.5
            per_origin.append(
                {
                    "origin": 200 + seed,
                    "as_of": "2021-06-01",
                    "actual": 0.5,
                    "grid": grid,
                    "crps": 0.0,
                    "ok": True,
                }
            )
            seed += 1
    cache = {
        "variant": "test",
        "dataset": "food_cpi",
        "model": "test-model",
        "levels": list(LEVELS_11),
        "per_origin": per_origin,
    }
    llm_recs = build_records_with_llm(recs, cache, FEATURE_NAMES)
    assert len(llm_recs) == 45

    fc = ProposedForecaster(h=1, regime_spec=DATASET_SPECS["m4"].regime_spec_fn())
    fc.models = [*CLASSICAL, "llm"]
    fc.fit(llm_recs, FEATURE_NAMES)
    assert set(fc._expert_per_regime.values()) == {"llm"}


# ---------------------------------------------------------------------------
# Combine: blocked_cv_predict
# ---------------------------------------------------------------------------


def test_blocked_cv_end_to_end():
    recs = [_fake_record(s, origin=300 + s) for s in range(20)]
    res = blocked_cv_predict(
        recs,
        FEATURE_NAMES,
        lambda: ProposedForecaster(h=1, dataset="food_cpi"),
        n_folds=4,
    )
    assert res["n"] == 20
    assert res["n_folds"] == 4
    assert len(res["crps"]) == 20
    assert len(res["baseline_crps"]) == 20
    assert all(np.isfinite(res["crps"]))
    stats = res["stats"]
    assert 0.0 <= stats["p"] <= 1.0
    assert 0.0 <= stats["win_rate"] <= 1.0
    assert res["baseline"].startswith("best_single(")
    assert len(res["per_fold"]) == 4
    assert sum(f["n_test"] for f in res["per_fold"]) == 20


def test_blocked_cv_series_folds_replicates_kfold_by_series():
    # Same fold assignment as exp42's kfold_by_series (seed=0).
    recs = []
    for s in range(6):  # 6 series x 3 origins
        for o in range(3):
            r = _fake_record(100 * s + o, origin=400 + o)
            r["series_id"] = f"s{s}"
            recs.append(r)
    res = blocked_cv_predict(
        recs,
        FEATURE_NAMES,
        lambda: ProposedForecaster(h=1, dataset="m4"),
        n_folds=3,
        fold_mode="series",
        seed=0,
    )
    assert res["fold_mode"] == "series"
    assert sum(f["n_test"] for f in res["per_fold"]) == len(recs)
    # every origin scored exactly once
    assert all(np.isfinite(res["crps"]))


def test_blocked_cv_bad_baseline_rejected():
    recs = [_fake_record(s, origin=500 + s) for s in range(10)]
    with pytest.raises(ValueError, match="baseline"):
        blocked_cv_predict(
            recs,
            FEATURE_NAMES,
            lambda: ProposedForecaster(),
            n_folds=2,
            baseline="nope",
        )


def test_blocked_cv_equal_tool_baseline():
    recs = [_fake_record(s, origin=600 + s) for s in range(10)]
    res = blocked_cv_predict(
        recs,
        FEATURE_NAMES,
        lambda: ProposedForecaster(h=1, dataset="food_cpi"),
        n_folds=2,
        baseline="equal_tool",
    )
    assert res["baseline"] == "equal_tool"
    assert len(res["crps"]) == 10
