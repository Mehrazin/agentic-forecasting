"""R2-4: exp14 rerun with a COMMON 0.10-0.90 CRPS grid.

Soundness fix for exp14: the original evaluated classical models on the
19-level COMMON_GRID (0.05-0.95) but TSFMs on the intersection with their
native levels (TimesFM only 0.10-0.90). Different quadrature grids ->
not directly comparable CRPS numbers.

This rerun evaluates EVERY model on the same 9-level grid
[0.1, 0.2, ..., 0.9]:
  - Classical (naive / snaive / conf_naive): residual quantiles computed
    directly on the 9-level grid.
  - Chronos2 (native 21 levels, 0.025-0.975): interpolated to the grid
    (genuine interpolation, no extrapolation).
  - TimesFM-2.5 (native 9 levels, 0.1-0.9): exact, no interpolation.

Framing A only (framing B is binary; no quantile-grid issue).
Same M4 Monthly sample as exp14 (Finance+Macro, seed 0, n=2000).
Chronos2 is stochastic -> 3 seeds (torch.manual_seed before batches).
TimesFM-2.5 quantile heads are deterministic -> 1 seed.

Usage:
    cd td-one && .venv/bin/python experiments/r2_exp14_common_grid.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from td_one.evaluation.metrics import interval_coverage  # noqa: E402
from td_one.evaluation.real_benchmark import (  # noqa: E402
    conformal_naive_quantiles,
    crps_from_quantiles,
    denormalize_quantiles,
    interp_quantiles_to_grid,
    normalize_to_unit_interval,
    trailing_residual_quantiles,
)
from td_one.models.tsfm import (  # noqa: E402
    CHRONOS2_LEVELS,
    TIMESFM_LEVELS,
    Chronos2Wrapper,
    TimesFMWrapper,
)

# Import the exp14 helpers we reuse (data loading + batching logic).
sys.path.insert(0, str(ROOT / "experiments"))
from exp14_m4_real import (  # noqa: E402
    _batch_chronos_quantiles,
    _batch_timesfm_quantiles,
    load_m4_sample,
)

#: The common grid: every model is scored on exactly these 9 levels.
GRID_09 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

HORIZON = 6
N_A = 2000
SEED = 0
BATCH = 16
MAX_CONTEXT = 512
CHRONOS_SEEDS = (0, 1, 2)


def framing_a_classical_common(history: np.ndarray) -> dict[str, np.ndarray]:
    """Naive / snaive / conf_naive quantile forecasts on GRID_09, shape (9, H)."""
    h = HORIZON
    resid = trailing_residual_quantiles(history, GRID_09, h, kind="naive")
    resid_s = trailing_residual_quantiles(history, GRID_09, h, kind="snaive")
    q_naive = np.stack([float(history[-1]) + resid[s] for s in range(1, h + 1)], axis=1)
    q_snaive = np.stack(
        [float(history[-12 + (s - 1)]) + resid_s[s] for s in range(1, h + 1)], axis=1
    )
    q_conf = conformal_naive_quantiles(history, GRID_09, h)
    return {"naive": q_naive, "snaive": q_snaive, "conf_naive": q_conf}


def run_tsfm_common_grid(
    series: list[dict],
    valid_idx: list[int],
    actuals: list[np.ndarray],
    tsfm_name: str,
    seed: int | None,
) -> dict[str, float]:
    """Run one TSFM on the valid subset, scored on GRID_09.

    Sets torch.manual_seed(seed) before batched inference for
    reproducibility of Chronos2's internal sampling.
    """
    import torch

    wcls = Chronos2Wrapper if tsfm_name == "chronos2" else TimesFMWrapper
    wrapper = wcls(
        prediction_length=HORIZON,
        use_mock=False,
        weights_dir=f"models/weights/{tsfm_name}",
    )
    contexts: list[np.ndarray] = []
    scales: list[tuple[float, float]] = []
    for i in valid_idx:
        hist = series[i]["train"][-MAX_CONTEXT:]
        norm, lo, hi = normalize_to_unit_interval(hist)
        contexts.append(norm)
        scales.append((lo, hi))
    if seed is not None:
        torch.manual_seed(seed)
    batch_fn = _batch_chronos_quantiles if tsfm_name == "chronos2" else _batch_timesfm_quantiles
    results = batch_fn(wrapper, contexts)
    wrapper.unload()
    del wrapper
    gc.collect()

    crps_list: list[float] = []
    cov_list: list[float] = []
    native = CHRONOS2_LEVELS if tsfm_name == "chronos2" else TIMESFM_LEVELS
    # Sanity: GRID_09 must lie within the native range (no extrapolation).
    assert float(native.min()) <= 0.1 and float(native.max()) >= 0.9, (
        f"{tsfm_name} native range {native.min()}-{native.max()} "
        "does not cover GRID_09"
    )
    for (levels, vals), (lo, hi), actual in zip(results, scales, actuals, strict=True):
        qv = interp_quantiles_to_grid(levels, denormalize_quantiles(vals, lo, hi), grid=GRID_09)
        crps_list.append(crps_from_quantiles(GRID_09, qv, actual))
        cov_list.append(interval_coverage(qv[0], qv[-1], actual))
    del contexts, results
    gc.collect()
    return {
        "crps": float(np.mean(crps_list)),
        "coverage": float(np.mean(cov_list)),
        "n": len(crps_list),
        "seed": seed,
    }


def main() -> None:
    t_start = time.time()
    print("Loading M4 sample (same as exp14)...", flush=True)
    series = load_m4_sample(N_A, SEED)
    print(f"  {len(series)} series", flush=True)

    results: dict = {
        "meta": {
            "date": datetime.now(UTC).date().isoformat(),
            "run_at_utc": datetime.now(UTC).isoformat(),
            "n_a": N_A,
            "horizon": HORIZON,
            "seed": SEED,
            "grid": GRID_09.tolist(),
            "framing_a_models": ["naive", "snaive", "conf_naive", "chronos2", "timesfm25"],
            "chronos_seeds": list(CHRONOS_SEEDS),
            "note": (
                "R2-4 soundness rerun of exp14: ALL models scored on the "
                "same 0.10-0.90 quantile grid. Classical residual quantiles "
                "computed directly on GRID_09; Chronos2 interpolated "
                "(interior, no extrapolation); TimesFM exact."
            ),
        },
        "framing_a": {},
    }

    # --- Classical (fast) ---
    print("Classical on GRID_09...", flush=True)
    crps: dict[str, list[float]] = {"naive": [], "snaive": [], "conf_naive": []}
    cov: dict[str, list[float]] = {"naive": [], "snaive": [], "conf_naive": []}
    actuals: list[np.ndarray] = []
    valid_idx: list[int] = []
    for i, s in enumerate(series):
        hist = s["train"]
        if hist.size < 40 or bool(np.isnan(hist).any()):
            continue
        actual = s["test"][:HORIZON]
        if bool(np.isnan(actual).any()):
            continue
        qd = framing_a_classical_common(hist)
        for name in ("naive", "snaive", "conf_naive"):
            qv = qd[name]
            crps[name].append(crps_from_quantiles(GRID_09, qv, actual))
            cov[name].append(interval_coverage(qv[0], qv[-1], actual))
        actuals.append(actual)
        valid_idx.append(i)
    for name in ("naive", "snaive", "conf_naive"):
        results["framing_a"][name] = {
            "crps": float(np.mean(crps[name])),
            "coverage": float(np.mean(cov[name])),
            "n": len(crps[name]),
        }
        print(
            f"  {name:10s} CRPS {results['framing_a'][name]['crps']:.4f}  "
            f"cov80 {results['framing_a'][name]['coverage']:.3f}",
            flush=True,
        )
    del crps, cov

    # --- TimesFM-2.5 (deterministic quantile heads: 1 seed) ---
    print("TimesFM-2.5 on GRID_09 (1 seed)...", flush=True)
    t0 = time.time()
    r = run_tsfm_common_grid(series, valid_idx, actuals, "timesfm25", seed=None)
    results["framing_a"]["timesfm25"] = r
    print(
        f"  timesfm25  CRPS {r['crps']:.4f}  cov80 {r['coverage']:.3f}  "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    # --- Chronos2 (stochastic: 3 seeds) ---
    results["framing_a"]["chronos2_seeds"] = {}
    for sd in CHRONOS_SEEDS:
        print(f"Chronos2 on GRID_09 (seed {sd})...", flush=True)
        t0 = time.time()
        r = run_tsfm_common_grid(series, valid_idx, actuals, "chronos2", seed=sd)
        results["framing_a"]["chronos2_seeds"][str(sd)] = r
        print(
            f"  chronos2 s{sd} CRPS {r['crps']:.4f}  cov80 {r['coverage']:.3f}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        # Checkpoint after each seed.
        out_path = ROOT / "experiments" / "r2_exp14_common_grid_results.json"
        results["meta"]["elapsed_s"] = round(time.time() - t_start, 1)
        out_path.write_text(json.dumps(results, indent=2))
    # Convenience aggregate: mean over seeds.
    seed_crps = [results["framing_a"]["chronos2_seeds"][str(s)]["crps"] for s in CHRONOS_SEEDS]
    seed_cov = [results["framing_a"]["chronos2_seeds"][str(s)]["coverage"] for s in CHRONOS_SEEDS]
    results["framing_a"]["chronos2"] = {
        "crps": float(np.mean(seed_crps)),
        "crps_se": float(np.std(seed_crps, ddof=1) / np.sqrt(len(seed_crps))),
        "coverage": float(np.mean(seed_cov)),
        "n": results["framing_a"]["chronos2_seeds"]["0"]["n"],
        "seeds": {str(s): results["framing_a"]["chronos2_seeds"][str(s)]["crps"] for s in CHRONOS_SEEDS},
    }

    results["meta"]["elapsed_s"] = round(time.time() - t_start, 1)
    out_path = ROOT / "experiments" / "r2_exp14_common_grid_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== R2-4 framing A (common 0.10-0.90 grid) ===", flush=True)
    for name in ("naive", "snaive", "conf_naive", "chronos2", "timesfm25"):
        r = results["framing_a"][name]
        extra = f"  se {r['crps_se']:.4f}" if "crps_se" in r else ""
        print(
            f"  {name:10s} CRPS {r['crps']:.4f}{extra}  cov80 {r['coverage']:.3f}  n {r['n']}",
            flush=True,
        )
    print(f"Wrote {out_path} ({results['meta']['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
