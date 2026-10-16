"""Experiment 14: M4 Monthly real-data benchmark.

Tests whether our synthetic-data conclusions survive real data.

Framing A (continuous, H=6): naive, snaive, conf_naive, chronos2,
    timesfm25 (real weights). Scored with quantile CRPS + 80% coverage.
Framing B (binary direction): is y[t+6] > y[t]? m0_binary,
    conf_m0_binary, chronos2_dir. Scored with Brier + binary CRPS.

Data: M4 Monthly, Finance + Macro categories, stratified sample
(seed 0). Framing A uses 2000 series; framing B uses a 1000-series
stratified subsample (seed 1) for the Chronos-2 direction forecasts.

TSFM batching: the wrappers are per-series, so this script loads each
model once via the wrapper, then batches through the underlying
pipeline (documented intimacy for throughput; the wrapper's validation
and weight-loading logic is reused, not bypassed).

Usage:
    cd td-one && uv run python experiments/exp14_m4_real.py
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

from td_one.evaluation.metrics import brier_score, interval_coverage  # noqa: E402
from td_one.evaluation.real_benchmark import (  # noqa: E402
    COMMON_GRID,
    binary_crps,
    binary_target_up,
    conformal_naive_quantiles,
    conformal_shrink_probability,
    crps_from_quantiles,
    denormalize_quantiles,
    framing_b_origins,
    interp_quantiles_to_grid,
    m0_origin_window,
    normalize_to_unit_interval,
    prob_up_from_quantiles,
    trailing_residual_quantiles,
)
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.conformal import ConformalM0  # noqa: E402
from td_one.models.tsfm import (  # noqa: E402
    CHRONOS2_LEVELS,
    TIMESFM_LEVELS,
    Chronos2Wrapper,
    TimesFMWrapper,
)

M4_DIR = Path("/tmp/m4")
HORIZON = 6
N_A = 2000
N_B = 1000
SEED = 0
BATCH = 16
MAX_CONTEXT = 512


def load_m4_sample(n: int = N_A, seed: int = SEED) -> list[dict]:
    """Load M4 Monthly train/test, filter Finance+Macro, stratified sample.

    Returns a list of dicts: {m4id, category, train, test}.
    Row order of the CSVs matches M4-info order for Frequency==12.
    """
    info = pd.read_csv(M4_DIR / "M4-info.csv")
    monthly = info[(info["Frequency"] == 12) & (info["SP"] == "Monthly")].reset_index(drop=True)
    keep = monthly[monthly["category"].isin(["Finance", "Macro"])].reset_index(drop=True)
    assert len(keep) == 21003, f"expected 21003 Finance+Macro, got {len(keep)}"

    train = pd.read_csv(M4_DIR / "Monthly-train.csv")
    test = pd.read_csv(M4_DIR / "Monthly-test.csv")
    assert len(train) == 48000 and len(test) == 48000

    # Stratified sample: proportional to category sizes, seed 0.
    rng = np.random.default_rng(seed)
    idx: list[int] = []
    for _cat, grp in keep.groupby("category"):
        k = int(round(n * len(grp) / len(keep)))
        idx.extend(rng.choice(grp.index.to_numpy(), size=k, replace=False).tolist())
    # Fix rounding drift to hit exactly n.
    idx = sorted(idx)
    while len(idx) > n:
        idx.pop(int(rng.integers(len(idx))))
    while len(idx) < n:
        cand = int(rng.integers(len(keep)))
        if cand not in idx:
            idx.append(cand)
            idx.sort()

    # Map keep-index -> train row: monthly.iloc[i] <-> train row i.
    # keep was reset; its positional index within `monthly`:
    monthly_pos = {m4id: i for i, m4id in enumerate(monthly["M4id"])}
    out: list[dict] = []
    for ki in idx:
        row = keep.iloc[ki]
        pos = monthly_pos[row["M4id"]]
        assert train.iloc[pos, 0] == row["M4id"], "train row order mismatch"
        tr = train.iloc[pos, 1:].dropna().to_numpy(dtype=float)
        te = test.iloc[pos, 1:].to_numpy(dtype=float)
        assert te.shape[0] == 18, f"{row['M4id']}: test len {te.shape[0]}"
        out.append({"m4id": row["M4id"], "category": row["category"], "train": tr, "test": te})
    return out


def framing_a_classical(history: np.ndarray) -> dict[str, np.ndarray]:
    """Naive / snaive / conf_naive quantile forecasts, shape (19, H)."""
    h = HORIZON
    resid = trailing_residual_quantiles(history, COMMON_GRID, h, kind="naive")
    resid_s = trailing_residual_quantiles(history, COMMON_GRID, h, kind="snaive")
    q_naive = np.stack([float(history[-1]) + resid[s] for s in range(1, h + 1)], axis=1)
    q_snaive = np.stack(
        [float(history[-12 + (s - 1)]) + resid_s[s] for s in range(1, h + 1)], axis=1
    )
    q_conf = conformal_naive_quantiles(history, COMMON_GRID, h)
    return {"naive": q_naive, "snaive": q_snaive, "conf_naive": q_conf}


def _batch_chronos_quantiles(
    wrapper: Chronos2Wrapper, contexts: list[np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Batched Chronos-2 quantile forecasts (uses wrapper-loaded pipeline)."""
    import torch

    wrapper._load()
    assert wrapper._pipeline is not None
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, len(contexts), BATCH):
        batch = contexts[i : i + BATCH]
        tensors = [torch.tensor(c, dtype=torch.float32) for c in batch]
        out = wrapper._pipeline.predict(inputs=tensors, prediction_length=HORIZON, batch_size=BATCH)
        # out is a list of per-series forecasts, each (1, 21, H).
        assert len(out) == len(batch), f"batch {i}: got {len(out)}, want {len(batch)}"
        for item in out:
            vals = np.asarray(item)
            if vals.ndim == 3:
                vals = vals[0]  # drop variate dim -> (21, H)
            vals = np.clip(vals, 0.0, 1.0)
            results.append((CHRONOS2_LEVELS.copy(), vals))
        del tensors, out
    return results


def _batch_timesfm_quantiles(
    wrapper: TimesFMWrapper, contexts: list[np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Batched TimesFM-2.5 quantile forecasts (uses wrapper-loaded model)."""
    wrapper._load()
    assert wrapper._model is not None
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(0, len(contexts), BATCH):
        batch = contexts[i : i + BATCH]
        _, q = wrapper._model.forecast(horizon=HORIZON, inputs=list(batch))
        q_arr = np.asarray(q)
        if q_arr.ndim == 2:
            q_arr = q_arr[None, :, :]
        assert q_arr.shape[0] == len(batch), f"batch {i}: {q_arr.shape[0]} vs {len(batch)}"
        for b in range(q_arr.shape[0]):
            # (H, 10) -> drop point col -> (9, H)
            vals = np.clip(q_arr[b][:, 1:].T, 0.0, 1.0)
            results.append((TIMESFM_LEVELS.copy(), vals))
        del q, q_arr
    gc.collect()
    return results


def run_framing_a(series: list[dict], tsfm_name: str | None) -> dict[str, dict[str, float]]:
    """Run framing A for classical models (+ one TSFM if named).

    Returns {model: {"crps": mean, "coverage": mean, "n": count}}.
    """
    crps: dict[str, list[float]] = {"naive": [], "snaive": [], "conf_naive": []}
    cov: dict[str, list[float]] = {"naive": [], "snaive": [], "conf_naive": []}
    if tsfm_name:
        crps[tsfm_name] = []
        cov[tsfm_name] = []

    # Classical first (fast, no weights).
    classical_q: list[dict[str, np.ndarray]] = []
    actuals: list[np.ndarray] = []
    valid_idx: list[int] = []
    for i, s in enumerate(series):
        hist = s["train"]
        if hist.size < 40 or bool(np.isnan(hist).any()):
            continue
        actual = s["test"][:HORIZON]
        if bool(np.isnan(actual).any()):
            continue
        classical_q.append(framing_a_classical(hist))
        actuals.append(actual)
        valid_idx.append(i)

    for qd, actual in zip(classical_q, actuals, strict=True):
        for name in ("naive", "snaive", "conf_naive"):
            qv = qd[name]
            crps[name].append(crps_from_quantiles(COMMON_GRID, qv, actual))
            i10 = int(np.argmin(np.abs(COMMON_GRID - 0.1)))
            i90 = int(np.argmin(np.abs(COMMON_GRID - 0.9)))
            cov[name].append(interval_coverage(qv[i10], qv[i90], actual))

    # TSFM on the valid subset.
    if tsfm_name:
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
        batch_fn = _batch_chronos_quantiles if tsfm_name == "chronos2" else _batch_timesfm_quantiles
        results = batch_fn(wrapper, contexts)
        wrapper.unload()
        del wrapper
        gc.collect()
        for (levels, vals), (lo, hi), actual in zip(results, scales, actuals, strict=True):
            qv = interp_quantiles_to_grid(levels, denormalize_quantiles(vals, lo, hi))
            crps[tsfm_name].append(crps_from_quantiles(COMMON_GRID, qv, actual))
            i10 = int(np.argmin(np.abs(COMMON_GRID - 0.1)))
            i90 = int(np.argmin(np.abs(COMMON_GRID - 0.9)))
            cov[tsfm_name].append(interval_coverage(qv[i10], qv[i90], actual))
        del contexts, results
        gc.collect()

    return {
        name: {
            "crps": float(np.mean(crps[name])),
            "coverage": float(np.mean(cov[name])),
            "n": len(crps[name]),
        }
        for name in crps
    }


def run_framing_b_m0(series: list[dict]) -> tuple[dict[str, dict[str, list[float]]], int]:
    """Framing B for m0_binary and conf_m0_binary (analytical, fast).

    Returns (per-model {brier_p, brier_y, crps} lists, total n_origins).
    """
    out: dict[str, dict[str, list[float]]] = {
        "m0_binary": {"brier_p": [], "brier_y": [], "crps": []},
        "conf_m0_binary": {"brier_p": [], "brier_y": [], "crps": []},
    }
    n_origins_total = 0
    for s in series:
        full = np.concatenate([s["train"], s["test"]])
        tlen = s["train"].size
        if tlen < 60:
            continue
        origins = framing_b_origins(tlen, 18, n_origins=24, horizon=HORIZON, m0_trailing=12)
        for t in origins:
            y = binary_target_up(full, t, HORIZON)
            win = m0_origin_window(t, HORIZON, 12)
            outcomes = np.array([binary_target_up(full, ox, HORIZON) for ox in win])
            m0 = HistoricalBaseline(trailing_window=12)
            m0.fit(np.ones(12), outcomes)
            p = float(m0.expected_rate)
            cm0 = ConformalM0(trailing_window=12, cal_months=4)
            cm0.fit(np.ones(12), outcomes)
            p_conf = conformal_shrink_probability(p, cm0.qhat_)
            out["m0_binary"]["brier_p"].append(p)
            out["m0_binary"]["brier_y"].append(y)
            out["m0_binary"]["crps"].append(binary_crps(p, y))
            out["conf_m0_binary"]["brier_p"].append(p_conf)
            out["conf_m0_binary"]["brier_y"].append(y)
            out["conf_m0_binary"]["crps"].append(binary_crps(p_conf, y))
            n_origins_total += 1
    return out, n_origins_total


def run_framing_b_chronos(series: list[dict]) -> dict[str, list[float]]:
    """Framing B for chronos2_dir (batched real-weight inference)."""
    # Collect (series, origin) contexts.
    jobs: list[tuple[np.ndarray, float, int]] = []  # (context, current, target)
    for s in series:
        full = np.concatenate([s["train"], s["test"]])
        tlen = s["train"].size
        if tlen < 60:
            continue
        origins = framing_b_origins(tlen, 18, n_origins=24, horizon=HORIZON, m0_trailing=12)
        for t in origins:
            ctx = full[: t + 1][-MAX_CONTEXT:]
            norm, lo, hi = normalize_to_unit_interval(ctx)
            jobs.append((norm, float(full[t]), binary_target_up(full, t, HORIZON), lo, hi))

    wrapper = Chronos2Wrapper(
        prediction_length=HORIZON, use_mock=False, weights_dir="models/weights/chronos2"
    )
    import torch

    wrapper._load()
    assert wrapper._pipeline is not None
    probs: list[float] = []
    targets: list[int] = []
    for i in range(0, len(jobs), BATCH):
        batch = jobs[i : i + BATCH]
        tensors = [torch.tensor(j[0], dtype=torch.float32) for j in batch]
        out = wrapper._pipeline.predict(inputs=tensors, prediction_length=HORIZON, batch_size=BATCH)
        assert len(out) == len(batch), f"batch {i}: got {len(out)}, want {len(batch)}"
        for item, (_, current, y, lo, hi) in zip(out, batch, strict=True):
            vals = np.asarray(item)
            if vals.ndim == 3:
                vals = vals[0]  # drop variate dim -> (21, H)
            vals = denormalize_quantiles(np.clip(vals, 0.0, 1.0), lo, hi)
            q_h6 = interp_quantiles_to_grid(CHRONOS2_LEVELS, vals[:, HORIZON - 1])
            probs.append(prob_up_from_quantiles(COMMON_GRID, q_h6, current))
            targets.append(y)
        del tensors, out
    wrapper.unload()
    del wrapper
    gc.collect()
    return {
        "brier_p": probs,
        "brier_y": [float(t) for t in targets],
        "crps": [binary_crps(p, t) for p, t in zip(probs, targets, strict=True)],
    }


def main() -> None:
    t_start = time.time()
    print("Loading M4 sample...", flush=True)
    series_a = load_m4_sample(N_A, SEED)
    print(
        f"  {len(series_a)} series "
        f"({sum(s['category'] == 'Finance' for s in series_a)} Finance, "
        f"{sum(s['category'] == 'Macro' for s in series_a)} Macro)",
        flush=True,
    )

    results: dict = {
        "meta": {
            "date": datetime.now(UTC).date().isoformat(),
            "run_at_utc": datetime.now(UTC).isoformat(),
            "n_a": N_A,
            "horizon": HORIZON,
            "seed": SEED,
            "framing_a_models": ["naive", "snaive", "conf_naive", "chronos2", "timesfm25"],
        },
        "framing_a": {},
        "framing_b": {},
    }

    print("Framing A: classical...", flush=True)
    fa_classical = run_framing_a(series_a, None)
    results["framing_a"].update(fa_classical)

    for tsfm_name in ("chronos2", "timesfm25"):
        print(f"Framing A: {tsfm_name} (real weights)...", flush=True)
        fa_tsfm = run_framing_a(series_a, tsfm_name)
        results["framing_a"].update({k: v for k, v in fa_tsfm.items() if k == tsfm_name})

    print("Framing A table:", flush=True)
    for name in ("naive", "snaive", "conf_naive", "chronos2", "timesfm25"):
        r = results["framing_a"][name]
        print(
            f"  {name:10s} CRPS {r['crps']:.4f}  cov80 {r['coverage']:.3f}  n {r['n']}", flush=True
        )

    # Framing B on a stratified 1000-series subsample.
    rng = np.random.default_rng(1)
    idx_b: list[int] = []
    cats = pd.Series([s["category"] for s in series_a])
    for cat in ("Finance", "Macro"):
        grp = np.where(cats == cat)[0]
        idx_b.extend(rng.choice(grp, size=N_B // 2, replace=False).tolist())
    series_b = [series_a[i] for i in sorted(idx_b)]
    results["meta"]["n_b"] = len(series_b)

    print("Framing B: M0 models...", flush=True)
    fb_m0, n_b_origins = run_framing_b_m0(series_b)
    print("Framing B: chronos2_dir (real weights)...", flush=True)
    fb_c2 = run_framing_b_chronos(series_b)

    for name, d in (("m0_binary", fb_m0["m0_binary"]), ("conf_m0_binary", fb_m0["conf_m0_binary"])):
        p = np.asarray(d["brier_p"])
        y = np.asarray(d["brier_y"])
        results["framing_b"][name] = {
            "brier": brier_score(p, y),
            "crps": float(np.mean(d["crps"])),
            "n_origins": len(p),
        }
    p = np.asarray(fb_c2["brier_p"])
    y = np.asarray(fb_c2["brier_y"])
    results["framing_b"]["chronos2_dir"] = {
        "brier": brier_score(p, y),
        "crps": float(np.mean(fb_c2["crps"])),
        "n_origins": len(p),
    }
    results["meta"]["n_b_origins"] = n_b_origins

    print("Framing B table:", flush=True)
    for name in ("m0_binary", "conf_m0_binary", "chronos2_dir"):
        r = results["framing_b"][name]
        print(
            f"  {name:14s} Brier {r['brier']:.4f}  CRPS {r['crps']:.4f}  n {r['n_origins']}",
            flush=True,
        )

    results["meta"]["elapsed_s"] = round(time.time() - t_start, 1)
    out_path = ROOT / "experiments" / "exp14_m4_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path} ({results['meta']['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
