r"""Experiment 2 v2: M0 vs M1 with real weights — redesigned per audit.

Design (pre-registered, fixed sample size):
- 3 scenarios: stationary / drift / regime_shift (see data_gen/scenarios.py)
- K seeds per scenario; rolling origins {6, 9, 12} (train on [o-6, o))
- Primary metric: CRPS on count distributions (proper score)
- Secondary: MAE (median), 80% PI empirical coverage
- Oracle anchor: Binomial(N, true_rate) CRPS (noise floor)
- M3 probe: fixed Delta grid on M1a's rate -> CRPS/coverage vs Delta,
  testing the bounded-adjustment hypothesis with a negative control
- M0: trailing-6-month Jeffreys Beta-Binomial (matches its docstring)

Models load ONCE and forecast all scenarios/seeds (weights stay resident).

Usage:
  cd td-one
  uv run --with torch --with chronos-forecasting --with timesfm \\
      --with huggingface_hub python experiments/exp2_real_weights.py

  TD_ONE_MOCK=1 ...  # mock mode (CI, no downloads)
  TD_ONE_SEEDS=5     # seeds per scenario (default 5)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from td_one.agent.controller import MAX_DELTA  # noqa: E402
from td_one.data_gen.scenarios import SCENARIOS, generate_scenario  # noqa: E402
from td_one.evaluation.metrics import crps_counts, interval_coverage, mae  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.hybrid import HybridReadout  # noqa: E402
from td_one.models.tsfm import Chronos2Wrapper, TimesFMWrapper  # noqa: E402

logger = logging.getLogger(__name__)

ORIGINS = (6, 9, 12)
HORIZON = 6
N_COUNT_SAMPLES = 1000
#: M3 probe: bounded adjustments plus an unbounded negative control.
DELTA_GRID = (-2.0, -MAX_DELTA, -0.25, 0.0, 0.25, MAX_DELTA, 2.0)
#: Explicit manifest key map (no substring guessing).
MODEL_KEYS = {"M1a_chronos2": "chronos2", "M1b_timesfm25": "timesfm25"}


def rate_quantiles_to_count_samples(
    levels: np.ndarray,
    q_values: np.ndarray,
    n: float,
    n_samples: int = N_COUNT_SAMPLES,
    seed: int = 0,
) -> np.ndarray:
    """Convert a rate quantile lattice to count predictive samples.

    Inverse-CDF sampling from the quantile lattice, then Binomial(N, rate).

    Args:
        levels: Quantile levels, shape (n_levels,).
        q_values: Rate quantiles, shape (n_levels, horizon).
        n: Number of renewals (single period).
        n_samples: Number of count samples.
        seed: RNG seed.

    Returns:
        Count samples, shape (n_samples, horizon).

    """
    rng = np.random.default_rng(seed)
    n_int = int(round(n))
    h = q_values.shape[1]
    u = rng.random((n_samples, h))
    rates = np.stack(
        [np.interp(u[:, hh], levels, q_values[:, hh]) for hh in range(h)], axis=1
    )
    rates = np.clip(rates, 0.0, 1.0)
    return rng.binomial(n_int, rates)


def run_experiment() -> dict:
    """Full experiment. Returns the results dict."""
    use_mock = os.environ.get("TD_ONE_MOCK", "0") == "1"
    n_seeds = int(os.environ.get("TD_ONE_SEEDS", "5"))
    manifest: dict = {}
    if not use_mock:
        mp = ROOT / "models" / "MANIFEST.json"
        if not mp.exists():
            raise FileNotFoundError(f"{mp} missing; run scripts/bootstrap_models.py")
        manifest = json.loads(mp.read_text())

    print("=" * 64)
    print(f"Experiment 2 v2 ({'MOCK' if use_mock else 'real weights'})")
    print(f"scenarios={list(SCENARIOS)} seeds={n_seeds} origins={list(ORIGINS)}")
    print("=" * 64)

    # Load models ONCE
    t0 = time.time()
    def wdir(k: str) -> str | None:
        return None if use_mock else str(ROOT / manifest["models"][k]["local_path"])
    m1a = Chronos2Wrapper(use_mock=use_mock, weights_dir=wdir("chronos2"))
    m1b = TimesFMWrapper(use_mock=use_mock, weights_dir=wdir("timesfm25"))
    hybrid = HybridReadout()
    if not use_mock:
        print("Loading Chronos-2...")
        m1a._load()
        print("Loading TimesFM-2.5...")
        m1b._load()
    print(f"models ready in {time.time() - t0:.1f}s")

    # results[scenario][model] = list of per-seed pooled CRPS
    results: dict[str, dict[str, list[float]]] = {
        s: {m: [] for m in ("M0", "M1a_chronos2", "M1b_timesfm25", "oracle")}
        for s in SCENARIOS
    }
    mae_res: dict[str, dict[str, list[float]]] = {
        s: {m: [] for m in ("M0", "M1a_chronos2", "M1b_timesfm25", "oracle")}
        for s in SCENARIOS
    }
    cov_res: dict[str, dict[str, list[float]]] = {
        s: {m: [] for m in ("M0", "M1a_chronos2", "M1b_timesfm25")}
        for s in SCENARIOS
    }
    # M3 probe: delta -> list of (crps, coverage) pooled per seed
    m3_crps: dict[str, dict[float, list[float]]] = {
        s: {d: [] for d in DELTA_GRID} for s in SCENARIOS
    }
    m3_cov: dict[str, dict[float, list[float]]] = {
        s: {d: [] for d in DELTA_GRID} for s in SCENARIOS
    }
    timings: dict[str, float] = {"M0": 0.0, "M1a": 0.0, "M1b": 0.0}

    for scenario in SCENARIOS:
        for seed in range(n_seeds):
            data = generate_scenario(scenario, seed=1000 + seed)
            n_all, a_all = data["n_renewals"], data["n_switches"]
            true_all = data["true_rates"]

            seed_crps: dict[str, list[float]] = {m: [] for m in results[scenario]}
            seed_mae: dict[str, list[float]] = {m: [] for m in mae_res[scenario]}
            seed_cov: dict[str, list[float]] = {m: [] for m in cov_res[scenario]}
            seed_m3_crps: dict[float, list[float]] = {d: [] for d in DELTA_GRID}
            seed_m3_cov: dict[float, list[float]] = {d: [] for d in DELTA_GRID}

            for origin in ORIGINS:
                train_n, train_a = n_all[:origin], a_all[:origin]
                train_rates = np.clip(train_a / np.maximum(train_n, 1), 0.01, 0.99)
                test_n, test_a = n_all[origin:origin + HORIZON], a_all[origin:origin + HORIZON]
                test_true = true_all[origin:origin + HORIZON]
                hh = len(test_n)

                # ---- M0 ----
                t_m0 = time.time()
                m0 = HistoricalBaseline(trailing_window=6)
                m0.fit(train_n, train_a)
                m0_conc = (m0._alpha or 1.0) + (m0._beta or 1.0)
                m0_samples = np.stack(
                    [m0.predict_counts(int(round(n)), N_COUNT_SAMPLES, seed=seed)
                     for n in test_n]
                ).T  # (n_samples, h)
                timings["M0"] += time.time() - t_m0
                m0_q10 = np.quantile(m0_samples, 0.10, axis=0)
                m0_q50 = np.quantile(m0_samples, 0.50, axis=0)
                m0_q90 = np.quantile(m0_samples, 0.90, axis=0)
                seed_crps["M0"].append(float(np.mean(crps_counts(m0_samples.T, test_a))))
                seed_mae["M0"].append(mae(m0_q50, test_a))
                seed_cov["M0"].append(interval_coverage(m0_q10, m0_q90, test_a))

                # ---- M1a: Chronos-2 ----
                t_m1a = time.time()
                c_levels, c_vals = m1a.forecast_quantiles(train_rates, horizon=hh)
                c_mean = m1a.forecast_mean(train_rates, horizon=hh)
                m1a_samples = np.array(
                    [
                        rate_quantiles_to_count_samples(
                            c_levels, c_vals[:, [i]], n, seed=seed
                        )[:, 0]
                        for i, n in enumerate(test_n)
                    ]
                ).T  # (n_samples, h)
                timings["M1a"] += time.time() - t_m1a
                m1a_q10 = np.quantile(m1a_samples, 0.10, axis=0)
                m1a_q90 = np.quantile(m1a_samples, 0.90, axis=0)
                seed_crps["M1a_chronos2"].append(float(np.mean(crps_counts(m1a_samples.T, test_a))))
                seed_mae["M1a_chronos2"].append(mae(c_mean * test_n, test_a))
                seed_cov["M1a_chronos2"].append(interval_coverage(m1a_q10, m1a_q90, test_a))

                # ---- M1b: TimesFM-2.5 ----
                t_m1b = time.time()
                f_levels, f_vals = m1b.forecast_quantiles(
                    train_rates, horizon=hh
                )
                f_mean = m1b.forecast_mean(train_rates, horizon=hh)
                m1b_samples = np.array([
                    rate_quantiles_to_count_samples(
                        f_levels, f_vals[:, [i]], n, seed=seed + 1
                    )[:, 0]
                    for i, n in enumerate(test_n)
                ]).T
                timings["M1b"] += time.time() - t_m1b
                m1b_q10 = np.quantile(m1b_samples, 0.10, axis=0)
                m1b_q90 = np.quantile(m1b_samples, 0.90, axis=0)
                m1b_crps = float(np.mean(crps_counts(m1b_samples.T, test_a)))
                seed_crps["M1b_timesfm25"].append(m1b_crps)
                seed_mae["M1b_timesfm25"].append(mae(f_mean * test_n, test_a))
                seed_cov["M1b_timesfm25"].append(interval_coverage(m1b_q10, m1b_q90, test_a))

                # ---- Oracle: true rates (noise floor) ----
                rng_o = np.random.default_rng(seed + 2)
                oracle_samples = np.stack([
                    rng_o.binomial(int(round(n)), float(np.clip(p, 0, 1)), size=N_COUNT_SAMPLES)
                    for n, p in zip(test_n, test_true, strict=True)
                ]).T
                seed_crps["oracle"].append(float(np.mean(crps_counts(oracle_samples.T, test_a))))
                seed_mae["oracle"].append(mae(np.median(oracle_samples, axis=0), test_a))

                # ---- M3 probe: Delta grid on M1a's median rate ----
                for delta in DELTA_GRID:
                    d_samples = np.stack([
                        hybrid.predict_counts(
                            float(np.clip(r, 0.01, 0.99)), delta,
                            int(round(n)), N_COUNT_SAMPLES, seed=seed,
                            concentration=m0_conc,  # comparable intervals
                        )
                        for r, n in zip(c_mean, test_n, strict=True)
                    ]).T
                    d_q10 = np.quantile(d_samples, 0.10, axis=0)
                    d_q90 = np.quantile(d_samples, 0.90, axis=0)
                    seed_m3_crps[delta].append(float(np.mean(crps_counts(d_samples.T, test_a))))
                    seed_m3_cov[delta].append(interval_coverage(d_q10, d_q90, test_a))

            # Pool origins -> one number per seed
            for m in seed_crps:
                results[scenario][m].append(float(np.mean(seed_crps[m])))
                mae_res[scenario][m].append(float(np.mean(seed_mae[m])))
            for m in seed_cov:
                cov_res[scenario][m].append(float(np.mean(seed_cov[m])))
            for d in DELTA_GRID:
                m3_crps[scenario][d].append(float(np.mean(seed_m3_crps[d])))
                m3_cov[scenario][d].append(float(np.mean(seed_m3_cov[d])))
            print(f"  {scenario} seed {seed}: done")

    m1a.unload()
    m1b.unload()

    # ---- Paired tests: per-seed CRPS differences vs M0 ----
    comparisons: dict[str, dict[str, dict]] = {}
    for scenario in SCENARIOS:
        comparisons[scenario] = {}
        m0_crps = np.array(results[scenario]["M0"])
        for m in ("M1a_chronos2", "M1b_timesfm25"):
            diff = np.array(results[scenario][m]) - m0_crps
            # Wilcoxon signed-rank (non-parametric, small n)
            try:
                w_stat, w_p = stats.wilcoxon(diff)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            comparisons[scenario][m] = {
                "mean_diff_crps": float(np.mean(diff)),
                "std_diff_crps": float(np.std(diff)),
                "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
                "wilcoxon_p": float(w_p) if w_p == w_p else None,
            }

    summary = {
        "mode": "mock" if use_mock else "real",
        "manifest_date": manifest.get("bootstrap_date"),
        "models": {
            m: manifest.get("models", {}).get(MODEL_KEYS[m], {})
            for m in ("M1a_chronos2", "M1b_timesfm25")
        },
        "design": {
            "scenarios": list(SCENARIOS),
            "n_seeds": n_seeds,
            "origins": list(ORIGINS),
            "horizon": HORIZON,
            "primary_metric": "CRPS",
        },
        "crps_mean_std": {
            s: {m: [float(np.mean(v)), float(np.std(v))] for m, v in models.items()}
            for s, models in results.items()
        },
        "mae_mean_std": {
            s: {m: [float(np.mean(v)), float(np.std(v))] for m, v in models.items()}
            for s, models in mae_res.items()
        },
        "coverage_mean": {
            s: {m: float(np.mean(v)) for m, v in models.items()}
            for s, models in cov_res.items()
        },
        "paired_vs_M0": comparisons,
        "m3_probe_crps": {
            s: {str(d): [float(np.mean(v)), float(np.std(v))] for d, v in ds.items()}
            for s, ds in m3_crps.items()
        },
        "m3_probe_coverage": {
            s: {str(d): float(np.mean(v)) for d, v in ds.items()}
            for s, ds in m3_cov.items()
        },
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }

    out = ROOT / "experiments" / "exp2_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out}")

    # Console summary
    print("\n" + "=" * 64)
    print("CRPS (mean ± std across seeds; lower is better)")
    print("=" * 64)
    for s in SCENARIOS:
        print(f"\n[{s}]")
        for m in ("oracle", "M0", "M1a_chronos2", "M1b_timesfm25"):
            mu, sd = summary["crps_mean_std"][s][m]
            print(f"  {m:15s} {mu:.3f} ± {sd:.3f}")
        for m in ("M1a_chronos2", "M1b_timesfm25"):
            c = comparisons[s][m]
            print(f"    {m} vs M0: Δ={c['mean_diff_crps']:+.3f}±{c['std_diff_crps']:.3f} "
                  f"p={c['wilcoxon_p']}")
    print("\n80% PI coverage (target 0.80):")
    for s in SCENARIOS:
        covs = summary["coverage_mean"][s]
        print(f"  [{s}] " + " ".join(f"{m}={v:.2f}" for m, v in covs.items()))
    print("\nM3 probe — CRPS vs Delta (drift scenario):")
    for d in DELTA_GRID:
        mu, sd = summary["m3_probe_crps"]["drift"][str(d)]
        cov = summary["m3_probe_coverage"]["drift"][str(d)]
        print(f"  Δ={d:+.2f}: CRPS {mu:.3f}±{sd:.3f}, cov {cov:.2f}")
    print(f"\nTimings (s): {summary['timings_s']}")
    return summary


if __name__ == "__main__":
    run_experiment()
