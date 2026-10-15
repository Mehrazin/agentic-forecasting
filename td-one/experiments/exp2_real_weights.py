"""Experiment 2: M0 vs M1 with REAL model weights.

Compares on synthetic mortgage-renewal data:
  - M0:  Historical baseline (Jeffreys Beta-Binomial)
  - M1a: Chronos-2 (amazon/chronos-2, real weights)
  - M1b: TimesFM-2.5 (google/timesfm-2.5-200m-pytorch, real weights)

Design notes:
  - Models are loaded ONE AT A TIME and unloaded after use (RAM constraint).
  - Uses weights from scripts/bootstrap_models.py (models/MANIFEST.json).
  - Set TD_ONE_MOCK=1 to run with mocks instead (CI / no-download path).

Usage:
  cd td-one
  uv run --with huggingface_hub --with torch --with chronos-forecasting \\
      --with timesfm python experiments/exp2_real_weights.py

RAM: ~1.5GB peak per model on CPU. Run one model at a time (default).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from td_one.evaluation.metrics import mae  # noqa: E402
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.models.tsfm import Chronos2Wrapper, TimesFMWrapper  # noqa: E402


def load_manifest() -> dict:
    """Read models/MANIFEST.json from bootstrap."""
    mp = ROOT / "models" / "MANIFEST.json"
    if not mp.exists():
        raise FileNotFoundError(
            f"{mp} not found. Run scripts/bootstrap_models.py first."
        )
    return json.loads(mp.read_text())


def synthetic_switch_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate (n_renewals, n_switches, test) arrays for 18 periods."""
    rng = np.random.default_rng(seed)
    n_periods = 18
    n_renewals = rng.integers(80, 120, size=n_periods).astype(float)
    # True rate drifts slightly upward (mild non-stationarity)
    true_rates = 0.16 + 0.004 * np.arange(n_periods)
    n_switches = np.array(
        [rng.binomial(int(n), p) for n, p in zip(n_renewals, true_rates)],
        dtype=float,
    )
    return n_renewals, n_switches, true_rates


def run_m0(n_renewals: np.ndarray, n_switches: np.ndarray) -> float:
    """M0: historical Beta-Binomial baseline. Returns MAE."""
    m0 = HistoricalBaseline()
    m0.fit(n_renewals[:12], n_switches[:12])
    preds = [
        m0.predict_quantiles(n_upcoming=int(n))[1]  # q50
        for n in n_renewals[12:18]
    ]
    return mae(np.array(preds), n_switches[12:18])


def run_m1a_chronos(
    rates: np.ndarray, n_renewals: np.ndarray, weights_dir: str | None, use_mock: bool
) -> float:
    """M1a: Chronos-2 zero-shot on historical rates. Returns MAE."""
    chronos = Chronos2Wrapper(use_mock=use_mock)
    if not use_mock:
        chronos._load(weights_dir)
    forecast = chronos.forecast(rates, horizon=6, n_samples=100)
    mean_rates = np.clip(forecast.mean(axis=0), 0.0, 1.0)
    return mean_rates, chronos


def run_m1b_timesfm(
    rates: np.ndarray, n_renewals: np.ndarray, weights_dir: str | None, use_mock: bool
) -> float:
    """M1b: TimesFM-2.5 zero-shot on historical rates. Returns (mean, wrapper)."""
    tfm = TimesFMWrapper(use_mock=use_mock)
    mean, _ = tfm.forecast_quantiles(rates, horizon=6, weights_dir=weights_dir)
    return np.clip(mean, 0.0, 1.0), tfm


def main() -> int:
    use_mock = os.environ.get("TD_ONE_MOCK", "0") == "1"
    manifest = {} if use_mock else load_manifest()

    print("=" * 60)
    print("Experiment 2: M0 vs M1 (real weights)" if not use_mock else "Experiment 2 (MOCK mode)")
    print("=" * 60)

    n_renewals, n_switches, true_rates = synthetic_switch_data()
    rates = np.clip(n_switches[:12] / n_renewals[:12], 0.01, 0.99)
    test_n = n_renewals[12:18]
    test_y = n_switches[12:18]

    results: dict[str, float] = {}

    # M0
    print("\n[M0] Historical baseline...")
    results["M0"] = run_m0(n_renewals, n_switches)
    print(f"  MAE: {results['M0']:.2f}")

    # M1a — Chronos-2 (load, forecast, unload)
    print("\n[M1a] Chronos-2...")
    wdir_a = None if use_mock else str(ROOT / manifest["models"]["chronos2"]["local_path"])
    mean_a, wrapper_a = run_m1a_chronos(rates, test_n, wdir_a, use_mock)
    results["M1a_chronos2"] = mae(mean_a * test_n, test_y)
    print(f"  MAE: {results['M1a_chronos2']:.2f}")
    wrapper_a.unload()
    del wrapper_a

    # M1b — TimesFM-2.5 (load, forecast, unload)
    print("\n[M1b] TimesFM-2.5...")
    wdir_b = None if use_mock else str(ROOT / manifest["models"]["timesfm25"]["local_path"])
    mean_b, wrapper_b = run_m1b_timesfm(rates, test_n, wdir_b, use_mock)
    results["M1b_timesfm25"] = mae(mean_b * test_n, test_y)
    print(f"  MAE: {results['M1b_timesfm25']:.2f}")
    wrapper_b.unload()
    del wrapper_b

    print("\n" + "=" * 60)
    print("RESULTS (MAE, lower is better)")
    print("=" * 60)
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        tag = " <- best" if v == min(results.values()) else ""
        print(f"  {k:15s} {v:.2f}{tag}")

    # Save results JSON
    out = ROOT / "experiments" / "exp2_results.json"
    out.write_text(json.dumps({
        "mode": "mock" if use_mock else "real",
        "manifest_date": manifest.get("bootstrap_date"),
        "models": {
            k: manifest.get("models", {}).get(
                "chronos2" if "chronos" in k else "timesfm25" if "timesfm" in k else k, {}
            ) for k in results if k != "M0"
        },
        "mae": results,
        "n_test_periods": 6,
        "seed": 42,
    }, indent=2))
    print(f"\nResults saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
