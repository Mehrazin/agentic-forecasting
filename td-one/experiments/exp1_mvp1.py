"""Experiment 1: M0 vs M1 on synthetic data.

Compares:
- M0: Historical baseline (Beta-Binomial)
- M1a: Chronos-2 (mock)
- M1b: TimesFM-2.5 (mock)

On a small synthetic dataset. Validates the pipeline end-to-end.
"""

from __future__ import annotations

import numpy as np

from td_one.data_gen.simulator import MortgageSimulator
from td_one.evaluation.metrics import mae
from td_one.models.baseline import HistoricalBaseline
from td_one.models.tsfm import Chronos2Wrapper, TimesFMWrapper


def run_exp1() -> dict[str, float]:
    """Run M0 vs M1 comparison.

    Returns:
        Dict of model_id -> MAE.

    """
    print("Generating synthetic data (500 customers, 24 months)...")
    sim = MortgageSimulator(n_customers=500, n_months=24, seed=42)
    records = sim.generate_history()
    print(f"  Generated {len(records)} records")

    # Simulate segment-level (N, A) history
    # For demo: use fixed synthetic switching data
    rng = np.random.default_rng(42)
    n_periods = 18
    true_rate = 0.18
    n_renewals = rng.integers(80, 120, size=n_periods)
    n_switches = rng.binomial(n_renewals, true_rate)

    results: dict[str, float] = {}

    # M0: Historical baseline
    print("\nM0: Historical baseline...")
    m0 = HistoricalBaseline()
    m0.fit(n_renewals[:12], n_switches[:12])
    # Predict next 6 periods (point forecast = expected count)
    m0_preds = []
    for n in n_renewals[12:18]:
        q10, q50, q90 = m0.predict_quantiles(n_upcoming=int(n))
        m0_preds.append(q50)
    m0_mae = mae(np.array(m0_preds), n_switches[12:18].astype(float))
    results["M0"] = m0_mae
    print(f"  MAE: {m0_mae:.2f}")

    # M1a: Chronos-2 (mock)
    print("\nM1a: Chronos-2 (mock)...")
    chronos = Chronos2Wrapper(use_mock=True)
    # Logit-transformed historical rates as input
    rates = n_switches[:12] / n_renewals[:12]
    rates = np.clip(rates, 0.01, 0.99)
    # Mock expects rates in [0, 1], not logits
    forecast = chronos.forecast(rates, horizon=6, n_samples=100)
    # Mock output is already in rate space
    mean_rates = forecast.mean(axis=0)
    m1a_preds = mean_rates * n_renewals[12:18]
    m1a_mae = mae(m1a_preds, n_switches[12:18].astype(float))
    results["M1a"] = m1a_mae
    print(f"  MAE: {m1a_mae:.2f}")

    # M1b: TimesFM-2.5 (mock)
    print("\nM1b: TimesFM-2.5 (mock)...")
    timesfm = TimesFMWrapper(use_mock=True)
    mean, _ = timesfm.forecast_quantiles(rates, horizon=6)
    # Mean is in rate space (mock returns rates directly)
    m1b_preds = mean * n_renewals[12:18]
    m1b_mae = mae(m1b_preds, n_switches[12:18].astype(float))
    results["M1b"] = m1b_mae
    print(f"  MAE: {m1b_mae:.2f}")

    print("\n=== Results ===")
    for model_id, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {model_id}: MAE = {score:.2f}")

    return results


if __name__ == "__main__":
    run_exp1()
