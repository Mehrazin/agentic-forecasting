"""Pre-registered evaluation scenarios for exp2.

Three scenarios, fixed 18-period budget each, defined BEFORE seeing results
(per the scientific-rigor audit — the DGP must be justified from the domain,
not tuned to flatter a method).

- stationary: constant 18% switch rate (M0's home turf)
- drift: linear 0.16 -> 0.228 drift (TSFM's home turf; the old exp2 DGP)
- regime_shift: discrete jump 0.15 -> 0.25 at t=9 (BoC-decision-like event;
  the realistic mortgage-renewal case)

All scenarios share realistic structure:
- Seasonal renewal volume: N_t = base * (1 + 0.25*sin(2*pi*t/12)) + AR(1)
- Overdispersion: switches ~ Beta-Binomial (cohort clustering), not Binomial
- AR(1) noise on the logit rate (persistence)

A method that only wins on its home-turf DGP has demonstrated nothing.
"""

from __future__ import annotations

import numpy as np

N_PERIODS = 18
BASE_N = 100
SEASONAL_AMP = 0.25
AR_RHO = 0.5
OVERDISPERSION = 30.0  # Beta concentration; lower = more overdispersion


def _scenario_rate(scenario: str, t: np.ndarray) -> np.ndarray:
    """Return the true switch rate per period for a scenario."""
    if scenario == "stationary":
        return np.full_like(t, 0.18, dtype=float)
    if scenario == "drift":
        return 0.16 + 0.004 * t
    if scenario == "regime_shift":
        return np.where(t < 9, 0.15, 0.25)
    raise ValueError(f"unknown scenario: {scenario}")


def generate_scenario(
    scenario: str,
    seed: int,
    n_periods: int = N_PERIODS,
) -> dict[str, np.ndarray]:
    """Generate one (N, A) series for a scenario.

    Args:
        scenario: One of "stationary", "drift", "regime_shift".
        seed: Random seed.
        n_periods: Series length.

    Returns:
        Dict with n_renewals, n_switches, true_rates.

    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods, dtype=float)

    # Seasonal volume + AR(1) noise
    seasonal = 1.0 + SEASONAL_AMP * np.sin(2 * np.pi * t / 12)
    ar = np.zeros(n_periods)
    for i in range(1, n_periods):
        ar[i] = AR_RHO * ar[i - 1] + rng.normal(0, 8)
    n_renewals = np.maximum(20, BASE_N * seasonal + ar)

    # True rate + AR(1) logit noise
    base_rate = _scenario_rate(scenario, t)
    logit_noise = np.zeros(n_periods)
    for i in range(1, n_periods):
        logit_noise[i] = 0.5 * logit_noise[i - 1] + rng.normal(0, 0.08)
    logit_p = np.log(base_rate / (1 - base_rate)) + logit_noise
    true_rates = 1 / (1 + np.exp(-logit_p))

    # Beta-Binomial draws (overdispersed vs Binomial)
    alpha = true_rates * OVERDISPERSION
    beta = (1 - true_rates) * OVERDISPERSION
    p_drawn = rng.beta(alpha, beta)
    n_switches = rng.binomial(np.round(n_renewals).astype(int), p_drawn).astype(float)

    return {
        "n_renewals": n_renewals,
        "n_switches": n_switches,
        "true_rates": true_rates,
    }


SCENARIOS = ("stationary", "drift", "regime_shift")
