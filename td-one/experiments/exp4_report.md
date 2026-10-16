# Experiment 4 Report: M3 Hybrid (Agent-Adjusted M0) vs M0

**Date:** 2026-10-16 (run_date: 2025-06-30).
 **Mode:** real.

M3 = M0 (trailing-6 Jeffreys Beta-Binomial) + bounded log-odds
adjustment Δ ∈ [-0.75, +0.75] applied via HybridReadout. The agent
proposes Δ per macro scenario; in mock mode the deltas are scripted
(directionally right, imperfect by design).

| Scenario | Δ | M0 CRPS | M3 CRPS | M0 cov | M3 cov |
|---|---|---|---|---|---|
| rate_shock_up | +0.18 | 19.87 | 16.33 | 1.00 | 1.00 |
| rate_shock_down | -0.25 | 4.18 | 5.01 | 1.00 | 1.00 |
| osfi_tightening | +0.30 | 9.95 | 15.22 | 1.00 | 1.00 |
| benign | +0.30 | 3.78 | 7.42 | 1.00 | 1.00 |
| family_squeeze | +0.20 | 14.60 | 7.96 | 1.00 | 1.00 |
| competitor_rate_war | +0.28 | 14.57 | 10.65 | 1.00 | 1.00 |

M3 beats M0 on CRPS in 3/6 scenarios.

## Caveats

- Mock-mode deltas are scripted, not LLM judgments; the comparison
  measures whether a *directionally correct* agent adjustment helps,
  not whether a real LLM would produce one.
- Each scenario injects a known true shift, so M3 has an information
  advantage over stationary M0 by construction — the question is
  whether the bounded readout converts that signal into better
  calibration without destroying it.
- The exp2 fixed-Δ grid row in exp4_results.json is historical context
  only (toy-series scale, not comparable).
- Coverage is measured on one test point per scenario (n=6), so it
  is uninformative here; CRPS is the decision metric.
