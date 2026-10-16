# Experiment 4 Report: M3 Hybrid (Agent-Adjusted M0) vs M0

**Date:** 2026-10-15 (run_date: 2025-06-30).
 **Mode:** mock.

M3 = M0 (trailing-6 Jeffreys Beta-Binomial) + bounded log-odds
adjustment Δ ∈ [-0.75, +0.75] applied via HybridReadout. The agent
proposes Δ per macro scenario; in mock mode the deltas are scripted
(directionally right, imperfect by design).

| Scenario | Δ | M0 CRPS | M3 CRPS | M0 cov | M3 cov |
|---|---|---|---|---|---|
| rate_shock_up | +0.50 | 19.87 | 9.40 | 1.00 | 1.00 |
| rate_shock_down | -0.40 | 4.18 | 4.30 | 1.00 | 1.00 |
| osfi_tightening | -0.25 | 9.95 | 8.26 | 1.00 | 1.00 |
| benign | +0.00 | 3.78 | 3.78 | 1.00 | 1.00 |
| family_squeeze | +0.35 | 14.60 | 7.65 | 1.00 | 1.00 |
| competitor_rate_war | +0.60 | 14.57 | 8.90 | 1.00 | 1.00 |

M3 beats M0 on CRPS in 4/6 scenarios.

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
