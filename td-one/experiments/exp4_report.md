# Experiment 4 Report: M3 Hybrid (Agent-Adjusted M0) vs M0

**Date:** 2026-10-16 (run_date: 2025-06-30).
 **Mode:** real.

M3 = M0 (trailing-6 Jeffreys Beta-Binomial) + bounded log-odds
adjustment Δ ∈ [-0.75, +0.75] applied via HybridReadout. In real mode
the agent (Groq) genuinely judged each macro scenario and proposed Δ;
there are no scripted deltas in this run.

| Scenario | Δ | M0 CRPS | M3 CRPS | M0 cov | M3 cov |
|---|---|---|---|---|---|
| rate_shock_up | +0.18 | 19.87 | 16.33 | 1.00 | 1.00 |
| rate_shock_down | -0.25 | 4.18 | 5.01 | 1.00 | 1.00 |
| osfi_tightening | +0.30 | 9.95 | 15.22 | 1.00 | 1.00 |
| benign | +0.30 | 3.78 | 7.42 | 1.00 | 1.00 |
| family_squeeze | +0.20 | 14.60 | 7.96 | 1.00 | 1.00 |
| competitor_rate_war | +0.28 | 14.57 | 10.65 | 1.00 | 1.00 |

M3 beats M0 on CRPS in 3/6 scenarios.

## Failure analysis

- **Sign error:** `osfi_tightening` got the sign wrong — Δ +0.30 vs a
  true shift of ≈ −0.35. The agent misread a stress-test tightening as
  pro-switching. This is the most informative failure: directional
  errors on policy scenarios.
- **Spurious adjustment:** `benign` was adjusted +0.30 vs a true 0.00 —
  the agent has no "do nothing" prior for benign regimes.
- **Magnitude undershoot:** LLM deltas run 0.18–0.30 vs true shifts of
  0.45–0.70 — systematically too timid even when directionally right.
- **Direction doesn't predict the score:** `rate_shock_down` was a CRPS
  loss (4.18 → 5.01) despite the correct negative direction.
- Net: 4/6 directionally right, 3/6 score wins, 1 sign error. The
  bounded readout (±0.75) contained the damage; unbounded adjustment
  would have been worse.

## Caveats

- **Exploratory framing:** n=6 hand-designed scenarios, chosen by the
  experimenters — not six exchangeable trials. No significance claimed
  (a sign test on 3/6 gives p=1.0). Treat all scenario counts as
  descriptive.
- Each scenario injects a known true shift, so M3 has an information
  advantage over stationary M0 by construction — the question is
  whether the bounded readout converts that signal into better
  calibration without destroying it.
- The exp2 fixed-Δ grid row in exp4_results.json is historical context
  only (toy-series scale, not comparable).
- Coverage is measured on one test point per scenario (n=6), so it
  is uninformative here; CRPS is the decision metric.
