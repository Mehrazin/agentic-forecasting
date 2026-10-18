# exp47: Proposed Method End-to-End — Report

**Date:** 2026-10-18
**Verdict: POSITIVE.** The combined proposed method significantly beats equal
weights, best single, AND the exp43 router alone on M4. The gates + conformal
calibration add real value beyond the router.

---

## 1. What was built

`src/td_one/agent/proposed_method.py` — the TD-One flagship, integrating every
positive-lift component (see `docs/POSITIVE_LIFT_AUDIT.md`):

| # | Component | Source | Role |
|---|---|---|---|
| 1 | Regime-probability router | exp43 | Backbone: hand regimes → LR → per-regime experts → soft combine |
| 2 | Trust flag (EASY/HARD) | exp32 | Report forecast trustworthiness |
| 3 | "Do nothing" gate | exp6 + decision_policy | Corroboration gating: fall back to global best when router uncertain + evidence weak |
| 4 | Conformal calibration | exp5/exp10 | Volatility-floored quantile widths |
| 5 | News modulation (gated, off) | exp44 | Adjust P(regime) only when news dense |
| 6 | LLM sensor (gated, off) | exp11 | Regime reading with abstention (conf ≥ 0.75) |

Kill list honored: no learned selection (exp23/37/41), no learned combination
(exp42), no unsupervised regimes (exp45), no news-as-feature (exp19/21/25/46).

`ForecastOutput` returns quantiles + regime_probs + expert_per_regime +
active_expert + trust + explanation + gated flag — fully interpretable.

## 2. Results (blocked CV, paired t-tests)

| Dataset | n | Proposed | Equal | Best single | Router-only | vs equal (p) | vs best (p) | vs router (p) |
|---|---|---|---|---|---|---|---|---|
| M4 | 600 | **0.1451** | 0.1507 | 0.1505 | 0.1490 | **6.5e-09** | **7.8e-11** | **6.7e-10** |
| Synthetic | 480 | 0.1036 | 0.1068 | 0.1028 | 0.1028 | 0.072 | 0.60 | 0.28 |
| SMM | 27 | 0.1389 | 0.1424 | 0.1377 | 0.1377 | 0.66 | 0.61 | 0.61 |
| Food CPI | 24 | 0.0141 | 0.0176 | 0.0143 | 0.0143 | 3.1e-18* | 0.005* | 0.005* |

\* Food CPI: 24 correlated origins from one series — p-values overstated
(same caveat as exp43).

### The key result: proposed beats router-only on M4

Proposed (0.1451) vs exp43-router-alone (0.1490), **p=6.7e-10**. The
difference is the gates + conformal calibration:

- **Gate** fired on 5% of M4 origins (uncertain router + weak evidence →
  fall back to global best single instead of the soft combination).
- **Conformal** volatility floor prevents degenerate intervals: quantile
  half-width floored at 0.5 × historical volatility (exp5 lesson).
- **Trust flag:** 91% of M4 origins rated EASY (top regime P ≥ 0.7).

This is the first evidence that the *combination of components* beats the
best individual component. The system is more than the sum of its parts.

### Honest negatives

- **SMM:** experts collapse to naive in every fold (same as exp43) — no
  regime value-add. Proposed = best single. n=27 (fewer valid origins
  with the 36-month train window).
- **Synthetic:** proposed ties best single (naive). The DGP's clean
  regimes don't need the gates.
- **Food CPI:** same expert-collapse pattern; p-values not trustworthy.

## 3. Interpretation

**Why the full system beats the router alone:** exp43's router is confident
(top P ≈ 0.9), so the soft combination rarely differs from hard selection.
The gains come from the *tails*: the 5% of origins where the router is
uncertain are exactly where the soft combination is most fragile — and the
gate correctly falls back to the global best there. The conformal floor
fixes the interval degeneracy that exp22 diagnosed (3-point sd → width
0.03 vs m0's 0.23).

**The proposed method is now the program's best combination method:**
it beats equal weights (exp42's target), beats best single, and beats the
exp43 router that was already the first to beat equal weights.

## 4. What remains

1. **TSFM expert:** add calibrated TimesFM/Chronos to the expert zoo
   (exp14/exp18b) — currently only classical models.
2. **News/LLM gates:** implemented but off by default (need dense news /
   Groq). exp44's mechanism + exp11's sensor are wired in.
3. **BoC:** discrete-task variant (EASY/HARD → rule/climatology) not yet
   integrated into `ProposedForecaster` (continuous-only for now).
4. **Learned regimes:** dead per exp45 — hand rules stay.

## 5. Reproducibility

```bash
cd td-one && .venv/bin/python experiments/exp47_proposed_method.py [--quick]
```
Full run: ~90s. Deterministic (fixed seeds). Results: `exp47_results.json`.

## 6. Files

- `src/td_one/agent/proposed_method.py` — the flagship module (ruff clean)
- `experiments/exp47_proposed_method.py` — end-to-end test (ruff clean)
- `experiments/exp47_results.json` — all numbers
- `experiments/exp47_report.md` — this file
- `docs/POSITIVE_LIFT_AUDIT.md` — the complete exp1–exp46 audit
- `docs/PROPOSED_METHOD.md` — architecture doc (to be written)
