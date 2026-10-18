# TD-One Final Report: Multi-Dataset Benchmarking Synthesis

**Date:** 2026-10-17
**Status:** Living document — updated as experiments complete

---

## Executive Summary

We built agentic forecasting infrastructure (event simulator, sandbox, 10 tools, code-executing agents) and validated it across 5 real datasets. The methodology works when signal exists and correctly identifies when it doesn't.

**Key results:**
- **BoC rates:** Agent discovers inflation interaction autonomously; regime awareness +7.2pp
- **SMM:** 12.2% TSFM edge confirmed as ceiling; agent finds real features (+1.21%)
- **Food CPI:** Agent discriminates seasonality correctly across categories
- **M4:** Selection fails (-30%) — method problem, not data problem
- **WTI:** Random walk — correctly identified as impossible task

---

## 1. Infrastructure Built

### Event-Augmented Simulator
- 5 event types, PIT-safe news feed, 84% news-leading (after calibration)
- Calibrated: 9.1pp → 2.2pp effects, realistic decay, BoC-style interactions
- 33/33 tests pass

### Agent Infrastructure
- Sandbox (AI2 port): 30s timeout, 1GB memory, secret stripping
- 10 tools: residuals, changepoints, stationarity, CV, model comparison, diagnostics, STL, event impact, scenarios
- react_v2: Code-executing ReAct loop
- Decision policy: James-Stein shrinkage, calibrated intervals, corroboration gating (t>3.5, min 3pp)
- Tool call tracker for optimization
- 66+ tests pass

---

## 2. Benchmark Results

### BoC Rate Decisions ✅ VALIDATED
- **Task:** CUT/HOLD/HIKE, 143 meetings, 28-day lead
- **Baselines:** Climatology 79.1%, Rule 83.7%, MNLogit 83.5%
- **Agent:** Discovered inflation_gap × rate_momentum interaction autonomously (no hints)
- **Regime:** EASY/HARD split gives +7.2pp (know when to trust the model)
- **Lesson:** Simple beats sophisticated on n=143. Agent does real data science.

### Freddie Mac SMM ✅ CEILING CONFIRMED
- **Task:** 1-month SMM forecasting, 63 months
- **Best:** TimesFM 12.2% vs ETS (p=0.076, n.s.)
- **Agent:** Found real features (rate momentum, spreads) → +1.21% held-out
- **Events:** Real BoC events don't help (-13.04%) — already priced in
- **Lesson:** 12.2% is the classical ceiling. Agent confirms, doesn't overturn.

### Food CPI ✅ REAL DATA SCIENCE
- **Task:** Monthly inflation forecasting, 2010-2026
- **Agent:** Correctly identified seasonality (produce 0.375 vs headline 0.000)
- **Caveat:** h=6 CV fragile (regime change in training window)
- **Lesson:** Agent discriminates correctly across categories. CV needs regime awareness.

### M4 Monthly ❌ SELECTION FAILED
- **Task:** 100 series, 600 origins, 4 models
- **Result:** Selection -30% vs best single (worse than random)
- **Lesson:** Hand-written diagnostic rules don't work on synthetic OR real data. Need learned policy or drop track.

### WTI Crude ✅ CORRECTLY IMPOSSIBLE
- **Task:** 5/10/21-day WTI forecasting
- **Result:** ETS/ARIMA/naive identical. Agent correctly finds nothing 92% of time.
- **Lesson:** Don't benchmark on efficient markets. Method correctly identifies no-signal.

---

## 3. What We Learned

### About Agents
1. **Information asymmetry is non-negotiable.** BoC works (macro predicts decisions); WTI doesn't (efficient market).
2. **Agents are competent data scientists** when signal exists (BoC interaction, Food CPI seasonality).
3. **Detection ≠ forecasting.** Finding patterns is easier than exploiting them (exp22: +6.6% when active, but high variance).
4. **Selection needs rethinking.** Hand-rules fail everywhere. Learned policy or bust.

### About Methodology
1. **Kill your wins.** 17.9% → 12.2%, macro "positive" → "no evidence", selection "promising" → "dead".
2. **Multi-seed or it didn't happen.** Seed 0 lies; seeds 1-4 tell truth.
3. **Real data validates.** Synthetic taught failure modes; real data taught what's possible.
4. **Document negatives.** They're more valuable than positives.

### About Synthetic Data
- Was too cartoonish (9.1pp shifts). Now calibrated (2.2pp, decay, interactions).
- Useful for infrastructure testing, not for validating agent value.
- Real data is the ground truth.

---

## 4. Open Questions

1. **Selection:** Learned policy or drop? (M4 says hand-rules are dead)
2. **CV fragility:** How to make model selection regime-robust? (Food CPI h=6 lesson)
3. **News value:** Can live news beat priced-in expectations? (Untested — need news API)
4. **Scale:** Do results hold with more data? (BoC n=143 is small)

---

## 5. The Combination Trilogy (exp42-44)

After selection failed, we tested three combination approaches:

### exp42: FFORMA-lite (learned weights)
**Honest negative.** Learned combination doesn't beat equal weights (p=0.156). Forecast combination puzzle confirmed at our scale.

### exp43: Regime-probability router
**Partial positive.** First method to significantly beat equal weights (M4 0.1282 vs 0.1295, p=0.018). Constrained structural priors beat flexible learning at small-n.

### exp44: News-as-selector
**Partial positive.** News-as-feature is dead, but news-as-selector mechanism proven on synthetic (p=0.0068). Real-data blocked by sparsity — needs live news API.

**Lesson:** Structure beats flexibility. Regimes > weights. News selects, doesn't predict.

---

### exp45: Learned regimes (unsupervised)
**Honest negative.** Hand-written rules significantly beat unsupervised discovery (M4: 0.1282 vs 0.1302, p=0.00046). The prior encodes domain knowledge that clustering can't find.

### exp46: News dense (definitive)
**Definitive negative.** With 86% BoC coverage (123/143 meetings), news still doesn't help regime classification. Sparsity was never the constraint. **News track closed for monthly forecasting.**

**Final verdict on news:** Mechanism works on synthetic, fails on real data. Real markets don't give news the monopoly on latent-state info that the DGP did. Reallocate to regime router.

---

## 6. Next Steps

1. **Literature survey:** How do others do agentic forecasting? What selection methods work?
2. **Synthetic re-runs:** Test exp21-style discovery on calibrated DGP
3. **Final report:** Consolidate everything for presentation

---

*This document grows as we learn. The honesty is the point.*
