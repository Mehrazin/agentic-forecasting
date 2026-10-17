# TD-One: The Full Story — From Synthetic Benchmarks to Agentic Infrastructure

**Date:** 2026-10-17
**Status:** Capstone document — everything we've learned

---

## The Arc

We started with a simple question: can LLM agents improve mortgage attrition forecasting? We ended with something more valuable: a rigorous methodology for testing agentic forecasting, a catalog of honest negatives, and infrastructure for the next iteration.

This document tells the whole story.

---

## Part 1: The Classical Foundation (exp1–exp18b)

### What we built
- **M0**: Jeffreys Beta-Binomial baseline for mortgage switching rates
- **Simulator**: 10k×66mo synthetic mortgage panel with planted shocks
- **M1–M5 ladder**: Classical → ReAct → Hybrid → Regime → Router
- **TSFM integration**: Chronos-2 and TimesFM-2.5 with real weights
- **LogScaleCalibrator**: For near-zero SMM series (0.0003–0.003)

### What we found
- **exp18b**: TimesFM beats ETS by **12.2% CRPS** (p=0.076, n.s.) on 63-month real Freddie Mac SMM
- The 17.9% vs m0_w3 was against a strawman; ETS is the real baseline
- **R2-4**: TSFM win survives on M4 (-24.1% Chronos2, -22.6% TimesFM)
- **Classical macro**: No evidence (5 seeds, p=0.064 against)
- **TSFM macro**: Hurts (4/6 runs worse)
- **News → SMM**: No (p=0.85, clean negative)

### The honesty
We killed our own wins:
- "Macro is dead" → retracted (was a bug, then disproved properly)
- "30% TSFM win" → corrected to 12.2% vs proper baseline
- "93% few-shot" → retracted (DGP artifact)

Argon's verdict: "This is the exact level of rigor required for enterprise-grade."

---

## Part 2: Why Agents Failed (M2/M3/M4/M5)

### The diagnosis
Our agents were **output-layer wrappers** on a closed mathematical system:
- **M2 ReAct**: 3 tools, LLM worse at math than the equations
- **M3 Hybrid**: Guessing logit adjustments, no new information
- **M4 Regime**: News didn't affect the DGP
- **M5 Router**: Vibes-based, inverse-CRPS is better

**Root cause:** Agents had (a) no information advantage, (b) no capability advantage, (c) the wrong task.

### The lesson
Agents win when they have:
1. Information the model doesn't have
2. Capabilities the model doesn't have  
3. A task requiring both

We had none of the three.

---

## Part 3: The Rebuild (Phases 1–3)

### Phase 1: Event-Augmented Simulator
- 5 event types with causal mechanisms (OSFI, BoC, housing, lender, macro)
- Synthetic news feed: noisy, delayed, asymmetric
- PIT-safe by construction
- 84% of events news-leading (after Phase 3 tuning)
- 30/30 tests pass

### Phase 2: Agent Infrastructure
- **Sandbox**: Ported from AI2 (30s timeout, 1GB memory, no network, secret stripping)
- **10 tools**: Residual analysis, changepoints, stationarity, CV, model comparison, diagnostics, STL, event impact, scenarios
- **react_v2**: Code-executing ReAct loop
- 66 tests pass, ruff + mypy clean

### Phase 3: Reflect and Adjust
- **Decision policy**: James-Stein shrinkage, calibrated intervals, corroboration gating (11 tests)
- **exp24**: Regime-aware selection → ties best single (no alpha, but robust)
- **exp25**: h=3 horizon → news **significantly worse** (+25%, p=0.0013)

### The three honest negatives
1. **exp21**: Features don't transfer (-21.6%) — news lags the effect
2. **exp22**: v2 is a data scientist (+6.6% when active) but high variance — needs better policy
3. **exp23**: Diagnostic selection = random (p=0.80) — backward can't predict forward

---

## Part 4: What We've Learned

### About agentic forecasting
1. **Information asymmetry is non-negotiable.** Without it, agents are just expensive wrappers.
2. **Capability > vibes.** Code execution + statistical tools >> LLM guessing.
3. **Detection ≠ forecasting.** Finding a changepoint is easy; forecasting after it is hard.
4. **News is hard.** Lagging indicators are useless; leading indicators are rare; the link attenuates with horizon.

### About methodology
1. **Kill your wins.** The audit trail is the credibility.
2. **Proper baselines matter.** ETS > m0; strawmen produce fake wins.
3. **Multi-seed or it didn't happen.** Seed 0 was lucky; seeds 1-4 told the truth.
4. **Document the negatives.** They're more valuable than the positives.

### About synthetic data
Our simulator is mathematically sound but may not capture real-world complexity. The next step is validation on real datasets.

---

## Part 5: What's Next — Real Data Benchmarks

The infrastructure is built. The synthetic experiments gave us honest negatives with sharp diagnoses. Now we need to test on real data to see if our methodology is solid or if our synthetic data is the problem.

**Candidate benchmarks from the Vector bootcamp:**
- **SP500**: Multivariate, leak-safe covariates — test our model selection
- **Energy/Oil**: Daily, news-driven, has code-executing agents already — direct comparison
- **Food CPI**: Aggregate inflation — test our calibration
- **BoC Rate**: Discrete events — test our regime detection

Each is a benchmark for a different part of our methodology. If we fail on real data too, the methodology needs work. If we succeed on real data but failed on synthetic, our synthetic data needs calibration.

---

## The Takeaway

We didn't get the agentic win we wanted. We got something better: a rigorous framework for asking the question, infrastructure for the next attempt, and the credibility that comes from killing your own hypotheses.

The story isn't "agents work" or "agents don't work." It's "here's exactly when, why, and how they fail, and here's what we're building next."

That's science.
