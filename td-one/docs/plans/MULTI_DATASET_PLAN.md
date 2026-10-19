# Multi-Dataset Benchmarking Plan: Real Data 0 to 100

**Date:** 2026-10-17
**Status:** Approved — executing
**Strategy:** B primary (real data), A in background (calibrate synthetic), C nudge (tighter FP control)

---

## Datasets

### 1. BoC Rate Decisions (primary)
- **Why:** Agent discovered real interaction autonomously. Most promising.
- **Task:** CUT/HOLD/HIKE, 28-day lead, 119 meetings (corrected 2026-10-19)
- **Experiments:**
  - exp30: Full multinomial logistic with all features (including 2yr yield if we can get it)
  - exp31: Agent autonomous discovery on full feature set (can it beat our manual diagnosis?)
  - exp32: Regime-aware BoC (does event density predict decision difficulty?)

### 2. Freddie Mac SMM (home turf)
- **Why:** We have 63 months, got 12.2% vs ETS. Real mortgage data.
- **Task:** 1-month ahead SMM forecasting
- **Experiments:**
  - exp33: Agent feature discovery on real SMM (does it find anything our manual analysis missed?)
  - exp34: Event-augmented SMM (can we label real events in 2021-2026 that moved SMM?)

### 3. Food CPI (new)
- **Why:** Inflation is persistent and forecastable. Tests calibration on macro data.
- **Task:** Monthly food CPI forecasting
- **Experiments:**
  - exp35: Baseline (ETS vs our agent on food CPI)
  - exp36: Does the agent find seasonal patterns autonomously?

### 4. M4 Monthly (validation)
- **Why:** We already have R2-4 results. Good sanity check.
- **Task:** Replicate with agent tools
- **Experiments:**
  - exp37: Agent model selection on M4 (does diagnostic selection work when signal is clear?)

## Background: Synthetic Calibration (A) — DONE 2026-10-17

Calibrated (see [docs/SYNTHETIC_CALIBRATION.md](SYNTHETIC_CALIBRATION.md)):
- Reduce event effects: 9.1pp → 2-3pp ✅ (mean nominal peak 2.19pp)
- Faster market adaptation: priced in within days not months ✅ (2.5-month
  half-life decay on all generated events)
- Add subtle interactions (like BoC's inflation×momentum) ✅ (`amplify_when`:
  policy shock × inflation_gap>2.0, ×1.8, on 25% of policy shocks)
- Document in `docs/SYNTHETIC_CALIBRATION.md` ✅

## Nudge: Tighter FP Control (C)

Small change now, revisit later if needed:
- In `decision_policy.py`: raise corroboration threshold from t>3.0 to t>3.5
- Add minimum effect size: |shift| ≥ 3pp (was 2pp)
- Log FP rate in tool tracker for monitoring

---

## Execution Order

1. **BoC exp30-32** (highest value)
2. **SMM exp33-34** (home turf)
3. **Food CPI exp35-36** (new domain)
4. **M4 exp37** (validation)
5. Background: synthetic calibration

Each experiment: survey → plan → execute → analyse → reflect → document.

---

## Success Criteria

- **BoC:** Agent beats 72.2% rule OR discovers new interaction
- **SMM:** Agent finds feature we missed OR confirms 12.2% is the ceiling
- **Food CPI:** Agent matches or beats ETS
- **M4:** Agent selection beats single-best

If 3/4 succeed: methodology is solid, synthetic was the problem.
If 0/4 succeed: methodology needs fundamental rethink.

---

## Pre-Final-Report Gate: Tool Call Audit (2026-10-18)

**Requirement (standing):** Before the final report, log and review ALL model
tool calls and code executions, then optimize the tools.

- Tracker: `src/td_one/agent/tool_tracker.py` (singleton; records name,
  duration, input hash, truncated output, success/failure).
- Review: aggregate call counts, durations, failure rates per tool across
  all experiments. Identify: (a) unused tools to prune, (b) slow tools to
  optimize, (c) failure modes to fix, (d) redundant calls to batch.
- Deliverable: `docs/TOOL_AUDIT.md` with per-tool stats and the optimization
  decisions, written before `docs/FINAL_SYNTHESIS.md` is finalized.
- Reasoning level: use the highest LLM reasoning level wherever the provider
  API supports it (Groq: check `reasoning_effort` / model options at call time).
