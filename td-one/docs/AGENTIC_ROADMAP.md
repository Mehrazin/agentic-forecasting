# TD-One Agentic Roadmap: From Output Wrappers to True Agents

**Date:** 2026-10-17
**Status:** Planning — approved by Vandad
**Context:** Our M2/M3/M4/M5 agents failed because they were output-layer wrappers on a closed mathematical system. This roadmap redesigns the agentic track from first principles.

---

## Part 1: Diagnosis (What We Learned)

### Why the first agentic wave failed

Our agents (M2 ReAct, M3 hybrid, M4 regime reader, M5 router) all shared three fatal design flaws:

1. **No information advantage.** The simulator is deterministic equations. Agents re-process the same numbers through an LLM — worse at math than the equations themselves.
2. **No capability advantage.** M2 had 3 tools (get data, get news, run forecast). M3/M4/M5 had zero tools. They couldn't analyze, plot, test hypotheses, or write code.
3. **Wrong task.** They adjusted final outputs. None touched middle parameters, engineered features, diagnosed failures, or selected models.

**The lesson:** Agents win when they have (a) information the model doesn't, (b) capabilities the model doesn't, (c) a task requiring both. We had none of the three.

This is a publishable negative result. It belongs in our story.

---

## Part 2: Event-Augmented Simulator

### The core idea

Add a **news/event stream to the DGP that actually affects outcomes**. Currently, "shocks" are conditional branches (`if origination_year == 2021`). Instead, we need exogenous events that:

- Arrive at random times (not tied to loan characteristics)
- Affect outcomes through causal channels (not just additive noise)
- Are observable through a news feed (with realistic noise/delay)
- Create information asymmetry: agents with news access see them; pure time-series models don't

### Design: `EventAugmentedSimulator`

**Event types** (each with a causal mechanism):

1. **Policy shocks.** OSFI announces stress-test changes. Affects `p(switch)` for uninsured borrowers with a 2-month implementation lag. News reports the announcement; the effect is delayed and heterogeneous.

2. **Rate regime shifts.** BoC makes surprise decisions (not on the scheduled calendar). Affects `rate_spread` for all borrowers resetting within 3 months. News reports the decision within days; the transmission to mortgage rates takes 2-4 weeks.

3. **Housing market disruptions.** Regional price drops (e.g., Toronto condo correction). Affects LTV ratios, which affect `p(switch)` through the equity channel. News reports with a 1-month delay and survivorship bias (only big drops get covered).

4. **Lender behavior shifts.** A major bank launches an aggressive retention campaign. Affects `p(switch)` for that bank's borrowers only. News may or may not cover it (asymmetric coverage = realistic).

5. **Macro surprises.** Unemployment spike in a specific sector/region. Affects `payment_shock` through income loss. News covers with delay; the DGP effect is immediate.

**Key properties:**
- Events are **PIT-safe**: the simulator generates the event timeline first, then the news feed references only events with `event_date <= article_date`.
- Events have **heterogeneous effects**: not every borrower is affected equally. This creates the need for agentic reasoning about *who* is affected.
- News coverage is **noisy**: some events get 20 articles, some get 0. Some articles are wrong about magnitude. Agents must triangulate.
- The **baseline models don't see events**: M0, ETS, ARIMA, and TSFMs only see the time series. The agent sees time series + news. This is the information asymmetry that justifies agentic methods.

### Implementation plan

- New module: `src/td_one/data_gen/events.py` — event timeline generator
- New module: `src/td_one/data_gen/news_synthesis.py` — synthetic news article generator (template-based, PIT-safe)
- Modify: `src/td_one/data_gen/simulator.py` — accept event timeline, apply causal effects
- New: `experiments/exp20_event_augmented.py` — validation that events create a detectable signal
- Success criterion: an oracle agent (sees true events) beats ETS by >15%. A news-reading agent beats ETS by >5%.

---

## Part 3: Agent Feature Discovery

### The idea

Instead of the LLM adjusting outputs, let it **analyze residuals and propose new features**.

**Loop:**
1. Run baseline model (e.g., ETS) on training data
2. Agent examines residuals: where is the model systematically wrong?
3. Agent proposes a feature: "residuals are high when X" → creates `feature_X`
4. Agent tests the feature: does it improve CV score?
5. If yes, keep it. If no, discard and try another.

**This is what human data scientists do.** We're automating the feature engineering loop.

### Tools needed
- Residual analysis (plot residuals vs. candidate features)
- Feature importance (permutation, SHAP)
- Cross-validation harness (PIT-safe, of course)
- Statistical tests (is this feature significant?)

### Success criterion
Agent-discovered features improve over hand-engineered features on held-out data. Even 2-3% would validate the approach.

---

## Part 4: Agent Model Selection & Diagnostics

### Model selection

Per-segment, per-origin: the agent examines data characteristics (volatility, trend strength, seasonality, structural breaks) and selects the best model from a zoo (M0, ETS, ARIMA, Chronos2, TimesFM, etc.).

This is NOT M5 (LLM vibes). This is: agent runs diagnostics, computes meta-features, and uses a learned selection policy.

### Diagnostics

When a forecast is wrong, the agent investigates:
- Was it a regime change? (changepoint detection)
- Was it an outlier? (robust statistics)
- Was the model misspecified? (residual diagnostics)
- Should we trust this forecast? (uncertainty quantification)

Output is not just a number — it's a number + explanation + confidence.

---

## Part 5: Code Execution & Sandboxing

### The gap

The bootcamp's energy/oil example has code-executing agents. Ours don't. This is the single biggest capability gap.

### What we need

Agents that can:
- Run Python (pandas, numpy, matplotlib, statsmodels, sklearn)
- Plot data and inspect it visually (multimodal)
- Test hypotheses iteratively
- Build and evaluate models

### Sandboxing (port from AI2)

The AI2 repo has a mature sandbox:
- `subprocess`-based isolation with resource limits
- Docker option for stronger isolation
- Path traversal guards
- Byte caps on outputs
- Timeout enforcement

We need to port the core of this to TD-One. Specifically:
- `src/td_one/agent/sandbox.py` — port from AI2's sandbox module
- Resource limits: 30s timeout, 512MB memory, no network, read-only filesystem except /tmp
- Audit logging: every code execution logged with input/output

### Safety considerations
- No network access in sandbox
- No filesystem writes outside /tmp
- LLM-generated code reviewed by a critic before execution (optional, for high-stakes)
- All tool outputs sanitized before returning to LLM (prompt injection defense)

---

## Part 6: 10 Proposed General Tools

These are reusable tools the agent can call. Each is a Python function with a clear contract.

1. **`analyze_residuals(y_true, y_pred, features)`** — Plots residuals vs. each feature, runs Breusch-Pagan for heteroskedasticity, reports top 3 features correlated with residual magnitude. *Why: the starting point for feature discovery.*

2. **`detect_changepoints(series, min_distance=6)`** — Runs PELT changepoint detection, returns changepoint locations + segment means. *Why: regime changes are where models break.*

3. **`feature_importance(model, X, y)`** — Permutation importance + SHAP values. Returns ranked feature list. *Why: understand what the model is actually using.*

4. **`cross_validate(model_fn, data, n_splits=5)`** — PIT-safe time-series CV. Returns mean ± SE of CRPS. *Why: honest evaluation of any proposed change.*

5. **`compare_models(model_fns, data)`** — Runs multiple models on the same CV splits, returns pairwise comparison with Diebold-Mariano tests. *Why: systematic model selection.*

6. **`plot_diagnostics(y_true, y_pred, dates)`** — Generates 4-panel diagnostic plot (residuals vs time, QQ plot, ACF of residuals, predicted vs actual). Returns image. *Why: visual inspection catches what statistics miss.*

7. **`test_stationarity(series)`** — ADF + KPSS tests. Returns whether series is stationary, trend-stationary, or neither. *Why: determines if differencing is needed.*

8. **`decompose_series(series, period=12)`** — STL decomposition into trend/seasonal/residual. Returns components + strength metrics. *Why: understand the signal structure.*

9. **`backtest_event_impact(events, series, window=6)`** — For each event, computes the average treatment effect on the series in the post-event window vs. a synthetic control. *Why: quantify whether news events actually move the needle.*

10. **`generate_synthetic_scenario(base_series, shock_type, magnitude)`** — Applies a hypothetical shock to the series and forecasts the impact. *Why: counterfactual reasoning ("what if rates rise 100bp?").*

Each tool has:
- Strict input validation (Pydantic)
- PIT safety checks (no future data)
- Clear error messages
- Unit tests

---

## Part 7: Experiment Plan

### exp20: Event-augmented simulator validation
- Build the event-augmented DGP
- Verify: oracle (sees true events) >> ETS; news-agent > ETS
- If news-agent doesn't beat ETS, the events are too weak or the news is too noisy — iterate

### exp21: Agent feature discovery
- Baseline: ETS with hand-engineered features
- Agent: discovers features from residuals
- Metric: CV CRPS improvement on held-out data

### exp22: Code-executing ReAct (M2 v2)
- Rebuild M2 with code execution + the 10 tools
- Compare to original M2 (which had 3 tools and no code exec)
- Hypothesis: v2 >> v1 because it can actually analyze data

### exp23: Agent model selection
- Agent selects per-origin best model from zoo
- Compare to: single best model, oracle selection, M5 (LLM vibes)
- Hypothesis: diagnostic-based selection > vibes-based

### exp24: Full agentic pipeline on event-augmented data
- Combine: event detection (news) + feature discovery + model selection
- This is the "does it all work together" test
- Compare to best non-agentic baseline

---

## Part 8: Timeline & Priorities

**Phase 1 (now):** Event-augmented simulator (exp20). This is the foundation — without it, agents have nothing to do.

**Phase 2:** Code execution + sandboxing + 10 tools. This is the capability unlock.

**Phase 3:** exp21 (feature discovery), exp22 (M2 v2), exp23 (model selection). These can run in parallel once Phase 1+2 are done.

**Phase 4:** exp24 (full pipeline). The capstone.

**Estimated effort:** Phase 1 is 2-3 days. Phase 2 is 3-4 days (mostly porting AI2 sandbox). Phases 3-4 are 1-2 weeks of experiments.

---

## Open Questions

1. How noisy should the synthetic news be? Too clean = trivial; too noisy = impossible.
2. Should events affect the real Freddie Mac data too, or only synthetic? (Real data already has real events — we could label them.)
3. How do we prevent the agent from overfitting to the synthetic event structure?
4. What's the right balance between template-based news and LLM-generated news?

---

**Next step:** Build the event-augmented simulator (Phase 1). Everything else depends on it.
