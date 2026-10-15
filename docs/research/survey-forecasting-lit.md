# Survey: Literature on LLM-Based Agents for Forecasting and Prediction

**Date:** 2026-10-15. **Purpose:** verify plan citations, extract design
patterns and evaluation standards for the TD-One mortgage-attrition system.
All arXiv IDs verified against arXiv on 2026-10-15. **All five cited papers
and Nexus exist.** No mismatches found.

## 1. Cited-paper verification

| Paper | arXiv ID | Status | Key facts |
|---|---|---|---|
| Xu et al., "LLM-based Agents for Forecasting and Prediction" | 2608.23058 | ✅ exists | 2026 survey; 15 authors; cs.AI |
| Karger et al., "ForecastBench" | 2409.19839 | ✅ exists | 2024; ICLR 2025; leaderboard at forecastbench.org |
| Williams et al., "Context is Key" (CiK) | 2410.18959 | ✅ exists | 2024; ServiceNow Research; v4 Jun 2025 |
| Schoenegger et al., "Wisdom of the Silicon Crowd" | 2402.19379 | ✅ exists | 2024; LSE/MIT/UPenn; v6 Jul 2024 |
| Requeima et al., "LLM Processes" | 2405.12856 | ✅ exists | 2024; U. Toronto; code on GitHub |
| Das et al., "Nexus" | 2605.14389 | ✅ exists | 2026; multi-agent TS forecasting; unofficial repro exists |

Bonus verifications (plan stretch goals): ADAS (Hu, Lu, Clune, arXiv:2408.08435) ✅;
ALMA (Xiong, Hu, Clune, arXiv:2602.07755, "Learning to Continually Learn via
Meta-learning Agentic Memory Designs") ✅.

## 2. Core contributions (agentic-forecasting relevance)

**Xu et al. (2608.23058).** The taxonomy paper. Organizes systems into three
groups: (a) standalone LLM workflows on encoded series/event context,
(b) tool- and retrieval-augmented agents, (c) hybrids pairing LLMs with
statistical or foundation models. Finds LLMs help most where context shifts
regimes; ablations sometimes show zero LLM uplift. Directly justifies the
plan's M0→M4 ablation ladder and the TemporalGuard leakage barrier.

**Karger et al. (2409.19839).** ForecastBench: 1,000 auto-generated,
regularly refreshed questions about *future* events (no answer exists at
submission time → contamination-proof by construction). Key result: expert
human forecasters beat the best LLM (p<0.001). Lesson for us: evaluate on
held-out *future* windows (the plan's 6 OOT months do this) and report
human/expert baselines, not just model-vs-model.

**Williams et al. (2410.18959).** CiK benchmark: numerical series + crafted
textual context where *every* task requires the text to solve. A simple LLM
prompting method beat statistical, TSFM, and fancier LLM forecasters — but
the paper also documents critical LLM failure modes. Justifies planting
verifiable macro/policy shocks in the synthetic DGP (the plan's §3.2): if
context never changes outcomes, the agent can't demonstrate uplift.

**Schoenegger et al. (2402.19379).** A 12-LLM ensemble on 31 binary questions
matched a 925-human crowd (not statistically different). Caveat: models show
an acquiescence bias (mean predictions >50% despite balanced outcomes).
Takeaway: ensemble diverse LLM personas/prompts and aggregate (median /
trimmed mean) — cheap variance reduction, but calibrate for the bias.

**Requeima et al. (2405.12856).** LLM Processes: elicits coherent *numerical
predictive distributions* from LLMs via rejection sampling and hierarchical
binning for continuous likelihoods, conditioned on natural-language priors.
Relevant to the plan's scenario-conditioning interface: natural language
("BoC cut 50bps, competitor undercuts by 30bps") → shifts/scales the
Beta-Binomial parameters.

**Das et al. Nexus (2605.14389).** Multi-agent decomposition of forecasting:
(1) Contextualization, (2) Dual-Resolution outlook generation (macro + micro
fluctuations separated), (3) Forecast Synthesis and Calibration. Beat
TimesFM-2.5 and CoT baselines on post-cutoff Zillow/stock data. Validates
the plan's controller-and-tools design and the macro/micro evidence split.

## 3. How winners combine LLMs with numerical models

The consistent pattern (Xu, Nexus, CiK): **LLM does context/reasoning,
TSFM does numerical extrapolation, combiner stays bounded.**
- Agency ladder: K=0 numerical-only → K=1 prompt-only LLM → K>1 closed-loop
  hybrid. Each rung must be ablated; the LLM rung sometimes adds nothing.
- Adjustments as bounded log-odds shifts (the plan's Δ∈[-0.75,+0.75]) match
  the literature: unbounded LLM edits destroy calibration.
- Multi-agent stage decomposition (Nexus) beats monolithic prompting for
  mixed numerical+textual tasks.
- Reflection loops (AlphaCast, arXiv:2511.08947; Cast-R1, arXiv:2602.13802)
  add a verify-and-revise step; Cast-R1 trains the policy with SFT + multi-turn RL.

## 4. Evaluation standards

- **Binary/event:** Brier score, log score (both strictly proper).
- **Continuous distributions:** CRPS (primary; reduces to MAE for point
  forecasts). Quantiles: pinball loss. Point: MAE/RMSE/MASE.
- **Calibration:** reliability diagrams, 80% PI empirical coverage
  (target 0.80±0.05), Winkler interval score.
- **Rule:** proper scores reward sharpness *subject to* calibration and
  punish unjustified confidence — exactly the failure mode of raw LLMs.
- Report cost and latency per forecast alongside accuracy (Xu; TokenCast
  shows token burn varies 10× across runs).

## 5. Pitfalls the papers warn about

1. **Leakage, two modes** (Xu): feature leakage (lookahead covariates) and
   parametric leakage (benchmark answers in pretraining). Mitigate with
   decision-time information barriers (TemporalGuard) and future-only eval.
2. **Contamination masquerading as skill**: benchmark gains may reflect
   memorization, not temporal reasoning — use live/refreshing questions.
3. **Overconfidence**: LLMs are poorly calibrated; always score with proper
   rules, never accuracy alone.
4. **Perturbation sensitivity**: small input changes can flip LLM forecasts —
   test robustness, don't trust single runs.
5. **Feedback loops**: deployed forecasts change the outcomes being forecast;
   relevant if manager actions shift future switching behavior.

## 6. Other 2025–2026 papers worth knowing

- **AlphaCast** (arXiv:2511.08947): human+LLM co-reasoning, reflector agent
  audits reasoning chains for hallucinations and circular reasoning.
- **Cast-R1** (arXiv:2602.13802): forecasting as sequential decision-making;
  tool-augmented workflow, memory-based state, SFT + multi-turn RL.
- **MemCast** (arXiv:2602.03164): memory-driven forecasting with
  experience-conditioned reasoning.
- **TokenCast** (arXiv:2609.35760): forecasts an agent run's own token
  consumption — useful for Groq 8k TPM budgeting.
- Curated list: `ustc-time-series/awesome-agentic-time-series` (GitHub).

## Implications for the plan revision

1. The M0→M4 ladder is exactly what Xu prescribes — keep it, freeze the
   model across ablations.
2. Add a reflector/verify step to the controller (AlphaCast pattern).
3. Keep Δ bounded; add calibration checks (coverage, Brier) as first-class
   gates, not afterthoughts.
4. The frozen-corpus backtest mode is the right call given no FRED/ADK keys;
   document it as the contamination control.
5. Budget tokens per forecast (TokenCast idea) — the Groq free tier is the
   binding constraint.
