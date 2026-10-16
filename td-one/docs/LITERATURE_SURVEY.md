# Literature Survey for TD-One Methodological Redesign

**Date:** 2026-10-16. **Purpose:** inform a redesign after the overnight
findings — tuned classical models (M0 w3, conformalized) beat real TSFMs
(Chronos-2, TimesFM-2.5) and the ReAct agent's interventions are
net-negative. Surveyed via web literature search; assessments are the
author's, with credibility notes per source.

**Reading guide:** §1–5 are per-area summaries. §6 is the redesign
recommendation (the actionable part). §7 lists what the literature does
*not* support — the guardrails.

---

## 1. LLM-based time series forecasting: what works vs hype

### 1.1 The optimistic case

**LLMTime (Gruver et al., NeurIPS 2023).** Treats forecasting as
next-token prediction over digit strings. Reports GPT-3 / LLaMA-2 70B
zero-shot beating ARIMA, TCNs, N-HiTS on CRPS and MAE across Monash and
Informer benchmarks. Proposed mechanisms: LLMs' repetition bias maps to
periodicity detection; arithmetic ability maps to trend extrapolation.
The paper itself flags data-leakage risk (undisclosed training data) and
finds chat-tuned versions forecast markedly *worse* than base models.
*Credibility: peer-reviewed, honest about limitations, but the headline
numbers rest partly on benchmarks that may be in training data.*

**Reprogramming / fine-tuning family: TIME-LLM, TEMPO, GPT4TS,
PromptCast, UniTime.** These freeze or lightly adapt a language model
backbone with time-series patching. TIME-LLM reports ~8% SMAPE
improvement over GPT4TS and competitiveness with N-HiTS on M4; TEMPO
shows cross-domain generalization. Few-shot (≤10% data) is where the
edge is largest. *Credibility: peer-reviewed; but the "baselines" are
mostly 2020–2022 deep models (Informer, Autoformer), not strong
classical methods, and few-shot is a niche regime.*

**TimeGPT (Nixtla).** Commercial foundation model, API-only, conformal
intervals by design. Independent evaluations show it competitive with
ARIMA/Prophet on standard benchmarks — roughly at parity, not dominant.

### 1.2 The skeptical case (more relevant to us)

**Park et al. (2025), "Revisiting LLMs as Zero-Shot Time-Series
Forecasters: Small Noise Can Break Large Models."** Finds LLM zero-shot
forecasters underperform even simple domain-specific models and are
fragile to input noise. Recommends fine-tuning over zero-shot prompting.
*Credibility: recent preprint, direct replication-style critique.*

**Zhou et al. (2025), "Why Do Transformers Fail to Forecast Time Series
In-Context?"** Theoretical result: under AR(p) data, linear
self-attention *cannot* beat classical linear models for in-context
forecasting; with chain-of-thought, predictions collapse to the mean
exponentially. *Credibility: theory paper; the AR(p) assumption is
narrow, but the CoT-collapse finding is a warning for any
"reason-then-predict-numbers" agent design — including our M2.*

**Risk.net (2025) reporting on UVA/UW work and Turnleaf Analytics.**
Ablating the LLM component from hybrid forecasters did not degrade
performance; on inflation forecasting, LLM errors were described as
"very large" versus simple ML. Practitioner quote: LLMs "train on as
much data as possible going back in time – data that may no longer be
relevant. They can't really adapt." *Credibility: journalism, but the
underlying claim (LLM adds nothing measurable) matches the preprints.*

### 1.3 Assessment for TD-One

There are **two distinct literatures** being conflated under "LLM
forecasting":

1. **TSFMs** (Chronos-2, TimesFM-2.5, Moirai, TimeGPT): transformers
   *trained on time-series data*. Legitimate, often strong — but our
   multi-seed results show they do not significantly beat a tuned
   classical baseline on our DGP. An independent retail benchmark found
   Chronos-2 winning while warning that "single-window rankings can be
   misleading (overfitting to a validation set is a real thing)."
2. **Raw-LLM number prompting** (LLMTime-style, our M2/M3's Groq calls):
   fragile, noise-sensitive, theoretically capped near linear-model
   performance, and CoT can actively collapse predictions.

Our negative M2 result is *exactly what this literature predicts*.
The honest M2 (real Groq + real Chronos-2, 2/3 segments much worse)
is consistent with: the LLM contributes no numerical skill, and its
overrides add noise.

---

## 2. Agentic forecasting: what roles work for LLMs

### 2.1 Published systems

**TimeSeriesScientist / TSci (2025).** Four specialized agents —
Curator (diagnostics/preprocessing), Planner (model selection),
Forecaster (fitting/ensembling), Reporter (audit trail). Reports
10.4% error reduction vs statistical baselines, 38.2% vs LLM
baselines, on eight benchmarks. **Critical detail: the LLM never
produces the forecast numbers.** It selects preprocessing, models,
and ensemble weights; classical/TSFM tools do the numerics.
*Credibility: preprint; the 10.4% is a mean over benchmarks the
authors chose. The architectural lesson (LLM orchestrates, tools
compute) is more robust than the number.*

**REATS (Microsoft Research, 2026).** LLM as *ensemble router*:
produces interpretable sample-adaptive ensemble weights via
chain-of-thought over textual + numerical features. Beats fixed
ensemble baselines. Again: the LLM outputs *weights*, not values.

**DCATS (2025).** LLM agent for *data-centric* optimization:
selects auxiliary series to enrich training data (6% error reduction
on traffic data). The LLM's job is dataset design, not forecasting.

**"LLM-based Agents for Forecasting and Prediction" (survey).**
Key inference-time finding: "verified reasoning traces can improve
performance, whereas **excessive reasoning tokens may reduce accuracy,
extended deliberation may worsen calibration**, and tested prompt
interventions often fail to improve accuracy." Useful compute
"diversifies or filters information rather than repeating unchanged
inputs."

### 2.2 Assessment for TD-One

The literature converges on a single viable division of labor:

| LLM role | Evidence | TD-One analog |
|---|---|---|
| Orchestrator / model selector | TSci | ❌ we don't do this |
| Ensemble weight router | REATS | ❌ we don't do this |
| Data/dataset designer | DCATS | ❌ we don't do this |
| Evidence synthesizer / reporter | TSci Reporter | ⚠️ partial (evidence corpus exists) |
| **Numerical forecaster** | **nothing supports this** | **M2 — our negative result** |
| **Numerical adjuster of forecasts** | **judgmental-adjustment lit says harmful (§3)** | **M3 — our negative result** |

Our M2 asked the LLM to do the two roles with the *least* support.
The survey's "deliberation worsens calibration" finding is a direct
mechanistic hypothesis for our honest-M2 outcome.

---

## 3. Hybrid classical + LLM: the judgmental-adjustment literature

This is the most directly relevant literature for M3 — and it is
decades old, from human forecasting. M3 (LLM adjusts a classical
forecast) is formally a **judgmental adjustment** mechanism, and the
literature has a clear verdict.

**Forecasting: Principles and Practice (Hyndman & Athanasopoulos,
2nd ed).** Adjustments should be used *sparingly*, only with
"significant additional information" not in the statistical model;
"adjustments should not aim to correct for a systematic pattern in
the data that is thought to have been missed by the statistical
model… Statistical models are much better at taking account of data
patterns, and judgmental adjustments only hinder accuracy." Small
adjustments, especially optimistic ones, hinder accuracy; large
adjustments with real information can help.

**Armstrong (evidence-based forecasting checklists); Meehl (1954);
Grove et al. (2000).** Adjustments to quantitative forecasts
"often increase errors." The standing recommendation since 1954:
do not subjectively adjust model forecasts.

**Franses and coauthors.** More expert adjustment (larger variance
of deviations from the model) associates with *lower* forecast
accuracy across GDP/inflation/unemployment consensus forecasts.

**ECB Survey of Professional Forecasters + LLM ensemble (2025).**
An LLM *combining* expert forecasts (not overriding a model)
reduced errors, especially under expert disagreement/inattentiveness.
Note the direction: LLM as *aggregator*, not as *adjuster*.

### Assessment for TD-One

The literature **predicted our exp4/exp6 results in advance**:
M3's over-adjustment (+0.30 on benign, wrong-sign on osfi) is the
textbook pathology. Our two fixes map exactly onto the literature's
prescriptions:

- **Regime gate** ("do nothing" prior) = "adjust only with significant
  additional information." Our held-out win (7.60 vs 7.86) is the
  first M3 variant the literature would endorse.
- **Shrinkage** (k=0.5) = dampening the small optimistic adjustments
  the literature says to avoid. It helped vs M3 but not vs M0 —
  consistent with "small adjustments hinder accuracy" (even shrunk
  ones, if applied without strong evidence).

**Design implication:** any future LLM-adjustment path must be
*gated and shrunk by default*, with the burden of proof on the
adjustment, not on the baseline. The current M3 default (always
adjust, bound ±0.75) inverts the literature's recommendation.

---

## 4. Conformal prediction for time series

**CQR — Conformalized Quantile Regression (Romano et al., NeurIPS
2019).** Start from quantile estimates, calibrate with
max(q_lo − y, y − q_hi) scores. Keeps heteroskedasticity *and* gives
finite-sample coverage. Multiple applied sources call it "usually the
right default."

**ACI — Adaptive Conformal Inference (Gibbs & Candès, NeurIPS
2021).** Online α update (α_{t+1} = α_t + γ(α − err_t)); long-run
coverage converges under *arbitrary* distribution drift. One tuning
parameter, no training overhead. The recommended upgrade when regimes
shift.

**EnbPI (Xu & Xie, ICML 2021).** Bootstrap ensemble + out-of-bag
residuals; avoids the train/calibration split (good for short
series); sliding residual window handles drift without refitting.

**SPCI; Conformal PID (Angelopoulos et al., NeurIPS 2023).** Refinements
for sequential dependence and tighter control-theoretic guarantees.

**Best practices (synthesized):**
- Never vanilla split-conformal on time series and expect coverage.
- Horizon-specific calibration (per-horizon quantiles beat joint).
- Normalized / locally adaptive scores for heteroskedasticity.
- Calibration window tuned to the regime-persistence half-life
  (too short = noisy q̂, too long = stale).
- Coverage is *marginal*, not conditional — report width and
  conditional coverage by regime alongside the headline number.

### Assessment for TD-One

Our `conformal.py` (split-conformal CQR-style widening in rate space)
is the literature's recommended default, and exp5 validates it
(0.671→0.914 coverage; conf_w6 beats m0_w6 on CRPS). Natural next
steps, in order: **(a)** ACI as the drift-aware upgrade (one γ to
tune); **(b)** horizon-specific calibration; **(c)** normalized scores
using the Beta-Binomial spread. None of this needs an LLM.

---

## 5. Macro-finance / credit-risk forecasting: how the industry does it

**Mortgage prepayment (the closest industry analog to attrition).**
The industry standard decomposes CPR into multiplicative factors:

- **Refinance incentive** — the dominant driver, modeled as an
  **S-curve** (arctan) of (note rate − market rate), not linear.
  Linear incentive models are a known failure mode.
- **Seasoning** — ramp over ~30 months (the PSA curve's core insight).
- **Burnout** — pools exposed to low rates lose refinancing propensity;
  must be tracked as max-historical-incentive per loan.
- **Seasonality** — spring/summer elevation.
- **Competing risks** — prepayment, default, curtailment as joint hazards.
- **Heterogeneity** — FICO × incentive interactions are first-order.

**Feature canon** (Freddie Mac + XGBoost studies with honest temporal
splits): rate_incentive, burnout, credit score, LTV, DTI, loan age,
HPI growth since origination, market rate (FRED MORTGAGE30US),
current LTV. Evaluated across regimes: 2021 refi wave (rate-driven)
vs 2023–24 lock-in (turnover-driven) — models must survive both.

**Bank churn/attrition.** Logistic regression and gradient boosting
dominate; transaction intensity and balance dynamics are the top
predictors; interpretability is a hard requirement for stakeholders.

### Assessment for TD-One

Our DGP's drivers (payment_shock, rate_spread, family_squeeze,
osfi_exempt) are directionally right but miss the industry's three
biggest structural features: **burnout, seasoning, and the S-curve
nonlinearity**. The exp8 macro overlay (FRED 30y rate driving the
logit) is the correct *kind* of mechanism — the industry would add:
incentive as S-curve of (origination rate − current market rate),
per-loan burnout state, and a 30-month seasoning ramp. These are
all implementable in the simulator and would make the DGP a much
harder, more realistic test — one where macro-aware models *should*
win and naive ones fail.

---

## 6. Methodological redesign: recommendations

### 6.1 Keep (validated by our experiments and/or the literature)

1. **Conformalized M0 with tuned window as the baseline.** Promote
   `conf_w3`/`conf_w6` to the default M0. The literature (§4) and
   exp5 agree: this is the right default, and it closes the project's
   open calibration problem.
2. **Multi-seed paired evaluation protocol.** Our strongest
   methodological asset. The window-direction flip (seed 2) proves its
   value. Keep 5+ seeds for every model comparison; report the M0
   envelope, not a point M0.
3. **FRED macro integration.** Exp8 (§5-aligned: rate incentive is the
   industry's #1 driver) justifies this workstream. The 17 cached
   series should feed models, not sit unused.
4. **Regime gate on any adjustment path.** The one M3 idea both our
   held-out test and 70 years of judgmental-forecasting literature
   endorse.

### 6.2 Kill or fundamentally rethink

1. **M2 as ReAct-numerical-forecaster.** Both our honest run and the
   literature (§1.2, §2.2) say the LLM should not produce or override
   numbers. Do not iterate on prompts; change the role.
2. **M3's default-always-adjust.** Invert the default per §3: the
   adjustment needs evidence (gate), and magnitude needs shrinkage.
   The ±0.75 bound alone is insufficient (ablation showed it rarely
   binds).
3. **Mock-backend claims about agentic skill.** The seg-1 "win" was a
   mock-sharpness artifact. Future agent evaluations run on real
   backends only.

### 6.3 Try (ordered by expected value)

1. **LLM as ensemble router (REATS-style).** The LLM outputs
   *weights over classical/TSFM forecasts* with natural-language
   rationale, never values. This is the best-supported agentic role
   in the literature and directly testable: does LLM weighting beat
   inverse-CRPS weighting (§exp5: 4.078)?
2. **LLM regime narrative → gate.** Instead of deltas, the LLM emits
   a structured regime assessment (direction + confidence) that
   *modulates the statistical gate threshold*. Keeps the LLM in the
   role the literature supports (information synthesis) while the
   numbers stay classical.
3. **ACI upgrade to conformal.py.** One γ parameter; handles the
   regime drift our DGP's shocks create. Compare against static CQR
   on the shock scenarios.
4. **DGP realism upgrade (burnout + seasoning + S-curve).** Makes the
   simulator a harder test that rewards macro-aware models — the
   current DGP may be *too easy* for M0, which flatters classical
   methods.
5. **TSFM + exogenous FRED.** Chronos-2/TimesFM-2.5 support covariates
   in some configurations; test whether macro-conditioned TSFMs beat
   the conformal baseline. This is the fair rematch the TSFMs deserve.

### 6.4 Explicitly do not try

- Prompt-engineering the M2 agent to "be more careful" (CoT-collapse
  literature suggests more deliberation hurts calibration).
- LLM zero-shot number forecasting as a product direction.
- Bigger LLM as the fix (LLMTime: chat-tuned/larger ≠ better at
  forecasting; our Groq is already 120B).

---

## 7. What the literature does NOT support

1. **No published evidence** that a ReAct-style agent beats tuned
   classical methods on probabilistic forecasting when the agent
   produces or overrides the numbers. TSci's 10.4% comes from
   orchestration, not agentic numerics.
2. **No evidence** that LLMs' economic "knowledge" translates into
   better forecasts without grounding — and positive evidence of
   harm (noise sensitivity, CoT collapse, deliberation hurting
   calibration).
3. **No support** for the claim that TSFMs should beat well-tuned
   classical baselines in general; the honest claim is narrower
   (strong zero-shot, good with covariates, competitive at parity).
4. **The judgmental-adjustment literature is one-sided**: 70 years,
   no replicated finding that routine expert adjustment of a good
   statistical forecast helps. Our M3-gate result is consistent with
   the *exception* clause (adjust only on strong extra information),
   not a refutation.
5. **Conformal coverage is marginal, not conditional.** No method in
   §4 guarantees coverage *within* a regime (e.g., during a rate
   shock). Our shock-scenario coverage should be reported
   conditionally, not assumed.

---

## References (key sources)

- Gruver et al., "Large Language Models Are Zero-Shot Time Series
  Forecasters," NeurIPS 2023. <http://arxiv.org/pdf/2310.07820>
- Park et al., "Revisiting LLMs as Zero-Shot Time-Series Forecasters:
  Small Noise Can Break Large Models," 2025. <https://arxiv.org/pdf/2506.00457>
- Zhou et al., "Why Do Transformers Fail to Forecast Time Series
  In-Context?" 2025. <https://arxiv.org/pdf/2510.09776>
- Jin et al., "TIME-LLM: Time Series Forecasting by Reprogramming
  Large Language Models," ICLR 2024.
- Cao et al., "TEMPO: Prompt-based Generative Pre-trained Transformer
  for Time Series Forecasting," 2023. <https://arxiv.org/pdf/2310.04948>
- He et al., "TimeSeriesScientist," 2025. <https://arxiv.org/pdf/2510.01538v2>
- Zhang et al., "REATS: LLM Reasoning-based Ensemble Learning for
  Adaptive Time Series Forecasting," Microsoft Research, 2026.
- Yeh et al., "DCATS: Data-Centric Agent for Time Series," 2025.
  <https://arxiv.org/pdf/2508.04231>
- "LLM-based Agents for Forecasting and Prediction: Methods, Training,
  Evaluation, and Applications" (survey). <https://arxiv.org/pdf/2608.23058>
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*,
  2nd ed., ch. on judgmental adjustments. <https://otexts.com/fpp2/judgmental-adjustments.html>
- Armstrong (ed.), "Forecasting methods and principles:
  Evidence-based checklists."
- Franses et al., "Does More Expert Adjustment Associate with Less
  Accurate Professional Forecasts?" *JRFM* 2020.
- Radhakrishnan et al., "Can LLM Improve for Expert Forecast
  Combination? Evidence from the ECB Survey," 2025. <https://arxiv.org/pdf/2506.23154>
- Romano et al., "Conformalized Quantile Regression," NeurIPS 2019.
- Gibbs & Candès, "Adaptive Conformal Inference under Distribution
  Shift," NeurIPS 2021.
- Xu & Xie, "Conformal Prediction Interval for Dynamic Time-Series
  (EnbPI)," ICML 2021. <https://arxiv.org/pdf/2010.09107v6>
- Angelopoulos et al., "Conformal PID Control for Time Series
  Prediction," NeurIPS 2023.
- PSA prepayment model. <https://en.wikipedia.org/wiki/PSA_prepayment_model>
- Risk.net, "The prediction of mortgage prepayment risks in the early
  stages of loan origination: a machine learning approach,"
  *Journal of Risk*.
- Risk.net, "Why know-it-all LLMs make second-rate forecasters," 2025.
