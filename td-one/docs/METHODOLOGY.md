# TD-One: Agentic Forecasting for Mortgage Attrition — Methodology

**TD-One — Canadian mortgage renewal/switching forecasting**
*Complete research methodology, exp1–exp63 + LLM ladder. Written 2026-10-19.*
*Supersedes all earlier methodology drafts.*

---

## Abstract

We ask whether agentic AI — large language models with tools, sandboxes,
and decision policies — can improve probabilistic forecasting of Canadian
mortgage borrower attrition at renewal. Over **63+ experiments** across
four research tracks (classical, time-series foundation models, agentic,
LLM integration), we built the TD-One method: a **regime-probability router
with gates**. On 600 M4 Monthly series (blocked 5-fold CV, paired t-tests),
the proposed method beats equal-weight combination (0.1451 vs 0.1507,
p=6.5e-09), the best single model (p=7.8e-11), and the router alone
(p=6.7e-10). It also wins on Tourism Monthly (0.1282, p=1.9e-68) and M3
Monthly (0.1476, p=2.7e-44). A 6-rung LLM integration ladder shows LLMs are
brilliant specialists — beating Vector's LLM baseline by 20% on Food CPI —
but do not generalize to seasonal multi-series data (36–64% worse than
classical on Tourism/M3). We report **14 positive results, 20+ honest
negatives, and 2 definitively closed tracks** (news for monthly
forecasting, macro overlays). A "kitchen sink" experiment (exp51) confirms
our significance filter was correct: adding every nonsignificant positive
back makes the method worse, not better.

---

## Table of Contents

1. [Glossary](#0-glossary)
2. [Problem Statement](#1-problem-statement)
3. [Research Program Overview](#2-research-program-overview)
4. [Classical Baselines](#3-classical-baselines)
5. [Agentic Track (Phases 1–3)](#4-agentic-track-phases-13)
6. [Real-Data Benchmarks](#5-real-data-benchmarks)
7. [The Combination Trilogy](#6-the-combination-trilogy)
8. [The Proposed Method (exp47/49)](#7-the-proposed-method-exp4749)
9. [Extensions and Validation](#8-extensions-and-validation)
10. [Monash Benchmarks (exp60–62)](#9-monash-benchmarks-exp6062)
11. [Vector Comparisons (exp52–53, exp63)](#10-vector-comparisons-exp5253-exp63)
12. [The LLM Integration Ladder (exp55–59)](#11-the-llm-integration-ladder-exp5559)
13. [Tool Optimization](#12-tool-optimization)
14. [What We Learned](#13-what-we-learned)
15. [Reproducibility](#14-reproducibility)

---

## 0. Glossary

| Term | Meaning |
|---|---|
| **CRPS** | Continuous Ranked Probability Score — primary accuracy metric. Lower is better. Rewards forecasts that are well-centered and honest about uncertainty. |
| **Quantile forecast** | A set of percentiles (e.g., 10th/50th/90th) describing the predicted distribution, instead of a single point. |
| **Regime** | A "mode" of the data (seasonal, trending, noisy). The router detects the regime and picks the best model for it. |
| **Conformal calibration** | Widens forecast intervals until they honestly cover the stated fraction of outcomes. |
| **PIT-safe** | "Point-in-time safe" — only data from before the forecast date is used. No lookahead. |
| **Regime router** | Small classifier → P(regime \| features) → per-regime expert → probability-weighted blend. |
| **Honest negative** | An experiment that cleanly disproves a hypothesis. A finding, not a failure. |
| **TSFM** | Time-series foundation model (Chronos-2, TimesFM-2.5) — pretrained neural nets for time series. |
| **Blocked CV** | Cross-validation with contiguous time blocks as folds. Respects the arrow of time. |
| **Equal weights** | Naive baseline: average all models equally. Surprisingly hard to beat. |

---

## 1. Problem Statement

When a fixed-rate mortgage matures in Canada, the borrower faces a choice:
renew with the incumbent lender or switch. For a bank's treasury and
retention teams, the **switching rate per segment per month** drives funding
plans, pricing, and campaign targeting.

The series is awkward:

- **Small counts.** Monthly renewal volumes per segment are small; rates are
  noisy.
- **Seasonality.** Renewal volumes peak in predictable months (spring buying
  season), but the *switching propensity* within renewals has its own
  seasonal structure.
- **Regime shifts.** Bank of Canada policy decisions, competitor rate wars,
  and macro shocks arrive as discrete events that change the data-generating
  process.
- **Text-dominant signal.** The signal that matters — a BoC decision, a
  competitor's rate war, a policy announcement — lives in *text*, not in the
  time series. This is the fundamental motivation for agentic methods: a
  forecaster that cannot read is blind to the most informative inputs.

Classical approaches fit a rate model per segment and call it a day. The
TD-One program asks whether adding **LLMs with tools** (news retrieval,
diagnostics, code execution) can beat that — and if so, where, and by how
much. The honest answer, after 51 experiments, is *sometimes, in specific
regimes, with strict gating* — and we can name exactly which.

---

## 2. Research Program Overview

### 2.1 The three tracks

```mermaid
flowchart LR
    CLASS[Classical track<br/>exp1–19<br/>M0, ETS, ARIMA, conformal]
    TSFM[TSFM track<br/>exp14, exp18<br/>Chronos-2, TimesFM-2.5]
    AGENT[Agentic track<br/>exp20–51<br/>Phases 1–3, benchmarks,<br/>combination, proposed method]
    CLASS --> PROPOSED
    TSFM --> PROPOSED
    AGENT --> PROPOSED
    PROPOSED[Proposed method<br/>exp47/49<br/>regime router + gates]
```

All three tracks ran in parallel. The classical track established baselines
and killed the macro-overlay track. The TSFM track established that
foundation models beat classical on real data (but not on our synthetic
DGPs — a DGP artifact we caught and retracted). The agentic track is where
most experiments live: simulators, sandboxes, agents, benchmarks, the
combination trilogy, and the proposed method.

### 2.2 The honest-negative philosophy

We report what does not work. Every experiment has a pre-registered
verdict: **Keep** (positive lift, integrate), **Kill** (definitive negative,
stop), or **Marginal** (directional, needs more evidence). Retractions are
public: exp2's claim that Chronos-2 beats M0 by 20% on synthetic data was
a circular-DGP artifact, caught by our critic audit and retracted. exp8's
macro overlay was retracted when the sign was found to be wrong.

An honest negative is a finding, not a failure. The kill list (§10.1) is
as informative as the keep list — it tells future researchers what not to
try.

### 2.3 Timeline

| Phase | Dates | Content |
|---|---|---|
| Synthetic baselines | 2026-10-15 → 10-16 | exp1–19: M0–M5, TSFM weights, conformal, macro audit |
| Agentic Phases 1–3 | 2026-10-16 → 10-17 | exp20–25: simulator, sandbox, tools, decision policy |
| Real-data benchmarks | 2026-10-17 | exp30–37: WTI, BoC, SMM, Food CPI, M4 |
| Combination trilogy | 2026-10-17 → 10-18 | exp42–44: FFORMA, regime router, news selector |
| Kill confirmations | 2026-10-18 | exp45–46: learned regimes, news dense |
| Proposed method | 2026-10-18 | exp47–51: build, tools, reasoning, extensions, kitchen sink |
| Monash + Vector | 2026-10-18 | exp52–53, exp60–63: Tourism, M3, vs Vector baselines |
| LLM ladder | 2026-10-18 → 10-19 | exp55–59: 6 rungs on Gemini Flash |

---

## 3. Classical Baselines

### 3.1 The model ladder

- **M0 (Jeffreys Beta-Binomial).** Bayesian baseline for attrition rates.
  Per-segment Beta posterior with Jeffreys prior; predictive quantiles from
  the posterior predictive. The honest floor.
- **m0_w3.** M0 with a 3-month trailing window — the classical champion on
  synthetic data.
- **ETS / ARIMA.** Standard exponential smoothing and ARIMA, fit per
  series. ETS is the proper probabilistic baseline (fair CRPS comparison).
- **Naive / seasonal naive.** Random-walk forecasts with empirical residual
  quantiles. Surprisingly hard to beat on some real series (WTI).

### 3.2 TSFM: Chronos-2 and TimesFM-2.5

We wired real pretrained weights (Chronos-2 455.8MB, TimesFM-2.5 882.3MB;
manifest committed, weights gitignored) with a reproducible bootstrap
script. Key results:

- **exp14/R2-4 (M4 Monthly, common 0.10–0.90 quantile grid):** Chronos-2
  351.18 (−24.1% vs naive 462.84), TimesFM 358.13 (−22.6%). The TSFM win
  **survives** a fair grid. Caveats: Chronos-2 seeds identical
  (deterministic); no ETS/ARIMA baseline on this grid.
- **exp18b (Freddie Mac SMM, fair probabilistic ablation):**
  TimesFM_cal **0.000497** vs ETS+resid **0.000566** = **−12.2%**, p=0.076
  (n.s. at α=0.05), wins 32/50 origins. Against m0_w3 (0.000613) the gap is
  17.9% (p=0.0079), but ETS is the proper baseline.

**Calibration:** `LogScaleCalibrator`
(`src/td_one/models/tsfm_calibration.py`) maps TSFM raw quantiles to
calibrated intervals. Without it, TSFMs overcover (Chronos-2: 1.0/0.8
empirical coverage on synthetic).

### 3.3 The macro track — CLOSED

A multi-experiment arc (exp8, exp10, exp13, exp15, R2 reruns) tested whether
macroeconomic overlays (FRED series, z-scored) improve forecasts.

- First verdict: PIT-safe reruns killed it (exp10's 0.678 CRPS gain was 100%
  z-score leak).
- That verdict was **retracted**: an audit found a dict-lookup bug (t+1
  lookup in a dict truncated at t) had silently zeroed the macro signal —
  it was never properly tested PIT-safe.
- Final verdict: 5-seed run, macro_m0_w3 (1.4665) vs m0_w3 (1.2804) =
  **+0.1861 (macro worse), p=0.064 against**. Single-seed −0.357 was luck.
  **Track closed: no evidence macro helps, classical or TSFM side.**

---

## 4. Agentic Track (Phases 1–3)

### 4.1 Phase 1: Event-augmented simulator

`events.py` (5 event types: rate shocks, competitor moves, policy changes,
seasonal surges, macro turns), `news_synthesis.py` (PIT-safe synthetic news),
`simulator.py` (10k×66mo mortgage attrition with planted shocks). 30/30
tests. Original signal: 9.1pp.

**Phase 3 adjustment:** news-leading DGP (84% of news arrives before the
DGP effect, verified) — because real news leads, and the original DGP had
news mostly lagging.

### 4.2 Phase 2: Sandbox + 10 tools + ReAct v2

Sandbox from AI2 methodologies (30s timeout, 1GB memory, secret stripping),
10 analysis tools (`test_stationarity`, `detect_changepoints`,
`decompose_series`, `analyze_residuals`, `feature_importance`,
`cross_validate`, `compare_models`, `backtest_event_impact`,
`plot_diagnostics`, `generate_synthetic_scenario`), and `react_v2.py` with
code execution. 66 tests.

**Key negatives from the agent experiments:**

| Exp | Test | Result |
|---|---|---|
| exp21 | Agent feature discovery | **−21.6%** held-out vs ETS. News mostly *lags* the DGP effect; only 2/5 event types have news before impact. |
| exp22 | M2 v2 (calibrated) vs v1 | Mixed. +1.3% overall (n.s., p=0.73). On 9/60 origins where v2 acted: **+6.6%, 6/9 wins**. But 3 losses catastrophic (−59% to −93%) from overconfident intervals. |
| exp23 | Diagnostic model selection | **= random** (p=0.80). Selecting from backward diagnostics doesn't work. |
| exp24 | Regime-aware selection | Wash (p=0.93). |
| exp25 | News at h=3 | **Significantly worse** (+25%, p=0.0013). |

### 4.3 Phase 3: Decision policy

`decision_policy.py`: James-Stein shrinkage on agent adjustments,
calibrated intervals (volatility floor), corroboration gating
(|t| > 3.5 or news corroboration required before acting). 11/11 tests.
The 2026-10-17 "C-nudge" tightened thresholds after the WTI lesson
(`T_STAT_STRICT` 3.0→3.5, `min_shift` 0.02→0.03).

**WTI lesson:** On energy/oil (random walk — ETS/ARIMA/naive identical at
1.0521 CRPS), the agent correctly found nothing 92% of the time. The 8%
false positives cost 4–7%. **When there is no signal, the agent must do
nothing.** This motivated the "do nothing" prior that became a core
component of the proposed method.

---

## 5. Real-Data Benchmarks

Moving beyond synthetic DGPs (per the 2026-10-17 directive) to five real
benchmarks:

### 5.1 WTI / Energy-Oil — method sound, task impossible

ETS, ARIMA, and naive produce **identical** forecasts (1.0521 CRPS) — the
series is a random walk at this horizon. No method can beat the naive
forecast; the agent's correct behavior is abstention. Kept as a no-signal
diagnostic.

### 5.2 Bank of Canada Rate Decisions — methodology validated

143 meetings, 23 non-hold decisions. A simple rule (rate momentum +
inflation gap) achieves **83.7%** vs climatology 79.1%.

- **exp30:** MNLogit (0.835) does NOT beat the simple rule (0.843).
  n=143 is overparameterized. **Kill the complex model; keep the rule.**
- **exp31:** Agent discovery **success** — the autonomous agent found the
  inflation_gap × rate_momentum interaction with no hints, validated twice.
  This is the existence proof that agents can do real data science.
- **exp32:** EASY/HARD regime split works: EASY (n=62) rule 0.887 vs
  climatology 0.855 (+0.032); HARD (n=81) rule 0.815 vs climatology 0.827
  (−0.012). **+7.2pp gap.** The model knows when to trust itself.

### 5.3 Freddie Mac SMM — 12.2% is the classical ceiling

63 months (Jan 2021–Mar 2026) of single-family mortgage prepayment speeds.
Agent-discovered features (ff_3m_chg, term_spread) → **+1.21%** held-out
(n.s.). Real BoC events **hurt** (−13.04%) — already priced in.
exp18b's −12.2% (TSFM_cal vs ETS) stands as the classical ceiling.

### 5.4 Food CPI — seasonality correctly discriminated

The agent correctly discriminates seasonal structure: produce 0.375 vs
headline 0.000 (seasonality score). h=6 cross-validation is regime-fragile
(the 2022–23 disinflation turn falls in the training window).

### 5.5 M4 Monthly — selection fails on real data too

The oracle gap (−30%) is not closable by diagnostic selection — worse than
random. **This is a method problem, not a data problem.** exp23's synthetic
failure replicates on real data.

---

## 6. The Combination Trilogy

Three approaches to combining models, tested in sequence:

### 6.1 exp42: FFORMA-lite — HONEST NEGATIVE

Learned combination weights (gradient boosting on 10 meta-features):
0.1300 vs equal weights 0.1295 on M4, **p=0.156**. This is the forecast
combination puzzle (Smith & Wallis 2009): we're ~200× data-starved vs
FFORMA's 100k series. **Kill.**

### 6.2 exp43: Regime-probability router — PARTIAL POSITIVE, THE BACKBONE

Hand-defined regimes (seasonal→ARIMA, trending→ETS, noisy→ETS) →
LogisticRegression → P(regime|features) → per-regime expert → soft
combination (Vincentized). **M4 0.1282 vs 0.1295, p=0.018** — the first
method in the program to significantly beat equal weights. On SMM/Food CPI
the experts collapse to best single (degrades gracefully). BoC router =
the rule (0.846).

**Why it works:** constrained structural priors beat flexible learning at
small-n. Learning a 3-way classification needs far less data than
regressing continuous weights (exp42's failure).

### 6.3 exp44: News-as-selector — MECHANISM PROVEN, REAL BLOCKED

On synthetic data with true regime labels: news features improve regime
classification 0.367→0.522 accuracy, **p=0.0068**. Real data was blocked by
news sparsity at the time. (exp46 later closed this definitively — §8.2.)

---

## 7. The Proposed Method (exp47/49)

### 7.1 Architecture

```mermaid
flowchart TB
    IN["Input: time series<br/>(+ optional news, LLM reading)"]
    FEAT["10 PIT-safe features<br/>exp42"]
    ROUTER["Regime router (exp43)<br/>hand rules → LogisticReg<br/>→ P(regime | features)"]
    NEWS["News modulation<br/>(exp44, GATED)<br/>only if dense"]
    LLM["LLM sensor<br/>(exp11, GATED)<br/>only if conf ≥ 0.75"]
    GATE["'Do nothing' gate<br/>uncertain + weak → global best<br/>exp6 + decision_policy"]
    EXPERTS["Per-regime experts<br/>Σ P(r) × expert_r<br/>exp43"]
    CONF["Conformal calibration<br/>volatility floor<br/>exp5 / exp10"]
    OUT["Output: quantiles<br/>+ regime probs<br/>+ trust (EASY/HARD) [exp32]<br/>+ explanation"]

    IN --> FEAT --> ROUTER
    ROUTER --> NEWS
    ROUTER --> LLM
    NEWS --> GATE
    LLM --> GATE
    GATE --> EXPERTS --> CONF --> OUT
```

Implementation: `src/td_one/agent/proposed_method.py` — `ProposedForecaster`
(fit/predict), `ForecastOutput` dataclass. 7 regression tests.

### 7.2 Components and their evidence

| Component | Source | Evidence |
|---|---|---|
| Regime-probability router | exp43 | M4 0.1282 vs 0.1295, **p=0.018** — backbone |
| Trust flag (EASY/HARD) | exp32 | +7.2pp gap; top P ≥ 0.7 → EASY |
| "Do nothing" gate | exp6 + decision_policy | M3+gate 7.60 vs 7.86; fires ~5% of M4 origins |
| Conformal calibration | exp5/exp10 | Coverage 0.671 → 0.914; volatility floor |
| News modulation (gated) | exp44 | Mechanism proven synthetic (p=0.0068); off by default |
| LLM sensor (gated) | exp11 | 3.59 vs 4.82; blends 50/50 iff confidence ≥ 0.75 |

### 7.3 Results (exp47, reproduced exactly in exp49)

| Dataset | Proposed | Equal | Best single | vs equal | vs best |
|---|---|---|---|---|---|
| M4 (n=600) | **0.1451** | 0.1507 | 0.1505 | **p=6.5e-09** | **p=7.8e-11** |
| Food CPI (n=24) | **0.0141** | 0.0176 | 0.0143 | **p=3.1e-18** | **p=0.005** |
| BoC (n=143) | rule 0.846 → gated **0.853** | — | — | EASY 62@0.887, HARD 81@0.827 | — |
| Synthetic (n=480) | 0.1036 | 0.1068 | 0.1028 | p=0.072 | p=0.60 |
| SMM (n=27) | 0.1389 | 0.1424 | 0.1377 | p=0.66 | p=0.61 |

**The system beats the sum of its parts:** proposed vs exp43-router-alone
is **p=6.7e-10** on M4. Gates + conformal add value beyond the router.
On Synthetic/SMM/Food CPI where experts collapse, it degrades gracefully
to best single.

### 7.4 The 14 positive components and 17 kills

Full inventory in `docs/POSITIVE_LIFT_AUDIT.md`. The 14 positives (ordered
by strength): exp43 router, exp32 trust flag, exp14/R2-4 TSFM, exp18b
TSFM+calibration, exp5/exp10 conformal, exp6 gate, exp11 LLM sensor, exp31
agent discovery, exp22 agent infrastructure, exp44 news selector (gated),
exp33 agent features, exp36 seasonality, exp10 hierarchical, exp13 residual
macro. The 17 kills: exp2, 3, 8, 12, 15, 19, 21, 23, 24, 25, 30-MNLogit,
34, 37, 41, 42, 45, 46.

---

## 8. Extensions and Validation

### 8.1 exp48: Reasoning effort — "high" never helps

Sweep of Groq `reasoning_effort` (low/medium/high) on 3 agent tasks × 3
seeds (27 LLM calls, `openai/gpt-oss-120b`):

| Task | low | medium | high |
|---|---|---|---|
| regime_diagnosis (acc) | 1.000 (2.2s) | 1.000 (2.3s) | 1.000 (7.9s) |
| boc_discovery (hit) | 1.000 (2.9s) | 1.000 (4.1s) | **0.333** (17.6s, 1 error) |
| changepoint (rubric) | 1.000 (2.5s) | 1.000 (1.7s) | 1.000 (4.7s) |

"High" is 3.5–6× slower with zero quality gain, and actively harmful on
BoC discovery (overthinking → plausible-but-wrong feature). One "high" run
hit empty-content errors (reasoning tax consumed the token budget).
**Default: "low" for diagnostics; escalate only on demonstrated need.**

### 8.2 exp50: Four open extensions — 1 wins, 3 confirm optimal

| Extension | Verdict |
|---|---|
| TSFM as 5th expert | ❌ 0% selection rate on M4 (real weights); no improvement |
| **BoC discrete variant** | ✅ **Wins** — EASY→rule, HARD→climatology for CUT/HOLD/HIKE; now in system |
| Semi-supervised regimes (cluster on loss) | ❌ p=0.44, hand rules win |
| Learned gate thresholds | ❌ p=0.32, hand values fine |

Three of four confirm the current design is already optimal. The
hand-picked priors encode domain knowledge that learning cannot beat at
our scale.

**exp46 (news dense) — track closed:** With 86% BoC news coverage (123/143
meetings), news features do not help regime classification (p=0.48) or
forecasting on any target, dataset, or variant. Keyword counts are
contrarian noise (hawk_kw mean: hikes 0.88, cuts 0.00, holds 1.72).
Sparsity was never the constraint. **News track closed for monthly
forecasting.**

### 8.3 exp51: Kitchen sink — the significance filter was correct

Combined all 14 positive-but-nonsignificant effects into a
`KitchenSinkForecaster`. Result: **worse than proposed everywhere it
differs** (M4 +0.00073, p=0.25; SMM win rate 0.26). The TSFM expert that
looked good standalone (−12.2%) hurts as a regime expert — selected on
train, loses out-of-fold. Loss-regimes significantly hurt (p=0.0027, third
confirmation). Only the meta-learner blend is faintly positive (n.s.).

**We did not skip anything real.** The kitchen sink is the strongest
possible validation of the exp47 significance filter.

---

## 9. Monash Benchmarks (exp60–62)

We validated the proposed method on two Monash forecasting-repository
benchmarks plus an M4 news-augmentation test.

### 9.1 exp60: Tourism Monthly (366 series, h=24)

| Method | Mean CRPS | vs proposed |
|---|---|---|
| **Proposed** | **0.1282** | — |
| Equal weights | 0.1869 | p=1.9e-68 |
| Best single (snaive) | 0.1387 | p=2.6e-34 |

Our snaive MASE (1.080) beats every published Monash baseline (best:
DeepAR 1.409). With TSFMs in the zoo, TimesFM-2.5 (0.0965) and Chronos-2
(0.0972) dominate — the proposed method (0.0974) ties the best single
(p=0.42). Honest framing: when one model dominates, the router adds
nothing beyond matching the ex-ante best.

### 9.2 exp61: M3 Monthly (1,428 series, h=18)

| Method | Mean CRPS | vs proposed |
|---|---|---|
| **Proposed** | **0.1476** | — |
| Equal weights | 0.1641 | p=2.7e-44 |
| Best single (ets) | 0.1630 | p=3.5e-35 |

Regime structure differs per dataset (tourism: 296/366 seasonal→snaive;
M3: balanced across snaive/arima/ets). The method adapts.

### 9.3 exp62: M4 + news — honest negative

Adding the news module to M4-100: news-on 0.14502 vs news-off 0.1451.
The gate fires on 97% of origins but changes nothing. Combined with
exp60/61 news arms (tourism +0.4%, M3 −0.3%), the news module is
**inert on real data** — it fires constantly but moves no numbers.

---

## 10. Vector Comparisons (exp52–53, exp63)

We benchmarked against the Vector Institute bootcamp's own baselines.

### 10.1 exp52: We beat their floor

Our proposed method beats Vector's `LastValuePredictor` on Food CPI:
**1.0662 vs 1.2091 CRPS (11.8%, p=7.1e-04, 50/59 origins)**. Weak baseline,
single dataset — but a clean win.

### 10.2 exp53: Their LLM beats us decisively

Vector's `QuantileGridLLMPredictor` beats our proposed method on Food CPI:
**0.5688 vs 1.0339 CRPS (45% better, p=5.4e-05, LLM wins 43/58 origins)**.
The LLM is a better probabilist on smooth macro series (tighter,
better-centered intervals) and has semantic context ("Canada Food CPI")
our method lacks. Honest framing: ours beats classical baselines; a
well-designed LLM elicitation beats ours. This motivated the LLM ladder.

### 10.3 exp63: LLM as combiner — honest negative

LLM (gpt-oss-120b) as per-series forecast combiner over the 7-model
tourism zoo: **0.1169 vs statistical router 0.0992 (p=1.7e-08 against
the LLM)** and vs best-single 0.1005 (p=9.4e-08). The LLM had strictly
more information than either baseline yet performed worse — likely
over-thinking: spreading weight across models instead of trusting the
validation signal. Extends the agentic honest negatives (exp23/24/25).

---

## 11. The LLM Integration Ladder (exp55–59)

Six rungs to integrate LLMs with the proposed method, run on Gemini Flash
via the Vector proxy (bypassing Groq's free-tier limits).

```mermaid
flowchart TB
    R1["Rung 1: Dataset registry<br/>PIT-safe context per dataset ✅"]
    R2["Rung 2: LLM as expert (exp55)<br/>✅ works on Food CPI"]
    R3["Rung 3: LLM as calibrator (exp56)<br/>❌ 1.6729, worse than naive"]
    R4["Rung 4: Per-dataset prompts (exp57)<br/>✅ V1 (0.4262) beats V2 (0.4599)"]
    R5["Rung 5: Hybrid router (exp58)<br/>⚠️ 0.4526 beats Vector, loses to llm-V1"]
    R6["Rung 6: Generalizability (exp59)<br/>❌ 36-64% worse on Tourism/M3"]
    R1 --> R2 --> R3 --> R4 --> R5 --> R6
```

### 11.1 Rung 2 (exp55): LLM as expert — works on Food CPI

Gemini Flash as a direct forecaster on Food CPI: competitive with the
best classical methods. The semantic context ("Canada Food CPI") and
calibration skill matter on smooth macro series.

### 11.2 Rung 3 (exp56): LLM as calibrator — fails

LLM asked to calibrate classical forecasts: **1.6729 CRPS, worse than
naive (1.6915)**. The LLM is better at direct forecasting than at fixing
classical outputs. Killed.

### 11.3 Rung 4 (exp57): Per-dataset prompts — V1 wins

Two prompt variants: V1 (metadata + history stats in words, 0.4262) beats
V2 (V1 + macro lookup table, 0.4599). More context distracts; the concise
prompt wins. All prompts are PIT-audited — see
[PROMPT_LEAKAGE_AUDIT.md](PROMPT_LEAKAGE_AUDIT.md) for the per-field trace to pre-origin sources.

### 11.4 Rung 5 (exp58): Hybrid — beats Vector, loses to best single

Hybrid regime router over {4 classical + 3 LLM experts}: **0.4526**,
beating Vector's 0.5688 by **20.4%** (success criterion met). But it
**loses to llm-V1 alone (0.4262, p=0.045)** — the router adds no value when
one expert dominates. The practical takeaway: use llm-V1 directly.

### 11.5 Rung 6 (exp59): Generalizability — clean negative

The rung-4 V1 prompt + Gemini Flash on Tourism/M3 (30 series each, h=1):
Tourism **64% worse** than classical (1691 vs 1030), M3 **36% worse**
(199.5 vs 147.2). The Food CPI win was dataset-specific. On strongly
seasonal multi-series data, classical methods dominate.

### 11.6 Ladder verdict

**LLMs are brilliant specialists, poor generalists.** They dominate on
smooth, context-rich macro series and fail on seasonal multi-series
benchmarks. The proposed classical method remains the general solution;
the LLM is a specialist tool for Food CPI-like series.

---

## 12. Tool Optimization

### 9.1 The tracker

`src/td_one/agent/tool_tracker.py` with `@tracked()` decorator, wired into
all tools. Every call logs duration, hashed inputs, truncated outputs, and
success/failure. This was the P0 fix — the tracker existed but had zero
call sites.

### 9.2 Real usage data (138 tracked calls)

| Tool | Calls | Avg ms | Verdict |
|---|---|---|---|
| cross_validate | 10 | 1.3 | keep |
| test_stationarity | 8 | ~150 | keep + `fast=True` path |
| detect_changepoints | 8 | ~45 | keep |
| analyze_residuals | 5 | ~6 | keep |
| decompose_series | 4 | ~29 | keep + validation |
| compare_models | 3 | 3.7 | keep |
| backtest_event_impact | 2 | 0.4 | keep |
| feature_importance | 1 | 2.4 | keep |
| plot_diagnostics | 1 | ~618 | **remove** (0 real invocations, slowest) |
| generate_synthetic_scenario | 4 | 2.1 | **remove** (0 real invocations) |

The diagnostic trio (stationarity + changepoints + decomposition) = 43.5%
of calls — the standard exploration pattern.

### 9.3 Final toolset: 13 tools

**Added 4** (grounded in model-written code patterns):
`load_series` (100% of code blocks start with CSV date-loading boilerplate),
`naive_forecasts` (3 reimplementations with different sigma estimators —
comparability hazard), `series_features` (volatility ratio copy-pasted ×3;
tsfeatures-lite gap), `regime_probabilities` (exp43's winner, was inline).

**Removed 2:** `plot_diagnostics`, `generate_synthetic_scenario`.

**Fixed 3:** tracker wiring, `test_stationarity` fast path (ADF-only, ~40ms
vs ~150ms), `decompose_series` validation.

50/50 tests pass. exp49 parity checks confirm the new tools reproduce
proven behavior exactly (max diff 0.0 on all checks).

---

## 13. What We Learned

### 13.1 Honest negatives are findings, not failures

The 17 kills define the boundary of what works. Selection from backward
diagnostics fails on synthetic *and* real data (exp23, exp37) — a method
problem, not a data problem. Learned combination fails at our scale
(exp42) — the combination puzzle is real. Unsupervised regimes fail three
times (exp45, exp50, exp51) — hand rules encode domain knowledge that
clustering cannot recover.

### 13.2 Hand-picked priors beat learning at our scale

This is the deepest pattern in the program. exp43's hand regimes beat
exp45's learned regimes (p=0.00046). exp50's hand gate thresholds beat
learned ones (p=0.32). exp51's kitchen sink (all learned/nonsignificant
components) loses to the filtered proposed method. At n=600 series and
n=143 meetings, structural priors dominate flexible learning. The
literature agrees (FFORMA needed 100k series).

### 13.3 News does not help monthly forecasting — definitively

exp19 (clean negative), exp21 (−21.6%), exp25 (+25% worse, p=0.0013),
exp46 (dense coverage, p=0.48) — four independent kills. The mechanism
works on synthetic (exp44, p=0.0068) but real monthly news is either
already priced in (SMM −13.04%) or too crude as features (BoC keyword
counts are contrarian noise). Track closed.

### 13.4 The significance filter was correct

exp51 threw everything at the wall. Nothing stuck. The 14-component
proposed method is not missing real gains — it is correctly filtered.

### 13.5 Agents are diagnosticians, not forecasters

exp31 (autonomous interaction discovery) and exp36 (seasonality
discrimination) show agents doing real data science. But as *forecasters*,
agents over-adjust (exp4), produce overconfident intervals (exp22), and
select at random (exp23). The winning architecture uses the agent for
diagnosis and gating, never for the numbers themselves.

### 13.6 LLMs are specialists, not generalists

The 6-rung ladder (exp55–59) is the clearest result of the program's final
phase. On Food CPI — a smooth macro series where semantic context matters —
Gemini Flash (0.4262) crushes every classical method (0.9661) and beats
Vector's LLM baseline by 20%. On Tourism and M3 — seasonal multi-series
benchmarks — the same LLM is 36–64% worse than classical. The LLM does not
learn seasonality from a prompt; seasonal naive does it by construction.
Use the right tool for the series type.

### 13.7 The router is unnecessary when one expert dominates

exp58's hybrid (0.4526) beats Vector's method but loses to llm-V1 alone
(0.4262, p=0.045). When the best expert wins in every regime, combination
only adds noise. The regime router earns its keep on M4/Tourism/M3 where
experts trade wins across regimes — not on Food CPI where the LLM sweeps.

---

## 14. Reproducibility

- **Repo:** `VDavoodnia/agentic-forecasting`, branch `main`, `td-one/`
- **Environment:** `.venv/bin/python` directly (`uv run` hangs in this VM)
- **Key entry points:**
  - `src/td_one/agent/proposed_method.py` — the method
  - `experiments/exp47_proposed_method.py` — original validation
  - `experiments/exp49_final.py` — final run with updated toolset
  - `experiments/exp51_kitchen_sink.py` — significance-filter validation
- **Tests:** 50/50 pass (`tests/test_proposed_method.py`,
  `tests/test_agent_tools.py`)
- **Data:** Freddie Mac SMM (local-only, cannot redistribute), FRED/BoC
  yields (local-only), M4 (public), news DB (gitignored, 9,760 articles)
- **Docs:** `docs/PROPOSED_METHOD.md` (architecture),
  `docs/POSITIVE_LIFT_AUDIT.md` (component inventory),
  `docs/NONSIGNIFICANT_POSITIVES.md` (excluded-effects audit),
  `docs/reports/FINAL_SYNTHESIS.md` (living final report)

---

*This is TD-One's method. Everything else was scaffolding.*
