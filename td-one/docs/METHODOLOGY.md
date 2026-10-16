# Agentic Forecasting for Mortgage Attrition: Methodology

**TD-One — Canadian mortgage renewal/switching forecasting**
*Methodology reference for the modeling crew. Written 2026-10-16; results
as of the latest runs in `experiments/`.*

---

## Abstract

We forecast **borrower attrition at mortgage renewal**: for each portfolio
segment and each month, how many renewing borrowers will *switch* lenders
vs stay? We compare four rungs of a modeling ladder — a Bayesian baseline
(M0), zero-shot time-series foundation models (M1: Chronos-2, TimesFM-2.5),
a ReAct LLM agent that plans its own analysis with tools (M2), and a
hybrid that lets the agent nudge the baseline inside strict bounds (M3) —
under a point-in-time evaluation protocol with CRPS as the primary metric.
On 10,000 synthetic customers × 66 months, the real-weight TSFMs beat M0
on mean CRPS (Chronos-2 4.46, TimesFM-2.5 4.59 vs M0 5.36); TimesFM
lands on nominal 80% coverage (mean 0.80 across segments). A Groq-driven ReAct agent finalized 2 of 3
pilot traces and beat M0 on one. We retract an earlier claim that
Chronos-2 beats M0 by 20% — it was a circular-DGP artifact caught by our
critic audit. Calibration remains the open problem.

---

## 1. Introduction

When a fixed-rate mortgage matures in Canada, the borrower faces a
choice: renew with the incumbent lender or switch. For a bank's treasury
and retention teams, the *switching rate* per segment per month drives
funding plans, pricing, and campaign targeting. The series is awkward:
monthly counts are small, renewal volumes are seasonal, rate shocks arrive
as regime shifts, and the signal that matters (a Bank of Canada decision,
a competitor's rate war) lives in text, not in the time series.

Classical approaches fit a rate model per segment and call it a day.
Foundation models for time series (TSFMs) promise zero-shot forecasting
without per-series training. LLM *agents* promise something else: the
ability to plan an analysis — pull the segment history, read macro
evidence, run a numerical forecast, and synthesize — the way an analyst
would. This project tests all three ideas against a strong Bayesian
baseline, honestly.

**Contributions.** (1) A four-rung ladder (M0–M3) with a shared
point-in-time evaluation harness. (2) A ReAct agent with injected,
cutoff-safe tools and a frozen evidence corpus. (3) A bounded hybrid
readout (M3) that converts LLM judgment into calibrated adjustments.
(4) A retraction: our first "Chronos-2 beats M0" result did not survive
a fair experimental redesign, and we document exactly why.

---

## 2. Related work

**Time-series foundation models.** Chronos-2 (Amazon) and TimesFM-2.5
(Google) are pre-trained on large time-series corpora and forecast
zero-shot via quantile heads. We use them strictly as *univariate*
forecasters over monthly switching-rate series — no fine-tuning, no
covariates. Prior work shows strong zero-shot performance on standard
benchmarks; our question is narrower: do they beat a domain baseline on
short, noisy, regime-shifting count panels?

**LLM agents for analysis.** ReAct (Yao et al., 2023) interleaves
*reasoning* traces with *actions* (tool calls), letting the model plan,
observe, and replan. We adopt the pattern but replace free-form
web/API tools with three injected, audited tools (Section 6.5). The
agent never touches raw customer data — only segment aggregates —
and every tool enforces the point-in-time cutoff.

**Hybrid judgmental-statistical forecasting.** The forecasting
literature has long combined statistical baselines with judgmental
adjustment; the failure mode is *over*-adjustment. M3 operationalizes
the lesson as a hard bound: the agent's logit adjustment is clipped to
±0.75, so the worst case is a bounded perturbation of M0, never a
replacement.

---

## 3. Problem formulation

For segment $s$ and month $t$, let $N_{s,t}$ be renewals coming due and
$A_{s,t}$ the number who switch. We forecast the *rate*
$r_{s,t} = A_{s,t}/N_{s,t}$ one month ahead as a **quantile lattice**
$\{q_\alpha\}_{\alpha \in \mathcal{L}}$ (model-native grids: 21 levels
for Chronos-2, 9 for TimesFM), evaluated as count distributions.

**Point-in-time (PIT) discipline.** Every forecast for origin $o$ may
use only information with `observed_asof ≤ o`. Two mechanisms enforce
this: (1) the panel builder masks training months to outcomes with
`outcome_observed_at ≤ origin` (settlement-lag leakage is structurally
impossible); (2) evidence retrieval filters the corpus to
`published_date < run_date`. A leakage auditor verifies 0% lookahead.

**Metrics.** CRPS on count distributions is primary (proper score);
MAE on expected counts and empirical 80% interval coverage are
secondary. Origins are consecutive month-ends (rolling evaluation).

---

## 4. Data

We train and evaluate on a **structural simulator** (10,000 customers ×
66 months, seed-fixed): origination months drawn year-round
(maturities spread across the calendar), payment-shock quintiles,
fixed/variable rate type, province region, and term buckets, with
planted macro shocks (rate jumps, OSFI-style tightening). Outcomes are
generated by a structural DGP with Beta-Binomial overdispersion and
AR(1) logit noise — deliberately *not* hand-tuned to flatter any
model (see the exp2 v1 retraction, Section 8).

**Segments.** Fifteen panels are discovered per origin: payment-shock
quintiles, rate type, province regions, term buckets. Each panel is a
monthly `(n_renewals, n_switches)` series.

**Macro evidence.** Two sources: (1) a **frozen 12-entry illustrative
corpus** (`experiments/evidence_corpus/corpus.json`) with BoC-style
announcements, date-guarded at the run date; (2) **live FRED** — 17
curated series (policy rates, Canada 3M/10Y, US 5Y/10Y, CPI, unemployment,
house prices, CAD/USD, credit spreads) bulk-downloaded to parquet and
PIT-filtered at use time.

```mermaid
flowchart LR
    SIM["Simulator<br/>(10k × 66)"] --> PIT["PIT store<br/>(DuckDB)"]
    PIT --> SEG["Segment discovery<br/>(15 panels)"]
    SEG --> M0["M0 baseline"]
    SEG --> M1["M1 TSFMs"]
    SEG --> AG["M2 ReAct agent"]
    FRED[("FRED macro<br/>17 series")] --> AG
    CORP[("Frozen corpus<br/>12 docs")] --> AG
    M0 --> HYB["M3 hybrid<br/>(bounded Δ)"]
    AG --> HYB
    M0 --> EV["Backtest<br/>(CRPS/MAE/cov)"]
    M1 --> EV
    AG --> EV
    HYB --> EV
```

---

## 5. The ladder

### 5.1 M0 — Jeffreys Beta-Binomial baseline

The incumbent. For each segment, fit a Beta-Binomial on the trailing 6
months with a Jeffreys prior (α=β=0.5): the posterior predictive for
next month's switches is the forecast distribution. Simple, calibrated
by construction on stationary stretches, and deliberately hard to beat
— any rung above it must earn its complexity.

### 5.2 M1 — zero-shot TSFMs

The panel's historical switching *rates* feed Chronos-2 or TimesFM-2.5
(zero-shot, `horizon=1`) on their native quantile grids. Two
engineering notes: (1) neural forecasters can emit **quantile
crossing** — we sort values ascending to enforce monotonicity
(regression-tested); (2) weights load once per process from a pinned
manifest and are explicitly unloaded between runs (RAM discipline on
small VMs).

### 5.3 M2 — the ReAct agent *(agentic primer)*

> **For readers new to agents.** A *ReAct* agent is a loop, not a
> model. At each step the LLM sees the task plus the trace so far and
> emits exactly one JSON action: call a *tool* (a typed function we
> inject — it cannot call anything else) or `finalize` with its
> forecast. The tool returns an *observation* (text), which is appended
> to the trace, and the loop continues — up to 6 steps. "Agentic" here
> means the LLM **plans its own analysis sequence** (e.g. pull data →
> read evidence → forecast → finalize) rather than executing a fixed
> pipeline. The intelligence under test is orchestration: does choosing
> what to look at, and in what order, beat the fixed baseline?

```mermaid
flowchart TD
    T["Task + run_date<br/>(PIT cutoff)"] --> LLM{"LLM:<br/>one JSON action"}
    LLM -->|"data_segment"| DS["Segment history<br/>(cutoff-safe)"]
    LLM -->|"evidence"| EV2["Macro evidence<br/>(date-filtered)"]
    LLM -->|"forecasting"| FC["Numerical forecast<br/>(M0 backend)"]
    LLM -->|"finalize"| OUT(["Rate quantiles"])
    DS --> TR["Trace append"]
    EV2 --> TR
    FC --> TR
    TR --> LLM
```

Concretely: `ReActAgent` drives `GroqChatClient`
(`openai/gpt-oss-120b` via a vault surrogate — the key never enters the
process) with a per-run call budget, bounded steps, malformed-JSON
recovery, and an AI2-grade retry policy (exponential backoff + jitter,
Retry-After honored, 60s floor on 429s, 2.5s process-wide throttle).
The evaluated M2 forecast is the agent's finalized quantile lattice.

### 5.4 M3 — bounded hybrid

M3 = M0's numerical posterior + the agent's judgment, combined as a
**logit adjustment** $\Delta \in [-0.75, +0.75]$ proposed per macro
scenario and applied through a hybrid readout. The bound is the point:
the agent can tilt M0 toward a rate shock or an OSFI tightening, but it
cannot replace the baseline — the worst case is a bounded perturbation.
This directly operationalizes the judgmental-forecasting lesson that
unbounded adjustment destroys calibration.

---

## 6. Experiments

| Exp | Question | Design | Headline |
|---|---|---|---|
| exp1 | Does the pipeline work? | 500×24 synthetic, M0 vs mock M1 | ✅ Plumbing validated (mocks only) |
| exp2 v2 | Do real TSFMs beat M0? | 3 pre-registered scenarios × 5 seeds, CRPS primary, oracle anchor | No model significantly beats M0 (all p > 0.18); TimesFM best mean CRPS n.s. |
| exp3 | Can the ReAct agent beat M0? | 3 pilot segments; mock vs real Groq | Mock: plumbing only. Real: 1/2 finalized segments beat M0 |
| exp4 | Does bounded agent adjustment help? | 6 macro scenarios, real Groq deltas | 3/6 beat M0; LLM directional but over-adjusts |
| scale-up | At full scale? | 10k×66, 14 seg × 6 origins | Real weights: Chronos-2 4.46, TimesFM 4.59 vs M0 5.36 |

### 6.1 Experiment 2 (the one that matters for M1)

v1 claimed "Chronos-2 beats M0 by 20%." The critic audit found three
fatal flaws: a circular DGP hand-tuned to flatter trend-extrapolating
TSFMs, n=6 test points on a single seed, and an M0 that didn't match
its own spec. **We retracted v1.** v2 redesigns from scratch:

| Scenario | Oracle | M0 | Chronos-2 | TimesFM-2.5 |
|---|---|---|---|---|
| stationary | 4.65 ± 1.27 | 4.85 ± 1.53 | 5.40 ± 2.18 | **4.28 ± 0.65** |
| drift | 5.02 ± 1.17 | 5.23 ± 1.34 | 6.04 ± 1.89 | **5.17 ± 0.80** |
| regime_shift | 5.58 ± 1.37 | 7.40 ± 1.77 | 7.44 ± 2.13 | **7.09 ± 1.48** |

CRPS mean ± std across 5 seeds. Paired Wilcoxon M1−M0: all p > 0.18 —
**not significant**. TimesFM-2.5 is the best of the three in every
scenario but the gaps don't clear noise. Coverage: Chronos-2
overcovers, M0 undercovers, TimesFM closest to nominal.

### 6.2 Experiment 3 (M2, real Groq)

| Segment | M0 CRPS | M2 CRPS | M2 steps |
|---|---|---|---|
| seg-0 | 0.76 | 5.22 | 6 (finalized, worse) |
| seg-1 | 1.33 | **0.82** | 5 (finalized, **win**) |
| seg-2 | 1.40 | — | 6 (never finalized) |

First real-LLM win on seg-1 (MAE 0.14 vs 2.00). The patient retry
policy (60s floor on 429s) was the difference between 1/3 and 2/3
finalized traces — free-tier rate limits, not model skill, were the
binding constraint. ⚠️ Honesty note: real-mode M2 still uses the
`timesfm_mock` numerical backend, so the seg-1 edge cannot be separated
from the mock sharpness artifact — the run demonstrates orchestration
(finalization), not numerical superiority. This was the third real-mode
attempt (attempts 1–2 were rate-limit-bound).

### 6.3 Experiment 4 (M3, real Groq)

| Scenario | Δ (LLM) | M0 CRPS | M3 CRPS |
|---|---|---|---|
| rate_shock_up | +0.18 | 19.87 | **16.33** |
| rate_shock_down | −0.25 | 4.18 | 5.01 |
| osfi_tightening | +0.30 | 9.95 | 15.22 |
| benign | +0.30 | 3.78 | 7.42 |
| family_squeeze | +0.20 | 14.60 | **7.96** |
| competitor_rate_war | +0.28 | 14.57 | **10.65** |

3/6 beat M0. The LLM's deltas are directionally sensible (negative on
the rate-drop scenario) but it **over-adjusts**: +0.30 on the benign
scenario, where the right answer is ≈ 0, is pure noise injection. Worst
failure: `osfi_tightening` got the **sign wrong** (+0.30 vs true ≈
−0.35 — the agent misread a stress-test policy as pro-switching).

### 6.4 Scale-up (10k × 66, real weights)

| Model | Mean CRPS | Mean coverage_80 |
|---|---|---|
| M0 | 5.360 | 0.671 |
| Chronos-2 (real) | 4.463 | ~0.93 (overcovers) |
| TimesFM-2.5 (real) | 4.591 | **0.800** (nominal) |
| chronos2_mock | 3.194 | 0.714 |
| timesfm_mock | 3.176 | 0.714 |

⚠️ The `*_mock` rows are **interface-only plumbing** — their CRPS edge
is a sharpness artifact (tight spreads around the last rate), not
evidence about the real models. The real-weight rows are the honest
comparison: both TSFMs beat M0 on mean CRPS at this scale and DGP, and
TimesFM lands exactly on nominal coverage. But paired tests on the 14
shared segments find **no significant difference** (Chronos-2 wins
7/14, p=0.212/0.583; TimesFM-2.5 wins 5/14, p=0.341/0.583) — exp2 v2's
conclusion reproduces at scale. Single seed, single DGP: any future
superiority claim must be pre-registered across seeds/DGPs.

---

## 7. Discussion and limitations

1. **Everything is synthetic.** The simulator is realistic but the
   DGP is ours; real portfolio data is the obvious next validation.
2. **Calibration is open.** M0 undercovers at scale; Chronos-2
   overcovers; only TimesFM is near nominal. Interval calibration —
   not point accuracy — is the next modeling task.
3. **The agent is rate-limit-bound, not skill-bound**, on the free
   tier. Paid-tier or local inference would separate the two.
4. **M3's over-adjustment** suggests the ±0.75 bound is necessary but
   not sufficient — the agent needs a "do nothing" prior for benign
   regimes.
5. **Scale-up is single-seed, single-DGP** — the paired
   non-significance guards against overclaim, but a multi-seed,
   pre-registered comparison is the next required step.

---

## 8. Conclusion

A Bayesian baseline remains stubbornly competitive: neither TSFMs nor
LLM agents beat it significantly in designed comparisons, though both
show real signal (TimesFM's mean CRPS edge, the agent's first ReAct
win, M3's 3/6 macro scenarios). The honest contribution is the
machinery — a PIT-clean ladder, a tool-injected agent with bounded
hybrid readout, and a documented retraction — more than any single
leaderboard number.

## References

- Yao et al. *ReAct: Synergizing Reasoning and Acting in Language
  Models.* ICLR 2023.
- Ansari et al. *Chronos-2.* Amazon, 2025.
- Das et al. *TimesFM-2.5.* Google, 2025.
- FRED, Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org
- `experiments/REPRODUCIBILITY.md` — exact reproduction steps.
- `docs/CRITIC_SYNTHESIS.md` — the audit that retracted exp2 v1.
