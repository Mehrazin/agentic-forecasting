# TD-One Manager Report

**Agentic Forecasting for Canadian Mortgage Attrition**
*Status: 2026-10-17. For colleagues new to the project.*

---

## 1. Executive Summary

**What is TD-One?** A research program asking a practical question: can modern
forecasting methods — time-series foundation models (TSFMs), LLM agents, and
news-aware models — beat a strong Bayesian baseline at predicting **borrower
attrition when Canadian mortgages renew**? We built a four-rung modeling ladder
(M0–M3, later M4/M5), a point-in-time evaluation harness with CRPS as the
primary metric, and tested everything on both synthetic data (10,000 customers
× 66 months) and real data (63 months of Freddie Mac single-monthly-mortality
rates, 2,000 M4 Monthly series).

**Headline result:** On 63 months of real Freddie Mac SMM data (50 rolling
origins, all models on a common 19-level quantile grid), **calibrated TimesFM-2.5
beats the best fairly-constructed classical baseline by 17.8% CRPS**
(0.000497 vs 0.000605, paired t = −2.77, p = 0.0079, survives Bonferroni
correction, wins 37/50 origins). Chronos-2 confirms the win independently at
16–17%. This is the first statistically robust TSFM victory in the program —
and it survived a deliberate attempt to kill it (see §5).

**Honest negatives:** Three agentic tracks were killed with evidence (M2 ReAct,
M3 hybrid, M5 router). A news/RAG regime signal failed to improve SMM forecasts
(exp19, p = 0.85). The macro-overlay track is exploratory, not confirmatory
(single seed, window-confounded). We retracted three of our own claims this
month — details in §5. The retractions are the point: this program is designed
to find out what's true, not to confirm a thesis.

---

## 2. Problem

When a Canadian fixed-rate mortgage matures, the borrower chooses: renew with
the incumbent lender or switch. For a bank's treasury and retention teams, the
**switching rate per segment per month** drives funding plans, pricing, and
campaign targeting.

The series is awkward to forecast. Monthly counts are small. Renewal volumes
are seasonal. Rate shocks arrive as regime shifts — and the signal that matters
(a Bank of Canada decision, a competitor's rate war) lives in **text**, not in
the time series. Classical approaches fit a rate model per segment and stop.
We asked whether TSFMs (zero-shot, no per-series training) or LLM agents (plan
an analysis the way an analyst would: pull history, read macro evidence,
synthesize) could do better.

We forecast two related targets:
- **Switching rates** on synthetic loan panels (the M0–M5 ladder).
- **SMM (single monthly mortality)** — the voluntary prepayment rate — on real
  Freddie Mac loan-level data, 63 months from January 2021 to March 2026
  (7.3M loans at risk by the end). SMM is the real-data analogue of the
  switching problem: same monthly-rate structure, same regime shifts (the 2022
  rate-hike cycle crushed SMM; the 2024–2026 easing lifted it).

---

## 3. Approach

**The ladder (synthetic data).** M0 is a Jeffreys Beta-Binomial baseline on a
trailing window — deliberately strong, not a strawman. M1 wraps Chronos-2 and
TimesFM-2.5 as zero-shot univariate forecasters. M2 is a ReAct LLM agent with
three injected, cutoff-safe tools (it never touches raw customer data). M3 is
a hybrid: the agent nudges M0's forecast, but the adjustment is hard-bounded
to ±0.75 in logit space, so the worst case is a bounded perturbation, never a
replacement. M4 is a regime reader (LLM classifies UP/DOWN/NONE/ABSTAIN from
news via RAG) and M5 a router that picks models per origin.

**Point-in-time discipline.** Every forecast for origin *o* may use only
information with `observed_asof ≤ o`. A panel builder masks training months;
news retrieval filters to `pub_date < run_date`; a leakage auditor verifies
0% lookahead. Two external audits (Google Argon, two passes) checked this
discipline and found real bugs — which we fixed and reran.

**The pivot to real data.** Synthetic DGPs can flatter their builders (we
caught ourselves doing it — §5). So we moved the decisive comparisons to real
data: Freddie Mac SMM (exp16/exp18/exp18b/exp19) and 2,000 M4 Monthly series
(exp14/R2-4). A `LogScaleCalibrator` was built to fix a genuine problem:
TSFM predictive spreads are calibrated in absolute terms, so on near-zero
series like SMM (0.0003–0.003) they blow up ~29× vs a trailing mean; forecasting
in log-space and exponentiating quantiles back fixes it in a principled way
(quantiles are preserved under monotonic transforms).

---

## 4. Key Results

### 4.1 exp18b — TSFMs beat classical on real SMM (the headline)

Fair ablation on 63-month Freddie Mac SMM, 50 rolling origins, **all 11
model×wrapper cells on one common 19-level quantile grid** (0.05–0.95):

| Cell | Mean CRPS | Mean MAE | 80% coverage |
|---|---|---|---|
| **timesfm25 + log-calibration** | **0.000497** | 0.000644 | 0.84 |
| **chronos2 + log-calibration** | **0.000507** | 0.000662 | 0.88 |
| ewm + residual quantiles (best classical) | 0.000605 | 0.000795 | 0.78 |
| m0_w3 + residual quantiles | 0.000613 | 0.000729 | 0.82 |
| m0_w3 + log-calibration | 0.000628 | 0.000736 | 0.82 |

TimesFM beats the best classical cell by **17.8%** (19.0% vs m0_w3);
paired t = −2.77, **p = 0.0079** (survives Bonferroni over 4 comparisons),
37/50 paired wins. Chronos-2 independently confirms at ~17% (t = −2.36,
35/50). Both TSFMs also lead on point accuracy (MAE), so the edge is real
forecast skill plus better-calibrated spreads — not a wrapper artifact.
Notably, giving the classical models the same log-calibration wrapper did
*not* close the gap (m0_w3 + logcal = 0.000628, worse than residual
quantiles): the wrapper fixes TSFM spread miscalibration but classical
residuals were already in the right scale.

*Limitations:* no ETS/ARIMA baseline yet (not in repo — in progress); single
TSFM seed (42); 50 origins on one series.

### 4.2 R2-4 — TSFM win survives on M4 real data

Rerun of exp14 with **all models on the same 0.10–0.90 quantile grid**
(soundness fix; the original used mixed grids). 2,000 M4 Monthly series,
6-step horizon: naive 462.8, **Chronos-2 351.2 (−24.1%)**,
**TimesFM 358.1 (−22.6%)**. The win is not a scoring artifact.

### 4.3 R2 macro — exploratory, method-specific

After fixing a dict-lookup bug that had zeroed the macro signal (Argon
Finding 1), the lagged PIT-safe rerun shows:
- **Classical:** macro_m0 1.923 vs m0_w6 2.280 (−0.357, ~53% of the leaky
  gain). *But:* single seed; m0_w3 (1.457) still beats macro_m0 by 32%; the
  gain may be window-confounding (3-month z[t] compensating a stale 6-month
  average) rather than true macro value. Needs macro_m0 with window=3 across
  seeds 0–4.
- **TSFM residual conditioning:** 1/6 helps, 4/6 hurt, 1/6 tie (sign test
  p = 0.375, n.s.). The residual approach — subtracting β·z in training,
  re-adding at forecast — distorts the input distribution. Direct logit
  shift works; residual conditioning doesn't. Method failure, not macro
  failure.

Verdict: **we don't know yet.** Suggestive, not confirmatory.

### 4.4 exp19 — news/RAG does not help SMM (clean negative)

Wired the news regime reader (50 Groq calls, PIT-safe RAG over 9,581
articles) into the SMM backtest: m0_w3 + regime vs m0_w3 gives
ΔCRPS = −7.1e-06, **t = −0.19, p = 0.85** (6/5/39 W/L/T). No confidence-gate
setting yields significance. Directional accuracy 0.64 (25/39, binomial
p ≈ 0.11, n.s.). The reader mostly abstained or reported low confidence
(Canadian-heavy KB for a US series; thin 2022–2024 coverage). Per Argon's
advice, the agentic track is frozen; the M4 regime work stands as a
structured taxonomy of failure modes, not a live track.

### 4.5 Killed tracks (with evidence)

- **M2 (ReAct forecaster):** honest negative on real data; the one apparent
  win couldn't be separated from a mock artifact.
- **M3 (hybrid):** LLM got a sign wrong on a key variable (+0.30 vs true
  −0.35) and systematically over-adjusted; the ±0.75 bound contained the
  damage but couldn't create value.
- **M5 (router):** v2 (critic-constrained) 2.488 still loses to dumb
  inverse-CRPS weighting (2.322). Dead.

---

## 5. What We Got Wrong

Three retractions this month. Each made the program stronger.

1. **"The macro track is dead" → retracted.** The R2 "PIT-safe kills macro"
   verdict was based on a bug: `fred_z.get(forecast_month)` looked up t+1 in
   a dict truncated at t, silently returning 0.0. The macro signal was never
   tested. The lagged rerun (§4.3) is exploratory-positive for classical.
2. **"TSFMs beat m0 by 30%" → corrected to ~18%.** exp18 scored point
   forecasts (CRPS = MAE) against quantile distributions — an apples-to-
   oranges comparison that inflated the win by ~12 points. exp18b (§4.1) is
   the fair number. Argon flagged the asymmetry; we confirmed it and reran.
3. **"OLS captures 93% of oracle macro gain" → retracted.** A wrong-sign
   artifact of the v1 DGP (exp15). Cited in three docs before we caught it;
   all three now carry the retraction.

We also retract an early synthetic result ("Chronos-2 beats M0 20%") as a
circular-DGP artifact. The pattern is deliberate: every claim gets a critic
pass, an external audit, and a rerun when the audit bites. Eight internal
critics (PIT, statistics, code, docs, reproducibility, adversarial ML,
security, presentation) ran this month; all findings were fixed or
documented.

---

## 6. What's Next

1. **Multi-seed macro** (highest priority): r2_exp10 on seeds 0–4 plus
   macro_m0 with trailing_window=3, to resolve the window-confounding
   question. Separates "macro helps" from "recency helps."
2. **ETS/ARIMA baseline** (in progress): the one gap Argon keeps flagging.
   "Beats trailing mean" is not "beats classical" — a competent ETS with
   log-calibration is the fairest classical challenger to the 17.8%.
3. **Agentic track frozen.** exp19 was the clean negative. Per Argon's
   advice, no more compute goes to making the LLM win on CRPS; the M4/M5
   post-mortems stand as the deliverable.
4. **Remaining audit items** (lower priority): segment-discovery cohort
   mining and known-N in M0 (Argon Finding 3); PITStore 46-column schema
   (Finding 4).

**Bottom line for the program:** the honest story is a 17.8% TSFM win on
real data with proper uncertainty quantification, a classical baseline that
refuses to die (m0_w3 + residual quantiles is well-calibrated at 0.82
coverage), an exploratory macro signal that needs more seeds, and a set of
clean agentic negatives. That's a solid POC — and the retractions are what
make it credible.

---

*Sources: `docs/RESULTS.md` (R2 registry), `docs/METHODOLOGY.md` (ladder design),
`docs/ARGON_REVIEW_2.md` (audit findings), `experiments/exp18b_fair_smm_results.json`,
`experiments/exp19_news_smm_results.json`, `experiments/r2_exp14_common_grid_results.json`.
All numbers recomputed from committed artifacts 2026-10-17. Reproduction commands:
`experiments/REPRODUCIBILITY.md`.*
