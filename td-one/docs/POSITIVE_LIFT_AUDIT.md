# Positive Lift Audit: exp1–exp46 + Benchmarks

**Date:** 2026-10-18
**Purpose:** Complete inventory of every positive result across the TD-One program, to select components for the proposed method (exp47).

---

## Audit table

| Exp | What was tested | Beat baseline? | Effect / p | Verdict |
|---|---|---|---|---|
| exp1 | Pipeline plumbing (mock) | n/a (infra) | — | **Keep** (infra) |
| exp2 | M0 vs TSFMs, synthetic, 5 seeds | No (n.s.) | TimesFM best mean, all p>0.18 | **Kill** claim (v1 retracted) |
| exp3 | M2 ReAct vs M0 (real Groq) | No | 2/3 lose w/ real backend; seg-1 win confounded by mock | **Kill** as forecaster |
| exp4 | M3 hybrid, real Groq, 6 scenarios | Mixed | 3/6 beat M0; direction right 4/6, magnitude wrong | **Marginal** (fixed by exp6 gate) |
| exp5 | Conformal baselines | **Yes (calibration)** | Coverage 0.671→0.914; nothing beats m0_w3 on CRPS | **Keep** conformal calibration |
| exp6 | M3+shrinkage / M3+gate | **Yes (gate)** | M3+gate 7.60 vs M0 7.86 held-out; 3TP/0FP/3TN/0FN | **Keep** "do nothing" prior |
| exp8 | Macro overlay (v1 DGP) | Apparent yes | 93% of oracle | **Kill** — RETRACTED (wrong sign) |
| exp9 | Simulator v2 | n/a (DGP) | Window ranking reverses (m0_w6 wins) | **Keep** (infra/DGP) |
| exp10 | Classical scaleup | **Yes (niche)** | Hierarchical +5–10% on thin segments; m0_w3 wins overall | **Keep** hierarchical niche; m0_w3 baseline |
| exp11 | M4 regime reader (real Groq) | **Yes (qualified)** | M4a 3.59 vs M0 4.82; fragile (1 confident error catastrophic) | **Keep** LLM-as-sensor w/ abstention |
| exp12 | M5 LLM router (real Groq) | No | 2.60 vs inv-CRPS 2.32 | **Kill** |
| exp13 | TSFM+macro residual | **Marginal** | +10–18% vs plain TSFM (3/3 seeds); beats baseline 1/3; classical macro_m0 still best | **Keep** mechanism only |
| exp14/R2-4 | M4 real, common grid | **Yes (strong)** | Chronos2 −24.1%, TimesFM −22.6% vs naive | **Keep** TSFMs for real data |
| exp15/R2-3 | Corrected-sign macro | No | Oracle ≈ 0; estimated overlay hurts (−0.15) | **Kill** |
| exp17 | News regime reader pilot | Marginal | 3/4 correct (n=6, n.s.) | **Marginal** (pilot only) |
| exp18 | SMM full (calibrated TSFMs) | Apparent yes | 30% vs m0 | **Superseded** by exp18b |
| exp18b | Fair SMM (ETS baseline) | **Yes (n.s.)** | TimesFM −12.2% vs ETS (0.000497 vs 0.000566, p=0.076, 32/50 wins) | **Keep** TSFM + LogScaleCalibrator |
| exp19 | News → SMM | No | ΔCRPS −7.1e-06, p=0.85 | **Kill** |
| exp20 | Event-augmented simulator | n/a (infra) | 5 event types, 84% news-leading | **Keep** (infra) |
| exp21 | Agent feature discovery | No | −21.6% held-out (features don't transfer) | **Kill** as method |
| exp22 | M2 v2 vs v1 | **Mixed** | +6.6% when active (6/9 wins); high variance, overconfident intervals | **Marginal** — keep code-exec agent + decision policy (needs calibration) |
| exp23 | Diagnostic model selection | No | = random (p=0.80); gap closed −6.3% | **Kill** |
| exp24 | Regime-aware selection | No | Ties best single (p=0.93) | **Kill** (idea evolved → exp43) |
| exp25 | News at h=3 | No | **Significantly worse** (+25%, p=0.0013) | **Kill** |
| exp30 | BoC MNLogit (4 feats) | No | 0.835 vs rule 0.843 | **Kill** MNLogit; **keep** simple rule |
| exp31 | Agent BoC discovery (no hints) | **Yes (method)** | Discovered inflation_gap × rate_momentum autonomously | **Keep** agent-as-diagnostician |
| exp32 | BoC EASY/HARD regime | **Yes** | +7.2pp gap (0.887 vs 0.815); rule adds value only on EASY | **Keep** regime-gated trust |
| exp33 | SMM agent features (real) | **Marginal** | +1.21% held-out (n.s., 6/12); +6–14% discovery CV | **Marginal** — keep agent feature discovery (directional) |
| exp34 | SMM real BoC events | No | −13.04% (hurts; regime confounding) | **Kill** |
| exp35 | Food CPI baseline | Mixed | Agent ties best (trail_mean 24/24 @h=1); h=6 negative (regime-fragile CV) | **Mixed** |
| exp36 | Food CPI categories | **Yes (diagnostic)** | Agent discriminates seasonality: produce 0.375 vs headline 0.000 | **Keep** diagnostic validation |
| exp37 | M4 selection (real) | No | −30% gap (worse than random) | **Kill** |
| exp40 | Calibrated DGP reruns | n/a (infra) | Fewer false discoveries (\|r\|<0.18 vs 0.43); transfer still fails | **Keep** calibrated DGP |
| exp41 | Learned selection | No | +0.9% gap (tied w/ best single); beats hand-rules but not a win | **Kill** (superseded → exp43) |
| exp42 | FFORMA-lite combination | No | p=0.156 vs equal weights | **Kill** |
| exp43 | Regime-probability router | **Yes** | M4 0.1282 vs 0.1295, **p=0.018** — first to beat equal weights | **KEEP — backbone** |
| exp44 | News-as-selector | **Marginal** | Mechanism proven synthetic (p=0.0068); muted real (sparse news) | **Marginal** — keep as gated module |
| exp45 | Learned regimes (unsupervised) | No | Hand wins (p=0.00046) | **Kill** learned; **keep** hand regimes |
| exp46 | News dense (definitive) | No | No help on any target/variant/dataset | **Kill** — news track closed (monthly) |
| WTI | Energy/oil benchmark | n/a | ETS/ARIMA/naive identical; agent correctly finds nothing 92% | **Keep** as no-signal diagnostic |

---

## Keep list (positive lift, ordered by strength)

1. **exp43 regime router** — M4 0.1282 vs 0.1295, p=0.018. First method to significantly beat equal weights. Constrained structural priors beat flexible learning. **→ BACKBONE of proposed method.**
2. **exp32 BoC regime gate** — EASY/HARD +7.2pp. Know when to trust the model. **→ Trust flag in proposed method.**
3. **exp14/R2-4 TSFM on real data** — Chronos2 −24.1%, TimesFM −22.6% vs naive on M4. **→ TSFM as expert option.**
4. **exp18b TSFM + LogScaleCalibrator** — 12.2% vs ETS on SMM (p=0.076, 32/50 wins). **→ Calibrated TSFM expert.**
5. **exp5/exp10 conformal** — Coverage 0.671→0.914. **→ Calibration wrapper.**
6. **exp6 M3 gate** — 7.60 vs 7.86 held-out; kills catastrophic adjustments. **→ "Do nothing" prior / corroboration gating.**
7. **exp11 M4a regime reader** — 3.59 vs 4.82 (real Groq); LLM-as-sensor with abstention. **→ Optional LLM sensor module.**
8. **exp31 agent discovery** — Autonomous interaction discovery. **→ Diagnostic mode.**
9. **exp22 M2 v2** — +6.6% when active (6/9 wins); code-executing agent. **→ Agent infrastructure (needs calibration).**
10. **exp44 news selector** — Mechanism proven synthetic (p=0.0068). **→ Gated optional module (dense news only).**
11. **exp33 agent features** — +1.21% held-out; sensible features. **→ Directional.**
12. **exp36 seasonality discrimination** — Agent correctly separates categories. **→ Diagnostic validation.**
13. **exp10 hierarchical** — +5–10% on thin segments. **→ Niche.**
14. **exp13 residual macro-conditioning** — +10–18% vs plain TSFM (mechanism). **→ Mechanism.**

## Kill list (definitive negatives)

exp2 (claim), exp3 (forecaster), exp8 (retracted), exp12, exp15, exp19, exp21, exp23, exp24, exp25, exp30-MNLogit, exp34, exp37, exp41, exp42, exp45-learned-regimes, exp46 (news monthly closed).

## Marginal / infrastructure

exp1, exp4 (fixed by exp6), exp9, exp17, exp20, exp35, exp40, WTI (no-signal diagnostic).

---

## What the proposed method (exp47) integrates

| Component | Source | Role |
|---|---|---|
| Regime-probability router | exp43 | Backbone: hand regimes → LR → per-regime experts → soft combine |
| Regime-gated trust flag | exp32 | EASY/HARD-style: report when model is trustworthy |
| "Do nothing" prior | exp6 + decision_policy | Corroboration gating before acting on regime shift |
| Conformal calibration | exp5/exp10 | Coverage correction on final quantiles |
| TSFM expert (optional) | exp14/exp18b | Add calibrated TSFM to expert zoo when available |
| LLM sensor (optional) | exp11 | Regime reading with abstention; never touches numbers |
| News modulation (gated) | exp44 | Adjust P(regime) only when news dense; off by default |
| Agent diagnostics | exp31/exp33 | Failure analysis mode, not forecast mode |
