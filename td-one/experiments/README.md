# TD-One Experiments

End-to-end results for the mortgage attrition forecasting ladder. Every
experiment is PIT-clean (no lookahead) and reproducible — see
`REPRODUCIBILITY.md` for the full reproduction steps.

## The ladder

| Rung | Model | What it is |
|---|---|---|
| M0 | Historical baseline | Trailing-6 Jeffreys Beta-Binomial on segment panels |
| M1 | Zero-shot TSFMs | Chronos-2 / TimesFM-2.5 on panel switching-rate series |
| M2 | ReAct LLM agent | Groq-driven tool loop (data_segment → evidence → forecasting) |
| M3 | Hybrid | M0 + bounded log-odds adjustment Δ ∈ [-0.75, +0.75] proposed by the agent |
| M4 | Regime reader | LLM reads news → regime label → model selection |
| M5 | Ensemble router | LLM routes among experts (REATS-style) |

Primary metric everywhere is **CRPS** (proper score on count distributions);
secondary: MAE and 80% interval coverage.

Verdict key: ✅ positive · ⚖️ partial/mixed · ❌ honest negative · 🔨 build/scaffolding · ▶️ running

## Experiment index

Numbering jumps from exp25 to exp40 (exp26–exp39 were never assigned).

| Exp | Question | Verdict | Report |
|---|---|---|---|
| exp1 | M0 vs M1 (mock TSFMs), 500×24 synthetic | ✅ M1a 3.26 / M1b 3.32 vs M0 3.50 MAE | [report](exp1_report.md) |
| exp2 | M0 vs M1 (real weights), 3 scenarios × 5 seeds | ✅ Chronos-2 MAE 4.00 (−20% vs M0); TimesFM-2.5 4.62 (−8%) | [report](exp2_report.md) |
| exp3 | M2 vs M0, 3 pilot segments, mock + real Groq | ⚖️ seg-1 win (0.82 vs 1.33), seg-0 worse, seg-2 never finalized | [report](exp3_report.md) |
| exp4 | M3 vs M0, 6 macro scenarios, mock + real Groq | ⚖️ 3/6 beat M0 — LLM directionally right but over-adjusts | [report](exp4_report.md) |
| exp5 | Better classical baselines vs M0 | 🔨 classical-track scaffolding | `exp5_results.json` |
| exp6 | M3 + shrinkage/gating vs M3/M0 | 🔨 decision policy built: James-Stein shrinkage + corroboration gate | `exp6_results.json` |
| exp7/7b | Regime-classifier hybrid (M3c), prototype + real-LLM test | 🔨 prototype → fed M4 regime-reader design | — |
| exp8 | Macro-grounded DGP: does macro structure reward macro-aware models? | ❌ macro track eventually closed (see exp10/13/15) | — |
| exp9 | Richer DGP (simulator v2): do conclusions change? | 🔨 DGP robustness check | — |
| exp10 | Classical track: promoted baselines under one roof | ❌ 0.678 CRPS gain was 100% z-score leak | `exp10_results.json` |
| exp11 | M4 regime-reader agent vs M0 | 🔨 led to regime-reader hardening (see `../docs/AGENT_AUDIT_M4.md`) | `exp11_results.json` |
| exp12/12b | M5 ensemble router v1, then critic-constrained v2 | ❌ v1 failed (see `../docs/AGENT_AUDIT_M5.md`); v2 tested | `exp12_results.json`, `exp12b_results.json` |
| exp13 | TSFM + FRED rematch — the fair fight | ❌ PIT-safe rerun reversed macro benefit to penalty | `exp13_tsfm_macro_results.json` |
| exp14 | M4 Monthly real-data benchmark | 🔨 real-data baseline for the agentic track | `exp14_m4_results.json` |
| exp15 | Macro overlay with corrected sign (negative lock-in) | ❌ neutralized — no evidence macro helps | `exp15_macro_corrected_results.json` |
| exp16 | Freddie Mac real-data SMM forecasting (PoC) | 🔨 real-data proof of concept | `exp16_freddie_smm_results.json` |
| exp17 | News-augmented regime reader, end-to-end | ⚖️ mechanism probe | `exp17_news_regime_results.json` |
| exp18/18b | Full 63-month Freddie SMM benchmark; fair ablation + ARIMA/ETS | 🔨 calibrated-TSFM real-data benchmark | `exp18_smm_full_results.json`, `exp18b_fair_smm_results.json` |
| exp19 | News-regime-augmented SMM forecasting | ❌ frozen — news doesn't move SMM | `exp19_news_smm_results.json` |
| exp20 | Event-augmented simulator validation | 🔨 build complete — 84% news-leading events (20 seeds) | [build summary](exp20_BUILD_SUMMARY.md) |
| exp21 | LLM feature discovery on event DGP | ❌ −21.6%: discovered features lag the news | [report](exp21_report.md) |
| exp22 | M2 v2 (decision policy) vs M0 | ⚖️ +6.6% when active, high variance | [report](exp22_report.md) |
| exp23 | Rule-based model selection | ❌ = random (p=0.80) | [report](exp23_report.md) |
| exp24 | Regime-aware selection (news-signal regimes) | ❌ honest wash | [report](exp24_report.md) |
| exp25 | h=3 horizon: does news help further out? | ❌ news worse by +25% | [report](exp25_report.md) |
| exp40 | Calibrated rerun of the event track | ⚖️ calibration cuts false discoveries; transfer problem remains | [report](exp40_report.md) |
| exp41 | Meta-learning (FFORMA-style) for model selection | ⚖️ learned beats hand-written; both lose to ARIMA | [report](exp41_report.md) |
| exp42 | FFORMA meta-learner vs equal weights | ❌ honest negative — doesn't beat equal weights | [report](exp42_report.md) |
| exp43 | Constrained regime router (3 regimes + fixed experts) | ✅ first to beat equal weights: M4 CRPS 0.1282 vs 0.1295, p=0.018 | [report](exp43_report.md) |
| exp44 | News as model selector (synthetic → real) | ⚖️ mechanism proven on synthetic (p=0.0068); real blocked by news sparsity | [report](exp44_report.md) |
| exp45 | Learned regimes (flexible) | ❌ honest negative — flexibility loses to the constrained prior | [report](exp45_report.md) |
| exp46 | Dense-news regime/selection | ❌ definitive negative — dense news doesn't help | [report](exp46_report.md) |
| exp47 | Proposed method (exp43 + exp44 combined) | ✅ beats equal weights (0.1451 vs 0.1507, p=6.5e-09), best single, and exp43 alone | [report](exp47_report.md) |
| exp48 | Reasoning-effort sweep (low/med/high × 3 tasks × 3 seeds) | ❌ "high" never helps: 3.5–6× slower, sometimes hurts (1.000→0.333 on BoC discovery). Default to "low" for diagnostics | [report](exp48_report.md) |

## Supporting reports

| Report | What it is |
|---|---|
| [multiseed_report.md](multiseed_report.md) | Multi-seed aggregation (macro track closure: 5-seed, no evidence macro helps) |
| [r2_comparison.md](r2_comparison.md) | R2 PIT-safe rerun comparison |
| [scale_up_report.md](scale_up_report.md) | Scale-up plumbing (10k×66, mock) |
| [scale_up_report_real-chronos2.md](scale_up_report_real-chronos2.md) / [-timesfm25](scale_up_report_real-timesfm25.md) | Scale-up with real TSFM weights |
| [scale_up_report_seed1…4-*.md](scale_up_report_seed1-chronos2.md) | Per-seed scale-up reports: [seed1](scale_up_report_seed1-chronos2.md) [seed2](scale_up_report_seed2-chronos2.md) [seed3](scale_up_report_seed3-chronos2.md) [seed4](scale_up_report_seed4-chronos2.md) (chronos2) · [seed1](scale_up_report_seed1-timesfm25.md) [seed2](scale_up_report_seed2-timesfm25.md) [seed3](scale_up_report_seed3-timesfm25.md) [seed4](scale_up_report_seed4-timesfm25.md) (timesfm25) · [seed1](scale_up_report_seed1-m0w3.md) [seed2](scale_up_report_seed2-m0w3.md) [seed3](scale_up_report_seed3-m0w3.md) [seed4](scale_up_report_seed4-m0w3.md) (m0w3) · [seed2](scale_up_report_seed2-m0w6.md) [seed3](scale_up_report_seed3-m0w6.md) [seed4](scale_up_report_seed4-m0w6.md) (m0w6) · [verify](scale_up_report_verify-m0w3.md) |
| [PHASE3_SUMMARY.md](PHASE3_SUMMARY.md) | Phase-3 adjustments from exp21–23 learnings (news-leading DGP, decision policy) |
| [exp21_BUILD_SUMMARY.md](exp21_BUILD_SUMMARY.md) | exp21 infrastructure build notes |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Environment, weights, and commands to reproduce any experiment |

Ablations (`abl_m0_window.py`, `abl_m3_bound.py`) and `collect_multiseed.py` are
utilities; their outputs feed the reports above. `evidence_corpus/` holds the
labeled evidence used by the news-track experiments — see its
[README](evidence_corpus/README.md).

## The honest-negatives ledger

Killed tracks stay killed only with evidence. The closed ones: R2 macro
(5-seed, no signal), M2/M3/M5 output-wrapper agents, FFORMA (exp42), flexible
regimes (exp45), dense news (exp46), news-for-SMM (exp19), h=3 news (exp25).
See [REDESIGN.md](../docs/REDESIGN.md) for what each kill taught us and
[POSITIVE_LIFT_AUDIT.md](../docs/POSITIVE_LIFT_AUDIT.md) for what survived
into the proposed method.
