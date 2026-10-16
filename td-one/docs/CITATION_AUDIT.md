# Citation Audit — LITERATURE_SURVEY.md

**Date:** 2026-10-16. **Auditor:** Muse (subagent, web verification).
**Scope:** the ~22 strongest empirical/technical claims in
`docs/LITERATURE_SURVEY.md`, checked against primary sources
(papers, book chapter, journal records). Three independent workers
each covered one section. This is an audit trail — the survey itself
was not rewritten.

**Verdict key:** CONFIRMED = claim matches the primary source.
PARTIALLY = core is right but something is off (noted).
UNVERIFIABLE = primary could not be accessed. WRONG = contradicted.

**Overall:** 18/22 CONFIRMED, 4/22 PARTIALLY, 0 UNVERIFIABLE,
0 WRONG. No claim was contradicted, but four need corrections and
there are nine citation-hygiene fixes (listed at the end).

---

## §1 — LLM-based time-series forecasting

| # | Claim | Cited source | Primary source checked | Verdict | Notes |
|---|---|---|---|---|---|
| 1.1 | LLMTime: GPT-3/LLaMA-2 70B zero-shot beats ARIMA, TCNs, N-HiTS on CRPS and MAE across Monash and Informer benchmarks | Gruver et al., NeurIPS 2023, arXiv 2310.07820 | https://ar5iv.org/html/2310.07820 (§4 Experiments) | **PARTIALLY** | The CRPS/NLL win over ARIMA/TCN/N-HiTS is on the **Darts** benchmark (8 univariate datasets), not Monash/Informer. Monash (19 datasets, GPT-3, MAE) and Informer (multivariate, LLaMA-2 70B only, MAE) report "best or second best" *aggregate* MAE against different baselines (e.g., Autoformer/FEDformer). "Beats" overstates the framing. |
| 1.2 | LLMTime: chat-tuned (instruction/RLHF) versions forecast markedly worse than base models | Same | ar5iv.org/html/2310.07820 (abstract, §2, Fig. 1 caption) | CONFIRMED | GPT-4 worse than GPT-3; paper attributes it partly to tokenization and partly to alignment ("likely the result of alignment interventions such as RLHF"). Caveat: "markedly" is the survey's gloss. |
| 1.3 | LLMTime: paper itself flags data-leakage risk from undisclosed training data | Same | ar5iv.org/html/2310.07820 (§4) | CONFIRMED | §4: "As we are using LLMs with undisclosed datasets, data leakage is an important concern that we address directly in Appendix B." |
| 1.4 | TIME-LLM: ~8% SMAPE improvement over GPT4TS; competitive with N-HiTS on M4 | Jin et al., ICLR 2024 | https://ar5iv.org/html/2310.01728 (§D.2) | CONFIRMED | Paper: "surpass GPT4TS by a large margin (e.g., **8.7% overall**, 13.4% on M4-Yearly, 21.5% on M4-Hourly/Daily/Weekly)"; "Compared to N-HiTS and PatchTST, Time-LLM exhibits comparable or superior performances." |
| 1.5 | TEMPO: cross-domain generalization; few-shot (≤10% data) is where the edge is largest | Cao et al., 2023, arXiv 2310.04948 | https://ar5iv.org/html/2310.04948 (abstract + full-text search) | **PARTIALLY** | Cross-domain generalization confirmed (zero-shot on unseen datasets). The few-shot claim is **unsupported**: "few-shot" occurs exactly once in the full paper as a generic statement, with no few-shot experiments. That result belongs to **TIME-LLM** (10%/5% few-shot MSE improvements) — the survey conflated the two papers. |
| 1.6 | TimeGPT: conformal intervals by design; independent evaluations show ~parity with ARIMA/Prophet, not dominant | Nixtla (no primary cited) | https://ar5iv.org/html/2310.03589 (TimeGPT-1 §5.3); https://www.mdpi.com/2813-0324/11/1/29 (MDPI 2025 independent eval) | CONFIRMED | TimeGPT-1 report: "We rely on **conformal predictions** based on historic errors to estimate prediction intervals." Independent MDPI 2025 study (TimeGPT zero-shot vs ARIMA vs Prophet): mixed results ≈ parity, never dominant. Caveats: that study uses synthetic datasets; Nixtla's own report excluded ARIMA/Prophet as benchmarks, so the parity framing rests on the independent study — accurately characterized. |
| 1.7 | Park et al. (2025): LLM zero-shot forecasters underperform even simple domain-specific models, are fragile to input noise; recommends fine-tuning | arXiv 2506.00457 | https://arxiv.org/abs/2506.00457 | CONFIRMED | Paper exists at that ID (Junwoo Park et al., submitted 31 May 2025). Abstract states all three: "underperforming even simple domain-specific models," "sensitivity to noise," fine-tuning "a more promising direction." Verified at abstract level only (no full-text read). |
| 1.8 | Zhou et al. (2025): under AR(p), linear self-attention cannot beat classical linear models in-context; CoT predictions collapse to the mean exponentially | arXiv 2510.09776 | https://arxiv.org/abs/2510.09776 | CONFIRMED | Paper exists at that ID (Yufa Zhou et al., submitted 10 Oct 2025). Abstract states all three results verbatim. Verified at abstract level only. |

---

## §2 — Agentic forecasting (+ ECB expert-combination from §3)

| # | Claim | Cited source | Primary source checked | Verdict | Notes |
|---|---|---|---|---|---|
| 2.1 | TSci: four agents (Curator/Planner/Forecaster/Reporter); 10.4% error reduction vs statistical baselines, 38.2% vs LLM baselines, on eight benchmarks | arXiv 2510.01538v2 | https://arxiv.org/abs/2510.01538v2 (v2, 6 Oct 2025) | CONFIRMED | Abstract verifies all four agents, "eight established benchmarks," and both percentages. Minor: body text says 38.3% vs abstract's 38.2% — the survey matches the abstract; harmless. |
| 2.2 | TSci: LLM is orchestration-only, never emits forecast numbers — selects preprocessing, models, ensemble weights; tools do the numerics | Same | arXiv PDF §3.4 (via indexed full text) | CONFIRMED | Forecaster "leverages an LLM-guided policy to select an ensemble strategy" (single-best, performance-aware averaging, robust aggregation); the ensemble model then produces the forecast. Nuance: the LLM selects the *strategy*; weights follow from it — it does not emit weights as raw numbers. Survey's phrasing is accurate in substance. |
| 2.3 | REATS (Zhang et al., Microsoft Research, 2026): LLM as ensemble router — CoT over textual+numerical features, sample-adaptive interpretable weights, beats fixed ensemble baselines | No arXiv ID in survey | https://arxiv.org/abs/2608.10149 (submitted 11 Aug 2026, Xu Zhang et al.); https://www.microsoft.com/en-us/research/publication/reats-llm-reasoning-based-ensemble-learning-for-adaptive-time-series-forecasting/ | CONFIRMED | Paper exists; MSR publication page confirms Microsoft Research. Abstract confirms router, joint textual temporal-pattern + numerical features, chain-of-thought, interpretable sample-adaptive weights, outperforming ensemble baselines on eight benchmarks with NL explanations. |
| 2.4 | DCATS (Yeh et al., 2025): LLM agent selects auxiliary series to enrich training data; 6% error reduction on traffic data | arXiv 2508.04231 | https://arxiv.org/abs/2508.04231v1 ; https://arxiv.org/pdf/2508.04231v2 | CONFIRMED | Authors: Chin-Chia Michael Yeh et al., **Visa Research**. 6% average error reduction across models/horizons on large-scale traffic data confirmed. Flag: the paper's true title is **"Empowering Time Series Forecasting with LLM-Agents"** — "Data-Centric Agent for Time Series" is the DCATS acronym expansion, not the title. |
| 2.5 | Survey quote: "verified reasoning traces can improve performance, whereas excessive reasoning tokens may reduce accuracy, extended deliberation may worsen calibration" | arXiv 2608.23058 | https://arxiv.org/abs/2608.23058 (24 Aug 2026, Xiaogang Xu et al., 15 authors) | CONFIRMED | Survey exists at that ID; the sentence appears **verbatim** in the paper's "Inference-Time Adaptation" section (confirmed via indexed full text; PDF fetch timed out). |
| 2.6 | Radhakrishnan et al. (2025): LLM *combining* expert forecasts (aggregator, not adjuster) reduced errors, esp. under expert disagreement/inattentiveness | arXiv 2506.23154 | https://arxiv.org/abs/2506.23154v1 (29 Jun 2025) | **PARTIALLY** | Substance CONFIRMED: zero-shot LLM ensemble combining ECB SPF forecasts; superior to simple averaging; "greater resilience to expert disagreement (H3)," "stable performance under forecaster inattentiveness (H4)." **Author attribution is wrong**: arXiv metadata lists **Yinuo Ren, Jue Wang** — not "Radhakrishnan et al." (the manuscript's running header reads "CV Radhakrishnan et al."). Correct to Ren & Wang or verify against the manuscript author list. Full-PDF read timed out; verdict rests on abs page + indexed body text — high confidence. |

---

## §3 + §4 — Judgmental adjustment and conformal prediction

| # | Claim | Cited source | Primary source checked | Verdict | Notes |
|---|---|---|---|---|---|
| 3.1 | FPP2: adjustments sparingly, only with "significant additional information"; don't adjust to fix missed systematic patterns; small optimistic adjustments hurt, large informed ones can help | Hyndman & Athanasopoulos, FPP 2nd ed., otexts chapter | https://otexts.com/fpp2/judgmental-adjustments.html | CONFIRMED | All three paraphrases map closely to the "Use adjustments sparingly" section. Book is CC-licensed and freely readable; cited URL is exact. |
| 3.2 | Armstrong: adjustments "often increase errors"; standing recommendation since 1954: don't subjectively adjust | Armstrong checklists; Meehl (1954); Grove et al. (2000) | Primary Armstrong text not accessible via browser; secondary deck confirms principle | **PARTIALLY** | The thrust — "Limit subjective adjustments of quantitative forecasts" is genuinely an Armstrong principle (confirmed in Fildes/Goodwin/Lawrence deck). But the exact phrase "often increase errors" was not verifiable in a primary Armstrong source, and attributing a "don't adjust" dictum "since 1954" to Meehl overstates specificity: Meehl's work was clinical-vs-statistical prediction broadly, not a rule about adjusting model forecasts. Soften to a paraphrase. |
| 3.3 | Meehl (1954), *Clinical versus Statistical Prediction* | U. Minnesota Press | Title/year/publisher cross-checked (bookseller records) | CONFIRMED | Title, year 1954, publisher (University of Minnesota Press) all verified. |
| 3.4 | Grove, Zald, Lebow, Snitz & Nelson (2000), "Clinical versus mechanical prediction" meta-analysis | Journal article | https://pubmed.ncbi.nlm.nih.gov/10752360/ | CONFIRMED | Authors, year, and finding (mechanical ≈10% more accurate) verified. **Venue correction:** *Psychological Assessment* 12(1), 19–30 — **not** Psychological Bulletin. |
| 3.5 | Franses & coauthors (JRFM 2020): more expert adjustment → lower accuracy, GDP/inflation/unemployment | J. Risk Financial Manag. 2020 | https://ideas.repec.org/a/gam/jjrfmx/v13y2020i3p44-d327516.html ; https://repub.eur.nl/pub/125158 | CONFIRMED | Exists: **Franses & Welz**, JRFM 2020, vol 13, article 44, doi:10.3390/jrfm13030044. Finding confirmed: "the more a forecaster deviates from a prediction from an econometric model, the less accurate are the forecasts." |
| 4.1 | CQR (Romano et al., NeurIPS 2019): score max(q_lo − y, y − q_hi); keeps heteroskedasticity; finite-sample coverage | Paper | https://papers.nips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf | CONFIRMED | Score Ei := max{q̂_lo(Xi) − Yi, Yi − q̂_hi(Xi)} is eq. (6); intervals [q̂_lo − Q̂, q̂_hi + Q̂] are adaptive in x (heteroskedastic) with finite-sample marginal coverage. |
| 4.2 | ACI (Gibbs & Candès, NeurIPS 2021): α_{t+1} = α_t + γ(α − err_t); long-run coverage under arbitrary distribution drift; one tuning parameter | Paper | https://arxiv.org/abs/2106.00170 | CONFIRMED | Abstract: coverage guarantees "irrespective of the true data generating process." Update rule and long-run bound verified. NeurIPS 2021 venue confirmed. |
| 4.3 | EnbPI (Xu & Xie, ICML 2021): bootstrap ensemble + OOB residuals; no train/calibration split; sliding residual window handles drift | arXiv 2010.09107 | https://arxiv.org/abs/2010.09107 | CONFIRMED | Abstract: "does not require data exchangeability… avoids data-splitting." Conference version (PMLR 139, ICML 2021) confirms no-split suitability for small samples and residual-window updates. |
| 4.4 | Conformal PID (Angelopoulos et al., NeurIPS 2023): sequential-dependence refinements, control-theoretic guarantees | Paper | https://arxiv.org/pdf/2307.16895 ; NeurIPS 2023 poster 69896 | CONFIRMED | Title, authors (Angelopoulos, Candès, Tibshirani), venue verified; PID feedback on coverage-error/quantile tracking confirmed. |

---

## Verification gaps (honest accounting)

- Claims 1.7, 1.8, 2.5, 2.6 were verified at **abstract + indexed-full-text level only**; full-PDF fetches timed out for the two August-2026 arXiv papers (2.5, 2.6). The quoted finding for 2.5 is verbatim per indexed text; 2.6's substance is high-confidence from the abstract plus indexed body text. A full-PDF recheck is cheap and worth doing before publication-grade use.
- Claim 3.2's primary Armstrong source was not accessible via browser — the verdict rests on a secondary deck reproducing Armstrong's principles.
- The repo's `docs/LITERATURE_SURVEY.md` References section was not re-read during this audit; the citation-hygiene fixes below are ground truth to compare against it.

---

## Citation-hygiene fixes for LITERATURE_SURVEY.md (not rewriting — parent's call)

1. **LLMTime (§1.1):** the CRPS win over ARIMA/TCN/N-HiTS was Darts-only; Monash/Informer were MAE "best or second best" aggregates vs different baselines. Rewrite the sentence to attribute each result to its benchmark.
2. **TEMPO (§1.1):** the few-shot claim belongs to TIME-LLM; remove or re-attribute.
3. **TIME-LLM References:** add missing arXiv ID **2310.01728**.
4. **REATS References:** add missing arXiv ID **2608.10149** (verified; Microsoft Research confirmed).
5. **ECB expert-combination (§3):** author attribution "Radhakrishnan et al." contradicts arXiv metadata (**Yinuo Ren, Jue Wang**, arXiv 2506.23154). Correct to Ren & Wang unless the manuscript author list says otherwise.
6. **Grove et al. 2000 References:** venue is *Psychological Assessment* 12(1), 19–30, not Psychological Bulletin.
7. **Franses References:** add co-author **Max Welz**; JRFM 2020, vol 13, article 44, doi:10.3390/jrfm13030044.
8. **§3 intro:** soften "often increase errors" (exact phrase not found in a primary Armstrong source) and the "recommendation since 1954" framing (overstates Meehl's specificity) to paraphrases.
9. **DCATS References:** true paper title is *Empowering Time Series Forecasting with LLM-Agents*; "Data-Centric Agent for Time Series" is the acronym expansion — fine as a descriptor, not as the title.

**Headline assessment:** the survey's substantive claims survived contact with the primaries. The four PARTIALLY verdicts are attribution/attribution-precision issues (wrong benchmark scope, conflated papers, wrong author list), not fabrications. Nothing was contradicted.
