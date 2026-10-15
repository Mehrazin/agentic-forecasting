TD-One Agentic Mortgage Renewal Attrition Forecasting System sets the technical specification, repository blueprint, data synthesis contracts, experimental matrix, and delivery execution plan for the Vector Bootcamp.
1. Problem Formulation & Literature Grounding
1.1 Mathematical Target & Point-in-Time Information Set
For a customer segment s and future renewal month m evaluated at run date t, let:
 * N_{s,m} \in \mathbb{Z}_{>0} be the known count of eligible mortgages scheduled for renewal in month m (observed at t via maturity_date in the mortgage history table).
 * A_{s,m} \in \{0, 1, \dots, N_{s,m}\} be the unknown future count of switches to another lender at renewal.
 * R_{s,m} = \frac{A_{s,m}}{N_{s,m}} \in [0, 1] be the segment switching rate.
The system forecasts the discrete predictive distribution of switching counts conditioned on the point-in-time information set \mathcal{I}_t available on or before run date t:
where:
 * A_{s,\text{past}}, N_{s,\text{past}} are historical segment-level switch and renewal counts whose resolution timestamps satisfy outcome_observed_at \le t.
 * X_{\le t} represents structured customer and mortgage monthly snapshots with snapshot_date \le t (plus optional renewal offers with offer_available_at \le t and propensity scores with score_available_at \le t).
 * H_{\le t} represents external macroeconomic time series and unstructured policy/market documents published on or before t.
To guarantee strict mathematical coherence between rates and counts across all models, if a backend predicts a continuous switching rate distribution \hat{R}_{s,m} \sim \text{Beta}(\alpha_{s,m}, \beta_{s,m}) or quantile trajectories, the count distribution is mapped via a Beta-Binomial readout A_{s,m} \sim \text{BetaBinomial}(N_{s,m}, \alpha_{s,m}, \beta_{s,m}), ensuring \hat{A}_{s,m} \in [0, N_{s,m}] and \mathbb{E}[A_{s,m}] = N_{s,m}\mathbb{E}[R_{s,m}].
1.2 Alignment with Survey Taxonomy (arXiv:2608.23058) & Bootcamp Literature
Following the unified formalism of Xu et al. (2026), LLM-based Agents for Forecasting and Prediction (arXiv:2608.23058), the TD-One system is defined as the five-tuple (\pi_\theta, \mathcal{A}, \mathcal{T}, \mathcal{M}, g):
 * Policy (\pi_\theta): The LLM controller orchestrating multi-step reasoning over segment profiles, numerical forecasts, and external macro/policy shocks.
 * Action Space (\mathcal{A}): \{\texttt{query\_segment}, \texttt{propose\_segment}, \texttt{retrieve\_evidence}, \texttt{run\_tsfm}, \texttt{adjust\_scenario}, \texttt{emit\_report}\}.
 * Toolkit (\mathcal{T}): Data & Segment Tool, Evidence Retrieval Tool, and Numerical Forecasting Tool.
 * Memory (\mathcal{M}): Point-in-time episodic trace cache and structured chronological context timeline \mathcal{H}_{1:t}.
 * Predictive Readout (g): A hybrid combiner mapping the numerical foundation model's baseline distribution and the agent's structured contextual adjustment into calibrated count quantiles (q_{10}, q_{50}, q_{90}), expected switches \mathbb{E}[A_{s,m}], expected rate \mathbb{E}[R_{s,m}], and cited manager recommendations.
| Reference Paper | Core Methodological Principle | Concrete Implementation in TD-One Repo | Target Phase |
|---|---|---|---|
| Xu et al. (2608.23058) LLM-based Agents for Forecasting | Graded agency continuum (K=1 prompt-only vs. K>1 closed-loop hybrid TSFM+LLM), strict decision-time t leakage barrier, and cost-accuracy Pareto reporting. | Implements the exact ablation ladder (Baseline \to TSFM \to Hybrid Agent), TemporalGuard middleware blocking post-t data, and token/latency telemetry per forecast run. | MVP1 Core |
| Karger et al. (2409.19839) ForecastBench | Dynamic, contamination-resistant evaluation using proper scoring rules (Brier, CRPS) and rolling evaluation cutoffs. | Rolling origin evaluation across 6 held-out months (t \in \{T-5, \dots, T\}) with strict record_asof snapshots and Brier/CRPS scoring. | MVP1 Core |
| Williams et al. (2410.18959) Context is Key (CiK) | Pairing numerical time series with essential textual context (regime shifts, policy constraints) where pure numerical models fail. | Plants explicit Canadian macro/policy shocks (BoC rate cuts, OSFI uninsured straight-switch MQR exemption, CCB thresholds) into synthetic DGP and evaluates context-gated uplift. | MVP1 Core |
| Schoenegger et al. (2402.19379) Wisdom of the Silicon Crowd | Ensembling diverse LLM personas/prompts rivals crowd calibration and reduces variance. | Multi-persona analyst ensemble (Macro Economist, Retail Risk Analyst, Competitor Pricing Analyst) aggregated via median/trimmed-mean log-odds adjustment in readout.py. | MVP2 Improvement |
| Requeima et al. (2405.12856) LLM Processes (LLMP) | Conditioning numerical predictive distributions and uncertainty intervals directly on natural language descriptions. | Natural-language scenario conditioning interface that shifts/scales the TSFM predictive Beta-Binomial parameters (\alpha, \beta) based on retrieved evidence severity. | MVP2 Improvement |
| Hu et al. (2408.08435) ADAS | Automated Design of Agentic Systems via meta-agent search over modular code/prompt workflows. | Meta-search script (experiments/exp5_meta_search.py) optimizing segment-discovery predicates and context-weighting prompts on validation months m \in [49, 54]. | Stretch Goal |
| Xiong et al. (2602.07755) ALMA | Meta-learning agentic memory designs so the agent continually learns which historical context patterns matter across episodes. | Episodic residual memory bank storing past forecast errors (A_{s,m} - \hat{A}_{s,m}) linked to macro regimes so the agent calibrates future adjustments across rolling months. | Stretch Goal |
| Zhang et al. (2505.22954 / 2603.19461) DGM & HyperAgents | Separate task-agent from meta-agent to recursively refine agent policies and segment discovery rules. | Outer-loop rule evolver that inspects trace failures on missed high-attrition cohorts and mutates candidate segment filter trees. | Stretch Goal |
2. Repository Architecture & Module Specifications
The repository (td-one-mortgage-agent) will be released under an MIT License with zero proprietary TD data. It adapts Vector's oil-forecasting reference implementation (for ReAct tool orchestration and evidence retrieval) and food-price forecasting reference implementation (for batched multi-segment time-series pipelines).
td-one-mortgage-agent/
├── LICENSE                             # MIT License
├── pyproject.toml                      # Dependencies: pandas, duckdb, pydantic, chronos-forecasting, timesfm, google-genai, httpx
├── README.md                           # Quickstart, architecture diagram, and reproducibility instructions
├── configs/
│   ├── synthetic_dgp.yaml              # Parameters for 10k customers, 66 months (60 train + 6 OOT), planted shocks
│   ├── segments_predefined.yaml        # Baseline segment definitions (e.g., Families, Low-rate borrowers)
│   ├── models.yaml                     # Chronos-2, TimesFM-2.5, and LLM controller configurations
│   └── experiments.yaml                # Cutoff dates, horizons (H=6), and evaluation metrics
├── data/
│   ├── raw_external/                   # Cached StatCan (Rows 2, 3), FRED (Rows 11, 12), OSFI (Row 59) & dated web corpus
│   ├── synthetic/                      # Generated Parquet tables matching Mortgage_Base_Data_and_Resources.xlsx
│   └── schemas/                        # Pydantic / Pandera strict schema validators
├── src/
│   ├── data_gen/
│   │   ├── mortgage_simulator.py       # Canadian semi-annual compounding + demographic/deposit generator
│   │   ├── shock_injector.py           # Injects BoC, CPI, OSFI, CCB, and competitor rate shocks into switching logits
│   │   └── export_parquet.py           # Outputs mortgage_history, renewal_outcomes, renewal_offers, propensity_scores
│   ├── data_service/
│   │   ├── pit_store.py                # Point-in-Time DuckDB query engine enforcing record_asof <= run_date
│   │   └── external_loader.py          # Loads StatCan, FRED, OSFI series aligned to publication release dates
│   ├── tools/
│   │   ├── data_segment_tool.py        # Tool 1: Filter validation, N/A/R aggregation, cohort profiling
│   │   ├── evidence_tool.py            # Tool 2: Google ADK / Perplexity / Local Frozen Corpus retrieval with date guards
│   │   └── forecasting_tool.py         # Tool 3: Calls numerical backend, applies Beta-Binomial bounds & contextual shifts
│   ├── models/
│   │   ├── baseline_historical.py      # Rolling historical segment switching rate + Jeffreys Beta-Binomial interval
│   │   ├── tsfm_chronos.py             # HuggingFace amazon/chronos-2 wrapper (Row 65)
│   │   ├── tsfm_timesfm.py             # HuggingFace google/timesfm-2.5-200m-transformers wrapper (Row 66)
│   │   └── hybrid_readout.py           # Combines TSFM trajectories with LLM log-odds / LLMP scenario adjustments
│   ├── agent/
│   │   ├── controller.py               # ReAct loop adapted from Vector oil-forecasting reference
│   │   ├── segment_discovery.py        # Proposes new high-risk subgroups from residual & evidence analysis
│   │   ├── memory_alma.py              # Optional ALMA episodic error memory across rolling forecast origins
│   │   └── report_compiler.py          # Compiles structured JSON + Markdown/HTML Manager Report
│   └── evaluation/
│       ├── metrics.py                  # Count MAE/RMSE/CRPS, Rate Brier/MAE, 80% PI Coverage, Top-K NDCG/Recall
│       ├── leakage_auditor.py          # Automated check verifying zero post-run_date timestamps in agent traces
│       └── backtest_runner.py          # Orchestrates rolling origin evaluation over the 6 OOT months
├── experiments/
│   ├── exp1_mvp1_predefined.py         # Historical vs. Chronos-2/TimesFM-2.5 vs. Hybrid Agent on predefined segments
│   ├── exp2_segment_discovery.py       # Predefined segments vs. Agent-discovered micro-segments
│   ├── exp3_context_ablation.py        # CiK-style ablation: No Context vs. Macro Only vs. Full Policy/Competitor Web
│   └── exp4_silicon_crowd_meta.py      # Stretch: Multi-agent ensemble + ADAS/ALMA memory refinement
└── tests/
    ├── test_schemas.py                 # Validates exact match against all 12 sheets of Mortgage_Base_Data_and_Resources.xlsx
    ├── test_canadian_amortization.py   # Verifies C001, C002, C003 exact payment & balance transitions
    └── test_pit_no_leakage.py          # Tests that pending/unobserved outcomes at run_date are masked

3. Data Engineering & Synthetic Generator Specification
3.1 Exact Schema Contracts (Mortgage_Base_Data_and_Resources.xlsx)
The synthetic generator (src/data_gen/mortgage_simulator.py) will simulate 10,000 active mortgage customers across 66 monthly snapshots (60 historical training months + 6 out-of-time evaluation months, e.g., April 2021 to September 2026, with renewal outcomes tracked through March 2027). All generated tables must pass strict Pandera/Pydantic validation matching Mortgage_Base_Data_and_Resources.xlsx:
 * mortgage_history.parquet (Required — Key: mortgage_id + snapshot_date):
   * 37 Core Fields: customer_id, household_id, mortgage_id, renewal_episode_id, snapshot_date, province, primary_borrower_age (18–90), household_size (1–10), children_under6 (0–6), children_6to17 (0–6, combined children \le 6 and < household_size), household_income_cad (0–500k), borrower_income_cad (\le household_income_cad), income_reference_year (\le snapshot year), td_tenure_months (0–600), origination_date, term_start_date, maturity_date, term_months \in \{12,24,36,48,60,84,120\}, mortgage_balance_cad (>0), mortgage_rate_pct (0.5–12.0), rate_type (fixed/variable), payment_structure (fixed_payment/adjustable_payment), remaining_amortization_months (1–360), mortgage_payment_monthly_cad, td_liquid_deposits_cad (0–500k), td_deposits_3m_change_cad, has_chequing, has_savings, has_credit_card, has_investments, credit_score_band (below_600, 600_659, 660_719, 720_plus, unknown), credit_score_asof (\le snapshot_date), missed_payment_count_12m (0–12), max_days_past_due_12m (0–365), marketing_contacts_90d (0–30), campaign_member_90d (bool), marketing_channel_last.
   * 9 Optional Fields (Included for Scenario & Discovery Experiments): employment_status (employed/self_employed/retired/other/unknown), adjusted_family_net_income_cad, government_benefit_income_monthly_cad, property_value_cad, property_value_asof, insured_flag, occupancy_type, other_debt_payment_monthly_cad, credit_utilization_pct.
   * Canadian Amortization Invariant: As verified against rows C001, C002, and C003 in Mortgage History Examples, fixed-rate monthly payments and principal transitions must use Canadian semi-annual compounding:
     
     
     (Verification: For C001 with B_t = \$420,000, rate = 3.0\%, n = 264, this yields \text{PMT} = \$2,171.20 and B_{t+1} = \$418,872.30, matching the Excel specification to the exact cent.)
 * renewal_outcomes.parquet (Required — Key: mortgage_id + renewal_episode_id + record_asof):
   * 9 Fields: customer_id, mortgage_id, renewal_episode_id, scheduled_maturity_date, outcome (renewed_td, switched_lender, paid_off, property_sale, default_exit, lender_declined, other_exit, pending, unknown), switched_lender_flag (1, 0, or NaN for pending/unknown), event_date, outcome_observed_at, record_asof.
   * Observation Delay Rule: Even after event_date passes, outcome_observed_at is simulated with a realistic settlement reporting lag of 1\text{–}7 days (e.g., C001 matures 2026-10-31, observed 2026-11-03). Any query at run date t < \text{outcome\_observed\_at} must see outcome = 'pending' and switched_lender_flag = NaN.
 * renewal_offers.parquet & propensity_scores.parquet (Optional for MVP1, Enabled for MVP2):
   * Implements the 15-field Renewal Offers Schema (tracking initial vs. superseded/revised rate offers and status_observed_at \le record_asof) and 7-field Propensity Scores Schema (open_td_investment_account, open_td_savings_account, open_td_chequing_account with score_available_at \le t).
3.2 Structural Causal Data Generating Process (DGP) for Switching
To ensure the synthetic benchmark meaningfully tests whether external context retrieval and segment discovery improve forecasting (following Context is Key, 2410.18959), individual switching probability p_{i,m} = P(\text{switched\_lender\_flag}_{i,m} = 1) at renewal month m is generated via a structural logit model combining internal behavioral features with external macro/policy regimes:
We plant three realistic, verifiable external regime effects in \gamma_{\text{policy}}(X_i, H_m) linked to the approved external sources:
 * Rate-Shock & Competitor Spread Effect (Poster Example 2 — Low-Rate Borrowers): Borrowers with rate_type == 'fixed', mortgage_rate_pct <= 3.0, and td_liquid_deposits_cad < 5000 who originated 5-year fixed mortgages in 2021 face a sharp payment shock upon renewing in 2026 against current market rates, elevating their switching rate to ~25% (N=100, \mathbb{E}[A]=25, 80% PI: 20\text{–}31).
 * OSFI Uninsured Straight-Switch Exemption Shock: Following OSFI's policy announcement exempting uninsured mortgage straight switches from the Minimum Qualifying Rate (MQR) stress test, uninsured borrowers (insured_flag == False) with high loan-to-income ratios and high credit scores (credit_score_band == '720_plus') experience a structural jump in switching propensity when competitor promo rates (RBC/Scotiabank web sources) undercut base renewal offers by \ge 25\text{ bps}.
 * Household Living-Cost & Benefit-Policy Squeeze (Poster Example 1 — Families): Households with household_income_cad < 100000 and children_under6 + children_6to17 >= 3 exposed to elevated CPI food/shelter inflation (StatCan Row 2) and benefit-policy adjustments (CRA CCB / Finance Canada announcements) exhibit a ~15% switching rate (N=200, \mathbb{E}[A]=30, 80% PI: 24\text{–}37) as they actively shop for lower monthly payments or cash-back switch incentives.
3.3 Approved External Data & Retrieval Sources Integration
All external inputs are strictly governed by the approved entries in Mortgage_Base_Data_and_Resources.xlsx:
| Resource Name | Workbook Row | Category | Integration Mode in Repo |
|---|---|---|---|
| StatCan Consumer Price Index (18-10-0004-11) | Row 2 | Core Structured | Ingested via external_loader.py with official monthly release timestamps; used as covariate & agent macro indicator. |
| Bank of Canada Interest Rates (10-10-0139-01) | Row 3 | Core Structured | Ingested via external_loader.py; provides policy rate & 5-year benchmark rate trajectories up to run date t. |
| FRED 10-Year Canada Govt Bond Yield (IRLTLT01CAM156N) | Row 11 (API Row 5) | Optional Structured | Fetched via FRED API (FRED_API_KEY) to model fixed-rate mortgage funding spreads. |
| FRED Real Residential Property Prices (QCAR368BIS) | Row 12 (API Row 5) | Optional Structured | Quarterly YoY house price change used for home-equity/LTV context. |
| OSFI M4 Consolidated Balance Sheet | Row 59 | Optional Reference | Macro banking sector mortgage volume benchmark. |
| Google ADK Search / URL Context & Perplexity API | API Rows 16 & 58 | Retrieval Tools | Queries whitelisted domains (bankofcanada.ca, canada.ca, osfi-bsif.gc.ca, rbcroyalbank.com, scotiabank.com) with publication date filters before:YYYY-MM-DD. |
> Reproducibility & Anti-Leakage Architecture (RetroSearch Pattern): Because live web pages for RBC/Scotiabank mortgage rates overwrite historical rates and LLMs have pretraining cutoffs, evidence_tool.py supports two modes:
>  * live_web mode: Calls Google ADK (Row 16) or Perplexity (Row 58) with strict date constraints for live demos.
>  * frozen_corpus backtest mode: Queries a local, version-controlled snapshot of dated BoC Monetary Policy Reports, OSFI advisories, Finance Canada releases, and competitor rate sheets indexed by published_at. The TemporalGuard middleware raises a hard exception if any retrieved document has published_at > run_date.
> 
4. Agent & Tooling Technical Specification
The system follows the controller-and-three-tools design from the system sketch poster:
4.1 Tool 1: Data & Segment Tool (src/tools/data_segment_tool.py)
 * Input Schema:
   class SegmentQuery(BaseModel):
    run_date: date
    horizon_months: int = Field(default=6, ge=1, le=12)
    segment_id: str
    filter_predicates: list[str]  # Validated SQL/DuckDB AST predicates on mortgage_history

 * Execution Logic:
   * Validates filter_predicates against allowed column names in Mortgage History Schema to prevent SQL injection or invalid column references.
   * Queries mortgage_history at snapshot_date <= run_date and joins renewal_outcomes where outcome_observed_at <= run_date.
   * Computes historical monthly series (N_{s,\tau}, A_{s,\tau}, R_{s,\tau}) for \tau \le t and known upcoming renewal counts N_{s,m} for future months m \in \{t+1, \dots, t+H\} based on maturity_date.
   * Enforces minimum sample size guardrails (flags warning if upcoming N_{s,m} < 20).
 * Output Schema: Historical (N, A, R) array, future N_{s,m} array, and a segment profile summary (mean contract rate, mean liquid deposits, % insured, mean household income, mean dependent children).
4.2 Tool 2: Evidence Retrieval Tool (src/tools/evidence_tool.py)
 * Input Schema: run_date: date, query: str, target_domains: list[str], segment_context: Optional[str].
 * Execution Logic:
   * Searches approved economic/policy sources (bankofcanada.ca, canada.ca, osfi-bsif.gc.ca, rbcroyalbank.com, scotiabank.com) and structured macro summaries (StatCan CPI/BoC tables) available as of run_date.
   * Following the Nexus (arXiv:2605.14389) and Survey (arXiv:2608.23058) architectures, extracts structured evidence cards containing: source_url, published_date, effective_date, policy_or_macro_fact, affected_borrower_predicates, and directional_impact (increase_switching / decrease_switching / neutral).
4.3 Tool 3: Forecasting Tool (src/tools/forecasting_tool.py)
 * Input Schema: segment_id: str, run_date: date, horizon_months: int, backend: Literal["historical", "chronos2", "timesfm2_5"], context_adjustment_logit: float = 0.0, uncertainty_multiplier: float = 1.0.
 * Execution Logic:
   * Historical Baseline (baseline_historical.py): Computes exponentially weighted or trailing 6-month historical switching rate \hat{R}_{s} = \frac{\sum_{\tau=t-5}^t A_{s,\tau}}{\sum_{\tau=t-5}^t N_{s,\tau}} with a Jeffreys prior \text{Beta}(0.5 + \sum A, 0.5 + \sum (N - A)).
   * Pretrained TSFM Backend (tsfm_chronos.py / tsfm_timesfm.py): Feeds the historical univariate/multivariate logit-transformed switching rate series \tilde{y}_{s,1:t} = \text{logit}\left(\frac{A_{s,\tau} + 0.5}{N_{s,\tau} + 1}\right) (and macro covariates when supported) into Chronos 2 (amazon/chronos-2, Row 65) or TimesFM 2.5 (google/timesfm-2.5-200m-transformers, Row 66) to sample S = 1,000 zero-shot future rate trajectories \hat{R}^{(k)}_{s,m} over m \in \{t+1, \dots, t+6\}.
   * Hybrid Contextual Readout (hybrid_readout.py): Applies the agent's evidence-grounded log-odds shift \Delta_{s,m} (bounded within [ -0.75, +0.75 ] to prevent runaway hallucinated adjustments) to each sampled trajectory:
     
     
     and draws count realizations A^{(k)}_{s,m} \sim \text{Binomial}\left(N_{s,m}, \hat{R}^{\text{adj},(k)}_{s,m}\right).
   * Bounds & Coherence Check: Verifies 0 \le \hat{A}_{s,m} \le N_{s,m}, computes expected switches \mathbb{E}[A_{s,m}], expected rate \mathbb{E}[R_{s,m}] = \mathbb{E}[A_{s,m}] / N_{s,m}, and the 80% prediction interval [q_{10}(A_{s,m}), q_{90}(A_{s,m})].
4.4 Agent Controller & Optional Segment Discovery (src/agent/controller.py)
Per run date t, the controller executes a structured workflow:
 * Step 1 (Predefined Segment Profiling): Calls data_segment_tool on the predefined baseline cohorts (e.g., All Renewals, Province cohorts, Fixed vs. Variable, Example 1: Families, Example 2: Low-rate borrowers).
 * Step 2 (Targeted Evidence Retrieval): Calls evidence_tool to retrieve macro trends (StatCan CPI, BoC rate decisions), regulatory shifts (OSFI straight-switch rules, Finance Canada / CCB updates), and competitor switch offers active prior to t.
 * Step 3 (Optional Segment Discovery — Experiment 2): Inspects recent residual spikes (A_{s,\tau} - \hat{A}_{s,\tau}) and newly retrieved policy rules (e.g., OSFI uninsured straight-switch exemption or low-deposit rate shock) to propose 1–3 candidate micro-segments via propose_segment (e.g., insured_flag == False AND mortgage_rate_pct <= 3.25 AND td_liquid_deposits_cad < 10000).
 * Step 4 (Forecasting & Scenario Synthesis): Calls forecasting_tool with the TSFM backend both with (\Delta_{s,m} \neq 0) and without (\Delta_{s,m} = 0) contextual adjustment, comparing against the historical-rate baseline.
 * Step 5 (Manager Report Compilation): Outputs a structured report ranking segments by expected switch volume \mathbb{E}[A_{s,m}] and excess switching rate over baseline, complete with 80% prediction intervals, cited external evidence URLs/dates, and concrete manager investigation actions.
5. Experimental Design & Evaluation Protocol
5.1 Evaluation Split & Forecasting Protocol
 * Total Horizon: 66 simulated months of customer history (N = 10,000 active mortgages).
 * Training / In-Context History: Months 1 \dots 60 (5 years of historical monthly snapshots and resolved renewals).
 * Out-of-Time (OOT) Test Window: Months 61 \dots 66 (final 6 simulated months reserved strictly for OOT testing).
 * Forecast Horizon: H = 6 months ahead at each monthly run date, plus rolling 1-to-6-month origin evaluation across the held-out window (using trailing 6 months of prior segment history for local rate calibration and up to 60 months for TSFM context).
5.2 Model Comparison Matrix (Ablation Ladder)
Following arXiv:2608.23058 and the Team Progress Tracker, every experiment compares five explicitly scoped configurations:
| Model ID | System Configuration | Agency Level (K) | Numerical Backend | Context / Retrieval | Segment Set |
|---|---|---|---|---|---|
| M0 | Historical-Rate Baseline | K=0 (Non-agentic) | Trailing 6m empirical rate + Jeffreys Beta-Binomial | None | Predefined only |
| M1a / M1b | Pure Numerical TSFM | K=0 (Non-agentic) | Chronos 2 (M1a) / TimesFM 2.5 (M1b) | Structured macro covariates only (CPI, BoC, FRED) | Predefined only |
| M2 | Prompt-Only LLM Forecaster | K=1 (Zero-shot prompt) | None (Direct LLM numerical/quantile readout) | Pre-fetched static macro summary | Predefined only |
| M3 (MVP1 Target) | Hybrid Agent + TSFM (Predefined) | K \in [3, 6] (ReAct Tool Agent) | Best of Chronos 2 / TimesFM 2.5 + Beta-Binomial readout | Active retrieval via evidence_tool (Macro + Web/Policy) | Predefined only |
| M4 (MVP2 / Stretch) | Hybrid Agent + Discovery + Silicon Crowd | K \in [5, 10] (Multi-step / Meta) | TSFM + 3-Persona Silicon Crowd Ensemble + ALMA Memory | Active retrieval + Offer/Score tables | Predefined + Agent-Proposed Groups |
5.3 Evaluation Metrics
To align with both statistical forecasting standards (ForecastBench, 2409.19839; Survey, 2608.23058) and TD business utility, every run logs five metric families:
 * Count Accuracy Metrics:
   
   
   plus Discrete Ranked Probability Score (CRPS) over the predictive count distribution P(\hat{A}_{s,m} = a).
 * Rate Accuracy Metrics:
   
   
   and Episode-level Brier Score.
 * Uncertainty & Interval Calibration:
   * 80% Prediction Interval Empirical Coverage: Fraction of actual switch counts A_{s,m} falling inside [q_{10}(\hat{A}_{s,m}), q_{90}(\hat{A}_{s,m})] (target: 0.80 \pm 0.05).
   * Winkler Interval Score (\alpha = 0.20): Penalizes both over-wide intervals and missed bounds:
     
 * Group Prioritization & Discovery Utility:
   * Top-K High-Attrition Recall@3 & NDCG@5: Agreement between the top segments ranked by predicted switching rate/count and actual highest-switching cohorts.
   * Excess Switches Identified (\Delta A_{\text{captured}}): Additional true switchers concentrated in agent-proposed segments relative to predefined coarse segments for the same outreach capacity budget (e.g., top 1,000 renewals flagged for manager review).
 * Agentic Quality, Safety & Cost Metrics (arXiv:2608.23058):
   * Temporal Leakage Rate: Must equal 0.0% (verified automatically by leakage_auditor.py checking every retrieved URL/document timestamp against run_date).
   * Citation Faithfulness Rate: % of claims in the Manager Report backed by a retrieved evidence_tool card.
   * Inference Cost & Latency: Mean tokens, API calls, and wall-clock seconds per segment forecast.
6. Execution Plan, Team Workstreams & Delivery Schedule
6.1 Workstream Allocation Across TD-One Team
To maximize parallel velocity across the 5-person team:
| Team Member | Role Title | Primary Technical Workstream & Deliverables |
|---|---|---|
| Mehrazin Alizadeh | Applied ML Scientist II (Team Lead) | System Architecture, Agent Controller & Report Compiler: Leads src/agent/controller.py, report_compiler.py, ReAct prompt engineering, and end-to-end integration across tools. |
| Vandad Davoodnia | Applied ML Scientist II | Numerical Foundation Models & Hybrid Readout: Implements tsfm_chronos.py (Row 65), tsfm_timesfm.py (Row 66), baseline_historical.py, and Beta-Binomial hybrid_readout.py. |
| Fateme Ghadami | Data Scientist II | Synthetic Data Engine & Point-in-Time Data Service: Builds mortgage_simulator.py, shock_injector.py, DuckDB pit_store.py, and data_segment_tool.py matching all 12 Excel sheets. |
| Mohammad Hassan Khatami | Applied ML Scientist I | External Data & Evidence Retrieval Tool: Builds external_loader.py (StatCan Rows 2–3, FRED Rows 11–12, OSFI Row 59) and evidence_tool.py (Google ADK Row 16, Perplexity Row 58, frozen backtest corpus). |
| Nazanin Mehregan | Applied ML Scientist I | Evaluation Harness, Leakage Auditor & Segment Discovery: Builds metrics.py, leakage_auditor.py, backtest_runner.py, and segment_discovery.py for Experiments 1–4. |
6.2 Step-by-Step Build & Delivery Roadmap
Phase 1: Data & Baselines ──► Phase 2: MVP1 Core Agent ──► Phase 3: Evaluate & Trace ──► Phase 4: Improve (2-3x) ──► Phase 5: Final Package
 (Schemas, DGP, PIT Store,     (3 Tools, ReAct Loop,        (6m OOT Backtest, Trace       (Segment Discovery,         (Reproducible Report,
  M0 Historical, M1 TSFMs)      Predefined Segments)         Inspection, Leakage Audit)    Silicon Crowd, LLMP)        Poster/Deck, MIT Repo)

 * Phase 1 — Data Service & Non-Agentic Baselines (Pre-Build / Build Day 1):
   * Complete src/data_gen/ to generate the 10k-customer, 66-month synthetic dataset (mortgage_history.parquet and renewal_outcomes.parquet) with planted CPI, BoC, OSFI, and CCB shocks.
   * Run pytest tests/ to verify 100% schema compliance against Mortgage_Base_Data_and_Resources.xlsx and zero-cent deviation on the Canadian semi-annual mortgage payment formula.
   * Implement pit_store.py and run non-agentic baselines M0 (Historical Rate) and M1a/M1b (Chronos 2 & TimesFM 2.5) on the predefined segments across the 6 OOT months.
 * Phase 2 — MVP1 End-to-End Agentic Workflow (Build Day 2):
   * Wire data_segment_tool, evidence_tool, and forecasting_tool into controller.py (adapting Vector's oil-forecasting agent loop).
   * Generate the first reproducible MVP1 Manager Report (M3) replicating Poster Example 1 (Families: Income < $100k, \ge 3 children) and Poster Example 2 (Low-rate borrowers: Fixed rate \le 3\%, Deposits < $5k) with 80% prediction intervals and cited evidence.
 * Phase 3 — Interactive Trace Analysis & Error Diagnosis (Build Day 3 — "Evaluate"):
   * Inspect full agent execution traces across the 6 held-out months.
   * Diagnose failure modes required by the Build Days rubric: (a) noisy web retrieval diluting macro signals, (b) over-adjustment of TSFM logits on small segments, and (c) missed attrition spikes hidden inside coarse predefined segments.
 * Phase 4 — Implementing 3 Measurable Improvements (Build Day 4 — "Improve"):
   * Improvement 1 (Structured Context Filtering / CiK): Add a chronological context distiller in evidence_tool.py that maps raw policy/rate text into structured impact cards before adjustment, reducing forecast variance.
   * Improvement 2 (Agent-Proposed Segment Discovery): Enable segment_discovery.py so the agent queries decision-tree residuals and retrieved policy criteria (e.g., uninsured straight-switch rules + renewal offer rate gaps from renewal_offers.parquet) to surface high-risk micro-segments.
   * Improvement 3 (Silicon Crowd Ensemble + ALMA Residual Memory): Aggregate 3 specialized LLM analyst personas (Wisdom of the Silicon Crowd, 2402.19379) with episodic error calibration across rolling months (ALMA, 2602.07755).
 * Phase 5 — Final Refinement, Documentation & Executive Presentation (Build Day 5 — "Refine & Document"):
   * Update the Team Progress Tracker daily with quantitative before/after metrics across M0–M4.
   * Freeze the public GitHub repository under the MIT license with a one-command reproduction script (make reproduce-all) that regenerates the synthetic data, executes the backtest, verifies 0% temporal leakage, and builds the final HTML/PDF Manager Report and presentation deck.
For a deeper visual walkthrough of how self-improving meta-agents and task-agents separate domain reasoning from recursive policy refinement in your stretch experiments, see Hyperagents Explained: Recursive Self Modification and Cross Domain Learning, which directly breaks down the arXiv:2603.19461 architecture from your reading list.
Would you like me to generate the starter Python code for the Canadian Synthetic Mortgage Simulator (mortgage_simulator.py) and Point-in-Time DuckDB Service (pit_store.py) first, or start with the ReAct Agent Controller and 3 Tool interfaces?

YouTube video views will be stored in your YouTube History, and your data will be stored and used by YouTube according to its Terms of Service
