# Real-Data Survey: Escaping the Synthetic Bubble

**Date:** 2026-10-16. **Status:** M4 done (exp14, `RESULTS.md` §2k);
Freddie Mac pending registration. Original survey text preserved below.
**Motivation:** all TD-One model comparisons so far run on synthetic DGPs we
designed ourselves. We need real outcomes to (a) validate that our rankings
transfer, and (b) give our methods a credible external baseline.

Our models forecast binary-outcome rates (P(event) per cohort, probabilistic,
scored with CRPS/Brier). Ideal dataset: real binary outcomes, monthly
frequency, 5+ years of history, 50+ backtest origins, finance/mortgage domain.

---

## 1. Freddie Mac Single-Family Loan-Level Dataset (SF LLD)

**What it is.** Loan-level records for US fixed-rate mortgages purchased or
guaranteed by Freddie Mac, originations from 1999 onward, with a monthly
performance record per loan from origination through termination. This is the
de facto standard public dataset for mortgage prepayment/default research —
the arXiv paper "A Bayesian approach to modeling mortgage default and
prepayment" (1706.07677) runs its competing-risks analysis on the 1999
vintage (672k mortgages), and multiple open-source projects build prepayment
pipelines on it.

**Access.** Free registration required (not paywalled, but gated).
Steps: register at <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>,
then download from the Clarity portal
(<https://claritydownload.fmapps.freddiemac.com/CRT/#/sflld>)
under "Standard Dataset Download by Year". Two tiers:
- **Sample files** (`sample_YYYY.zip`): a 50,000-loan random sample per
  origination vintage — a few hundred MB, the tractable starting point.
- **Full standard files**: ~50 GB uncompressed across vintages.

**Format.** Pipe-delimited (`|`) text, no header rows. Two files per vintage:
`sample_orig_YYYY.txt` (one row per loan: static attributes) and
`sample_svcg_YYYY.txt` (one row per loan-month: performance). A published
user guide gives exact column positions. Key columns:
- Origination: loan sequence number, origination quarter, Freddie Score
  (FICO), original LTV, DTI, note rate, original UPB, property state/MSA,
  property type, loan purpose, number of borrowers.
- Performance: loan sequence number, monthly reporting period, current UPB,
  current loan delinquency status (`00` = current … `99` = n payments past
  due, `RA` = REO acquisition), zero balance code, zero balance effective
  date, current interest rate.

**Deriving prepayment.** The zero balance code is set once per loan, at the
first terminating event. `01` = prepaid or matured (voluntary payoff — our
target). `02`/`03`/`09`/`15` = credit events (third-party sale, short
sale/charge-off, REO disposition, note sale) — competing risks, censor for
prepayment analysis. `06`/`16`/`96` = administrative removals (repurchase,
reperforming securitization, defect) — censor, not borrower behavior.
Known limitation (documented in the user guide and flagged by practitioners):
code `01` conflates refinance-driven payoff with sale-driven payoff, and
payoff with scheduled maturity. For attrition modeling this is acceptable —
both are attrition from the investor's view — but never claim to distinguish
"moves" from "refinances" on this data.

**Framing for our models.** Cohort by (origination year-quarter × FICO bucket
× note-rate bucket) → monthly panel of at-risk counts and voluntary
prepayment counts → monthly SMM, rolled to 3-month and 12-month CPR. Origins:
monthly from ~2021 onward on the 2019–2021 vintages (50+ origins; the COVID
refi boom gives real regime variation). Features: trailing cohort SMM,
borrower/loan attributes, FRED macro (MORTGAGE30US spread vs cohort WAC,
unemployment, HPI). Forecast: predictive distribution of next-12-month cohort
CPR. Score: CRPS/Brier vs realized. Models: M0 Beta-Binomial, macro_m0,
TSFMs, regime-gated agent — the full ladder, unchanged.

**License.** Free for research; the data use agreement prohibits
redistribution — raw files stay out of the repo (gitignored), derived
aggregates only.

**Tractability on 7 GB VM.** One 50k-loan sample ≈ 2M loan-month rows —
fits comfortably in DuckDB or chunked pandas. Full vintages need streaming
but are feasible.

**Fit score: 5/5.** Real outcomes, exact problem shape (binary → cohort
rates → probabilistic forecast → CRPS), 25 years of monthly history,
regime variation included.

---

## 2. M4 Competition — Monthly finance/macro subsets

**What it is.** The M4 competition's 100,000 real time series across 6
frequencies and 6 domains (micro, industry, macro, finance, demographic,
other). The monthly slice: 48,000 series (10,987 finance, 10,016 macro),
lengths 60–2812, fixed forecast horizon of 18. The most-cited forecasting
benchmark in the literature; 61 published methods to compare against.

**Access.** No registration. Direct raw CSVs from the competition repo:
- `https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv`
- `https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv`
- `https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/M4-info.csv`
  (metadata: category is the second column — filter to Finance/Macro)
- Total raw M4 ≈ 250 MB. A Kaggle mirror also exists
  (`kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset`).

**Format.** Row-per-series CSVs, NaN-padded; train/test split is fixed
(one origin per series, horizon 18).

**Framing for our models.** Mismatch: series are continuous levels, not
binary outcomes. Two adaptations: (a) forecast the predictive distribution
of the series over the 18-month horizon and score with CRPS against the
held-out test segment — our probabilistic machinery applies directly;
(b) derive binary events (e.g., P(series rises more than x% over horizon))
and score with Brier. Adaptation (a) is cleaner. Note: M4's published
leaderboard is point-forecast sMAPE/MASE, so we compare against our own
baselines on CRPS, not against the 61 methods head-to-head.

**License.** Competition data, freely mirrored for research; no formal
license file — de facto research use (standard practice in hundreds of
papers).

**Tractability.** Trivial — 250 MB total, and we'd subset to ~2k
finance/macro series anyway.

**Fit score: 2/5** for our problem shape (wrong outcome type), **5/5** as a
famous-benchmark credibility check. Best used as a secondary "does our
probabilistic machinery work on real series" test, not the primary
evaluation.

**Update 2026-10-16:** ran as exp14 (`RESULTS.md` §2k, `evaluation/real_benchmark.py`).
Both real TSFMs beat classical ~20% CRPS on 2,000 Finance+Macro series —
the synthetic "M0 beats everything" did not generalize.

---

## 3. FRED-MD (McCracken–Ng monthly macro database)

**What it is.** 134 monthly US macro indicators, 1959–present, updated
monthly by the St. Louis Fed — the standard "big macro data" panel
(output/income, labor, housing, consumption, money/credit, rates, prices).
The Monash archive ships a convenient 107-series × 728-month snapshot.

**Access.** Free, no registration:
<https://research.stlouisfed.org/econ/mccracken/fred-databases/> →
`current.csv` (a few MB). Monash snapshot via the archive (see §4).

**Framing for our models.** Same shape mismatch as M4 (continuous series).
Two roles: (a) as a *benchmark*, frame binary events — e.g., NBER recession
months, or P(series crosses a threshold) — and score Brier; (b) as the
*exogenous data store* for our mortgage work — this is the higher-value use.
Our macro_m0 and TSFM-macro models currently use 19 hand-picked FRED series
(+WTI crude and Canada household debt-to-GDP added 2026-10-16);
FRED-MD gives 134, including the housing block most relevant to prepayment
(housing starts, permits, house prices) that we haven't touched.

**Fit score: 3/5** as a benchmark (wrong shape, right domain), **5/5** as the
macro data store our agent layer should consume. The user's "hook them up
with data stores" direction points here.

---

## 4. Monash Time Series Forecasting Archive — finance-relevant slices

**What it is.** 30 curated datasets in a uniform format (Godahewa et al.
2021), downloadable from Zenodo; also loadable via the R `tsibbledata`
package. Finance/economics-relevant slices:
- **NN5** (banking): 111 daily ATM cash-withdrawal series, 791 days each.
- **CIF 2016** (banking): 72 monthly series, 34–120 months.
- **Bitcoin** (economic): 18 daily series.
- **FRED-MD** (economic): 107 monthly series × 728 months — the convenient
  single-file version of §3.

**Access.** Free, no registration. Example:
`https://zenodo.org/records/4656132` (traffic); dataset DOIs listed in the
archive paper. Format: `.tsf` (a simple tagged text format).

**Framing.** Same continuous-series adaptation as M4. NN5 is the most
interesting: daily banking data with real calendar effects; quantile
forecasting of withdrawal volumes is a legitimate probabilistic task.

**Fit score: 2/5.** Useful only as a tertiary check; M4 dominates it on
fame and FRED-MD dominates it on domain relevance.

---

## 5. Literature mortgage-prepayment benchmarks

There is **no canonical "M4 of mortgage prepayment"** — the Freddie Mac SF
LLD (§1) *is* the de facto standard; papers benchmark against each other on
it (e.g., the Bayesian default/prepayment paper on the 1999 vintage). Two
adjacent public sources worth knowing:
- **FHFA Prepayment Monitoring Report** (quarterly PDFs at fhfa.gov):
  aggregate 3-month CPR by coupon × vintage × agency. Useful as a *sanity
  anchor* for our derived CPR levels (do our cohort CPRs land in the same
  range as FHFA's published aggregates?), not as a modeling dataset.
- **Fannie Mae loan-level disclosure** (similar pipe-delimited files via
  Fannie Mae's portal): a second opinion on the same market, if we ever
  need cross-agency validation. Same registration-gated access pattern.

**Fit score: n/a** — these support §1 rather than competing with it.

---

## 6. In-repo option: Vector's own examples

The forked repo ships worked examples under `implementations/`:
`boc_rate_decisions` (Bank of Canada rate-direction prediction — binary
direction on real data, structurally the closest to our binary-outcome
framing), `sp500_forecasting`, `energy_oil_forecasting` (WTI),
`food_price_forecasting` (CPI). These are worth a look as *secondary*
real-data checks — especially `boc_rate_decisions`, where our regime
classifier and gate machinery could be tested on a real binary task with
almost no adaptation. But they are small, single-task demos, not a
replacement for a real mortgage dataset.

---

## Recommendation

**First: Freddie Mac SF LLD sample files (2019–2021 vintages).** Fit 5/5,
tractable on the 7 GB VM, exact problem shape, free registration, real
regime variation (COVID refi boom). This is the dataset that can actually
tell us whether our rankings transfer off synthetic data.

**Second (in parallel, as a quick win): M4 Monthly finance+macro subsets.**
One-hour download, zero registration, famous benchmark. It won't validate
our mortgage conclusions, but it stress-tests our probabilistic machinery
on 2k real series while the Freddie Mac downloads and registration proceed.

**Do not start with:** full Freddie Mac files (50 GB — defer until the
sample pipeline works), FRED-MD-as-benchmark (better used as the macro
data store), Monash slices (dominated on both axes).

### Top-pick download instructions (Freddie Mac sample)

1. Register for a free account at
   <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>.
2. Open the Clarity portal:
   <https://claritydownload.fmapps.freddiemac.com/CRT/#/sflld> →
   "SFLLD Data" → "Standard Dataset Download by Year".
3. Download `sample_2020.zip` first (50k-loan random sample of the 2020
   vintage — the refi-boom vintage), then `sample_2019.zip` and
   `sample_2021.zip` for baseline and post-boom comparison.
4. Unzip each into `td-one/data/freddie_mac/raw/` (gitignored — the data
   use agreement prohibits redistribution):
   - `sample_orig_YYYY.txt` — origination records
   - `sample_svcg_YYYY.txt` — monthly performance records
5. Fetch the user guide / data dictionary from the same dataset page for
   exact column positions (pipe-delimited, no headers).

### Pipeline sketch (raw → monthly panel → backtest origins)

1. **Parse origination** → static table keyed by loan sequence number:
   origination quarter, FICO, LTV, DTI, note rate, original UPB, state,
   property type, loan purpose.
2. **Parse performance** → monthly records keyed by (loan sequence number,
   reporting period YYYYMM): current UPB, delinquency status, zero balance
   code, zero balance effective date.
3. **Derive events.** For each loan, the first month with a zero balance
   code terminates the loan: `01` → voluntary prepayment (target event);
   `02`/`03`/`09`/`15` → credit event (competing risk — censor from the
   prepayment at-risk set); `06`/`16`/`96` → administrative censoring.
   Loans with no zero balance code by the last reporting period are
   right-censored (~30% expected).
4. **Build the cohort panel.** Group loans by (origination year-quarter ×
   FICO bucket × note-rate bucket). For each calendar month: count loans
   at risk at month start, count voluntary prepayments during the month →
   monthly SMM; roll to 3-month and 12-month CPR. Sanity-check aggregate
   CPR levels against the FHFA Prepayment Monitoring Report.
5. **Assemble features per cohort-month.** Trailing 3/6/12-month SMM,
   cohort WAC, burnout proxy (cumulative prior CPR), seasoning (weighted
   average loan age), plus FRED macro: MORTGAGE30US spread vs cohort WAC
   (refinance incentive), unemployment, HPI growth. All features PIT-clean
   (`record_asof <= origin`).
6. **Backtest origins.** Monthly origins from 2021-01 onward (50+
   origins): train on trailing 36 months, forecast the predictive
   distribution of next-12-month cohort CPR, score CRPS/Brier against
   realized. Run the full ladder unchanged: M0 Beta-Binomial, conformal
   M0, macro_m0, TSFMs, regime-gated agent.
7. **Compare against synthetic findings.** The key questions: does the
   m0_w6-beats-m0_w3 reversal from simulator v2 hold on real data? Does
   the M4 regime-reader still beat M0? Does the macro overlay transfer?
   (Note: exp8's "93%-of-oracle" was retracted as DGP artifact — exp15.)
   Negative answers are as valuable as positive ones — that's
   the point of leaving the bubble.

### Note on the LLM-context direction

The user's follow-up direction — giving the LLM news and data stores rather
than bare time series — maps onto this plan directly: FRED-MD (§3) becomes
the macro data store, the Freddie Mac cohort panel (§1) becomes the outcome
data, and a news corpus (to be sourced) becomes the third leg. The regime
reader's structured output (regime + confidence + reason) is the natural
interface: real macro series in, no-numbers-out, exactly as REDESIGN.md
requires.
