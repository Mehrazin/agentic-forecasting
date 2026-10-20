# exp66c: Deterministic Grouping + LLM Explanations — Report

**Date:** 2026-10-19
**Status:** Complete. Implements the exp66b pivot.
**Thesis:** LLM group *discovery* is flaky (not reproducible); deterministic
clustering + tree-surrogate rules replace it. The LLM keeps its role in
*explanation* generation via the citation-enforced `risk_reporter`.

## What was built

- `src/td_one/agent/deterministic_grouper.py` — KMeans (fixed seed) on
  standardized borrower metadata → shallow `DecisionTreeClassifier`
  predicts cluster labels → each leaf becomes a `GroupDefinition` whose
  rule is the root-to-leaf path (thresholds un-standardized to raw
  units). All rules pass `validate_rule()`. Deterministic: same input +
  seed → identical groups (tested).
- `experiments/exp66c_deterministic_groups.py` — 4 arms (A global, B
  random, C hand-written, G deterministic), per-group
  `ProposedForecasterV2`, LLM explanations with citation enforcement,
  risk-bin calibration.
- `tests/test_deterministic_grouper.py` — 7 tests (reproducibility, rule
  validity, full coverage, no outcome leakage, rule↔assignment
  consistency).

PIT: grouper fit on training-period metadata only (last snapshot with
`snapshot_date < cutoff`, all borrowers); test assignments use
training-period attributes.

## Results

### CRPS (member-weighted)

| Arm | seed 42 | seed 123 |
|---|---|---|
| A: Global | 0.00042 | 0.01006 |
| B: Random | 0.02851 | 0.02729 |
| C: Hand-written | 0.00711 | 0.01297 |
| G: Deterministic | 0.01679 | 0.01898 |

Grouping costs CRPS: G is worse than global (A) and hand-written (C) on
both seeds, but beats random (B) by ~40%. Cost of G vs A: +0.016 / +0.009.
Cost of G vs C: +0.010 / +0.006.

### Reproducibility

- Same seed + same data → byte-identical groups and assignments
  (unit test, 2/2 runs).
- Different seeds → different groups (expected: different simulated
  data). Rules remain interpretable in all runs (see below).

### Rule readability

10 groups, 2–4 conditions per rule (mean 3.4), 2–4 distinct attributes.
Example rules (seed 42):

- `g6`: `mortgage_rate_pct > 5.804 and adjusted_family_net_income_cad > 118951`
- `g1`: `mortgage_rate_pct <= 5.804 and government_benefit_income_monthly_cad <= 225 and borrower_income_cad <= 84754.2 and mortgage_payment_monthly_cad <= 2507`

Human rubric (1–5, applied by the author to all 10 rules):

| Criterion | Score | Note |
|---|---|---|
| Interpretability (could a risk manager paraphrase it?) | 4 | Thresholds are plain CAD/%; `children_6to17 > 0.5` is awkward (means ≥1 child) |
| Actionability (does it suggest a lever?) | 3 | Rate/income/payment splits map to product levers; benefit-income is proxy-ish |
| Parsimony (≤4 conditions) | 5 | Max 4 conditions, all used |
| No jargon / no leakage | 5 | All attributes are plain metadata; no outcome columns |

### Explanation quality

- Citation pass rate: **10/10** with the tightened contract ("every
  sentence must end with [stat] or [url:N]"). With the module-default
  contract, interpretive closing sentences ("suggests a stable borrower
  profile") fail the checker 50% of the time — the checker cannot
  distinguish interpretation from factual claims. Lesson: the contract
  must be total, not "factual sentences only".
- Explanations are grounded in computed stats (n, income, province,
  realized/predicted attrition, risk level) — no invented numbers observed.

### Risk-bin calibration (honest negative)

Bins from training-period realized attrition tertiles (PIT-clean).
Binning the v2 *predicted* means is degenerate: per-group forecasts are
near-identical, tertiles collapse, all 10 groups land HIGH. That itself
is a finding — the forecaster does not differentiate group risk.

Test-period realized attrition by bin:

| Bin | n groups | train realized | test realized |
|---|---|---|---|
| HIGH | 4 | 0.189 | 0.087 |
| MEDIUM | 3 | 0.082 | 0.068 |
| LOW | 3 | 0.030 | 0.128 |

**Inverted.** Training-period group attrition does not transfer to the
test window: switching is driven by time-varying factors (rate path),
not static group membership. Group-level risk bins from static
attributes are not calibrated — do not ship them as predictions.

## Sample risk report

See `exp66c_sample_risk_report.json` for the full machine-readable
report (10 groups). Rendered excerpt for one evaluation window:

> **Cohort g2** — `mortgage_rate_pct <= 5.804 and
> government_benefit_income_monthly_cad <= 225 and
> borrower_income_cad <= 84754.2 and mortgage_payment_monthly_cad > 2507`
> **Risk: MEDIUM** (n=209, train realized 0.083, test realized 0.071 over
> 14 episodes, predicted 0.093)
>
> "This cohort, consisting of 209 members, is currently classified as
> having a medium risk level [stat]. The members of this group have a
> mean household income of 79,468.0 CAD and are primarily located in
> Ontario [stat]. While the predicted mean attrition rate for this group
> is 0.0933, the realized attrition rate observed during training was
> 0.0833 [stat]. Recent testing across 14 episodes has shown a realized
> attrition rate of 0.0714 [stat]."

Every sentence carries a `[stat]` citation; the checker verified 10/10.

## Verdict

✅ **Pivot validated on its own terms.** Deterministic grouping delivers
what the LLM grouper could not: reproducible, interpretable cohorts
(10 groups, ≤4 plain-English conditions each) with citation-grounded LLM
explanations at 100% pass rate.

⚠️ **Two honest negatives inside the win:**
1. Grouping costs CRPS vs global (+0.009–0.016) and vs hand-written
   (+0.006–0.010). The explainability price is quantified.
2. Static-attribute risk bins do not calibrate: LOW-bin groups realized
   *higher* test attrition than HIGH-bin groups. Risk levels from
   training tertiles should not be presented as forward-looking.

**Recommendation:** keep deterministic grouping + LLM explanations as the
explainability layer (risk reports, cohort narratives); do not use
group-level forecasts or static risk bins for decisions. The forecaster
of record stays global (arm A).
