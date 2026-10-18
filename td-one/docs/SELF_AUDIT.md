# Self-Audit: Threats to Our Evaluation Integrity

Written 2026-10-18. Deliberately adversarial. Each item is a way our results
could be wrong or overstated, with the evidence for/against and what would
resolve it. Nothing here is hidden from the final report.

## A. Multiple comparisons (FAMILY-WISE ERROR)

**Concern:** 53 experiments, dozens of p-values, no family-wise correction.
With enough looks, something will cross p<0.05 by chance.

**Evidence for:** exp43 (p=0.018) and exp44-synthetic (p=0.0068) are the kind
of marginal results that would not survive Bonferroni over ~50 tests.

**Evidence against:** The headline claims rest on p≈1e-9–1e-11 (exp47/49),
which survive any correction. The exp42→43→44 sequence was directed by the
user *before* results were known (a pre-registered-ish trilogy), which
mitigates data-dredging for those three. exp51 (kitchen sink) was an explicit
attempt to *break* the filter and failed — the opposite of p-hacking.

**Verdict:** Headline results stand. Marginal results (exp43 alone, exp44)
should be reported with the multiple-comparison caveat, not as standalone
discoveries. **Action:** add the caveat to FINAL_SYNTHESIS.

## B. Development-on-test (M4)

**Concern:** The regime rules (exp43) and the proposed method (exp47) were
developed on M4 and evaluated on M4. The 0.1451 vs 0.1507 gap is measured on
the dataset that shaped the method.

**Evidence for:** Hand regimes (seasonal→arima etc.) were chosen after seeing
M4 behavior. This is a real optimism bias.

**Evidence against:** Food CPI (0.0141 vs 0.0176, p=3.1e-18) and BoC
(0.846→0.853) are independent datasets where the method also wins, and the
method was *not* tuned on them. SMM/Synthetic show graceful degradation
rather than collapse, which is what a non-overfit method should do.

**Verdict:** Partially mitigated but not eliminated. The honest framing is:
"developed on M4, confirmed on Food CPI/BoC." A truly held-out dataset (the
two new ones another agent is adding) is the clean test.

## C. Gate thresholds tuned with test knowledge

**Concern:** The 0.6/3.5/0.03 gate parameters were hand-picked. Were they
chosen by looking at test results?

**Evidence for:** They were set during Phase 3 development with full result
visibility. No held-out tuning split was maintained.

**Evidence against:** exp50 explicitly tested learned thresholds vs hand
values (p=0.32, no difference) and the kitchen sink's +gates ablation was
null. If the hand values were overfit, learning should have found something
better or the ablation should have hurt. It didn't.

**Verdict:** Likely fine, but the *procedure* was sloppy. Future thresholds
should be tuned on a validation split.

## D. BoC discrete "win" is one observation

**Concern:** exp50 reported BoC rule 0.846 → gated 0.853 as a win for the
discrete variant.

**Evidence:** 0.853×143 = 122.0 vs 0.846×143 = 121.0. The entire "win" is
**one additional correct prediction out of 143**. No significance test was
run. This should never have been framed as a positive result.

**Verdict:** Downgrade to "implemented, no measurable effect (n=143, Δ=+1)".
**Action:** correct the framing in PROPOSED_METHOD.md §6 and exp50_report.md.

## E. exp52 reimplemented Vector's baseline instead of running it

**Concern:** exp52 replicated `LastValuePredictor` logic by hand rather than
instantiating Vector's class.

**Evidence against:** Their logic is trivial (predict last value, degenerate
quantiles) and the replication is exact by construction. exp53 runs their
*actual* predictor class, which supersedes this concern for the LLM comparison.

**Verdict:** Minor. Noted for the record.

## F. exp53 model substitution

**Concern:** exp53 runs Vector's predictor architecture with gpt-oss-120b via
Groq, not their deployed Gemini-via-proxy model. A different model is a
different predictor.

**Evidence:** This is unavoidable (no proxy access) and disclosed. The test
measures their *elicitation architecture* (quantile-grid prompting, JSON
schema, history serialization), which is model-agnostic by design — their own
docs say participants can "swap models freely."

**Verdict:** Valid test of architecture, not of deployment. Frame it that way.

## G. LLM stochasticity in exp53 (temperature=1.0, single run)

**Concern:** Vector's predictor defaults to temperature=1.0. One run per
origin means an unknown share of the CRPS variance is sampling noise, and the
paired t-test doesn't account for it.

**Verdict:** Real limitation. A 3-seed repeat is the correct follow-up when
TPM budget allows. Report the single-run result with this caveat.

## H. CRPS scorer verified

**Check performed (this audit):** our 9-level quantile CRPS vs closed-form
normal CRPS: ratio 1.0895 ± 0.012 over 500 trials. The 9% is the standard
coarse-grid discretization bias, constant across methods — relative
comparisons are unbiased. The formula was also verified term-by-term against
the pinball-loss definition. **Scorer is correct.**

## I. PIT leakage re-verification

- News audits (date-level: 0 leakage in 33/34 checks; content-level: clean)
  were re-read; no new concerns.
- exp47/49/51 blocked CV: features computed on `train` windows only, records
  built from history before fold test origins. No future leakage found on
  re-read of `exp49_final.py` / `exp51_kitchen_sink.py`.
- exp53: Vector's `ForecastContext(as_of)` enforces cutoff at data access;
  verified in code (`data/context.py`, `data/service.py`).

## J. Reporting completeness

**For:** every experiment exp1–exp53 is recorded including kills
(`experiments/README.md` honest-negatives ledger); negatives were published
with the same prominence as positives (exp42/45/46, the news-track closure).

**Against:** the *emphasis* in summaries naturally favors wins. The
experiments index mitigates this structurally.

## K. Environment side effect (flagged, not integrity)

Installing `litellm` into td-one's `.venv` downgraded
`huggingface_hub` 2.2.0 → 1.33.0. TSFM wrapper imports still work
(verified), but weight-download behavior should be re-verified before the
next TSFM run.

## L. Pre-existing test failures (not caused by this work)

- `tests/test_decision_policy.py::test_gate_constants_sane` asserted
  `T_STAT_STRICT == 3.0` after commit eda6ab8 tightened it to 3.5 — **fixed
  in this change** (test now asserts 3.5 with the commit reference).
- 3 failures in `tests/test_fetch_fred.py` (`_FakeClient.observations()`
  missing `vintage_dates` kwarg) are pre-existing, unrelated to exp53, and
  left for the owner to decide.

## Bottom line

The headline claims (proposed method beats equal weights / best single /
exp43 alone, all p<1e-9) survive every threat examined here. The items that
bite are at the margins: the BoC discrete "win" (one observation), the
M4 development bias (mitigated by Food CPI/BoC), and exp53's single-run
stochasticity. None overturn the core result; all are disclosed.
