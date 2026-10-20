# Grouping + Explainability: Worked Examples

**Source:** exp66d (seed 42), deterministic grouping (kmeans + tree-surrogate
rules) + LLM explanations with citation enforcement.
**Status:** showcase examples for reporting — a risk manager should read one
and immediately understand the cohort, its risk, and why.

## How to read an example

Each example follows the same five beats:

1. **Rule** — the deterministic, reproducible `pandas` query defining the cohort.
2. **Members** — who is in it (n, income, province).
3. **Forecast** — the v2 per-group forecast (predicted mean attrition).
4. **Risk** — two independent risk reads: deterministic training-tertile bins
   vs the LLM classifier (stats + PIT-safe news).
5. **Explanation** — the LLM's citation-grounded narrative. Every factual
   sentence ends with `[stat]` (a computed number) or `[url:N]` (a
   PIT-safe news article). The checker verified 4/4.

Honest framing first: the grouper does **not** recover the simulator's
planted cohorts (ARI 0.01 — see exp66d). What it finds is real,
reproducible structure (e.g. rate_type splits, ARI 0.24), but not the
DGP's causal joints. The explanations below are true descriptions of
real borrower segments — read them as segment narratives, not as
discoveries of the true risk drivers.

---

## Example 1 — g0: the closest thing to a planted cohort

**Rule:**
`mortgage_rate_pct <= 5.804 and government_benefit_income_monthly_cad > 225 and household_income_cad <= 158960 and children_6to17 > 0.5`

**Members:** 1,301 borrowers, mean household income 86,479 CAD, primarily Ontario.

**Forecast:** predicted mean attrition 0.0933.

**Risk:** bins say **HIGH** (train realized 0.1071) · LLM says **LOW**.

**Test realized:** 0.0702 over 114 episodes — the LLM's downgrade was the
right call.

**Cohort mapping:** best match is the planted `low_rate` cohort
(precision 0.42, recall 0.47, F1 0.45) — the rate split (`<= 5.804`)
partially captures the simulator's low-origination-rate borrowers.

**Explanation:**
> This borrower cohort consists of 1301 individuals primarily located in
> Ontario [stat]. While automated models categorize the risk level as
> high, the LLM assessment suggests a low risk of attrition [stat].
> Recent policy decisions indicate that the Bank of Canada has maintained
> stable interest rates [url:4]. Consequently, the observed attrition rate
> for this group in testing is 0.0702, which is lower than the predicted
> mean attrition of 0.0933 [stat].

**Why it matters:** this is the LLM classifier earning its keep — it
disagreed with the deterministic bin, cited stable-rate news, and the
test data vindicated it.

---

## Example 2 — g8: a mixed, genuinely high-risk segment

**Rule:**
`mortgage_rate_pct > 5.804 and adjusted_family_net_income_cad <= 118951 and mortgage_balance_cad > 332101 and mortgage_rate_pct <= 6.3255`

**Members:** 625 borrowers, mean household income 83,983 CAD, primarily Ontario.

**Forecast:** predicted mean attrition 0.0933.

**Risk:** bins say **HIGH** (train realized 0.2179) · LLM says **HIGH**.

**Test realized:** 0.1143 over 70 episodes — elevated, both reads agree.

**Cohort mapping:** best match is `control` (F1 0.32) — this group does
not correspond to any planted cohort. It is a kmeans-discovered segment:
high-rate, high-balance, mid-income borrowers.

**Explanation:**
> This cohort consists of 625 members primarily located in Ontario
> [stat]. With a mean household income of 83,983 CAD and high mortgage
> balances, these borrowers face significant pressure in the current
> mortgage market [stat][url:2]. The group has been categorized as high
> risk due to a historical realized attrition rate of 21.79% [stat].
> Although the Bank of Canada has previously maintained interest rate
> holds, this specific segment remains vulnerable to churn [stat][url:4].

**Why it matters:** the archetypal risk-report row — interpretable rule,
both risk reads agree, explanation grounds the HIGH label in the 21.79%
training rate and balance pressure.

---

## Example 3 — g5: the affluent segment the bins misread

**Rule:**
`mortgage_rate_pct <= 5.804 and government_benefit_income_monthly_cad > 225 and household_income_cad > 158960`

**Members:** 316 borrowers, mean household income 221,424 CAD, primarily Ontario.

**Forecast:** predicted mean attrition 0.0933.

**Risk:** bins say **MEDIUM** (train realized 0.0645) · LLM says **LOW**.

**Test realized:** 0.0312 over 32 episodes — very low attrition.

**Cohort mapping:** none — g5 is not any planted cohort's best match.
It is a pure income split the tree found.

**Explanation:**
> This cohort consists of 316 members with a high mean household income
> of 221,424 CAD [stat]. While the predicted mean attrition for this
> group is 0.0933, the realized attrition in our test episodes has been
> significantly lower at 0.0312 [stat]. These borrowers benefit from
> mortgage rates below 5.804%, a stable environment supported by the
> Bank of Canada's recent decisions to hold interest rates steady
> [url:0]. Consequently, this cohort is categorized as having a low risk
> level regarding potential attrition [stat].

**Why it matters:** shows the LLM using the income stat + rate news to
correctly push below the bin label. Also shows grouping surfacing a
segment (affluent, low-rate) that is managerially meaningful even
though it matches no planted cohort.

---

## Example 4 — g3: the honest failure

**Rule:**
`mortgage_rate_pct <= 5.804 and government_benefit_income_monthly_cad > 225 and household_income_cad <= 158960 and children_6to17 <= 0.5`

**Members:** 242 borrowers, mean household income 84,373 CAD, primarily Ontario.

**Forecast:** predicted mean attrition 0.0933.

**Risk:** bins say **LOW** (train realized 0.0000) · LLM says **LOW**.

**Test realized:** 0.1667 over 24 episodes — the **highest** of any group.

**Cohort mapping:** none.

**Explanation:**
> This cohort consists of 242 members primarily located in Ontario with
> a mean household income of $84,373 [stat]. Although the predicted mean
> attrition rate for this group is 0.0933, recent testing episodes
> revealed a realized attrition rate of 0.1667 [stat]. The Bank of Canada
> has maintained a consistent stance on interest rates, which influences
> the broader mortgage market environment [url:0] [url:4]. Consequently,
> while the risk level is currently categorized as low, market
> participants should remain aware of potential shifts in borrower
> behavior [stat].

**Why it matters:** both risk reads agreed on LOW and both were wrong —
train realized was literally 0.0000, test came back 16.67%. With 24
episodes this is partly small-sample noise, but it is also the core
exp66c finding in miniature: **static-attribute group attrition does not
transfer across windows.** We include it because a showcase that hides
its failures is a brochure, not a report.

---

## What the four examples show together

| Group | Rule theme | Bins | LLM | Test realized | Lesson |
|---|---|---|---|---|---|
| g0 | low-rate + kids | HIGH | LOW ✅ | 0.070 | LLM+news can beat bins |
| g8 | high-rate + high-balance | HIGH | HIGH ✅ | 0.114 | Agreement on real risk |
| g5 | affluent | MEDIUM | LOW ✅ | 0.031 | LLM corrects downward |
| g3 | low-rate, no kids | LOW | LOW ❌ | 0.167 | Both wrong — signal doesn't transfer |

The pattern: the LLM classifier is directionally better calibrated than
training-tertile bins (monotonic HIGH > MEDIUM > LOW on test vs the
bins' inversion), but n=10 groups is too small for a significance
claim. And g3 is the permanent caveat — no classifier built on static
attributes survives a regime where attrition is driven by time-varying
factors.
