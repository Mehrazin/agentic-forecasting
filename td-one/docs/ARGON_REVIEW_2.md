# Argon Feedback Review — Second Pass (2026-10-17)

Re-reading the 6 findings to check: (1) what's already addressed, (2) where Argon is mistaken, (3) what we missed.

## Finding 1 (P0): R2 macro bug — ✅ FULLY ADDRESSED, Argon correct

Fixed in d91d5fe. Rerun complete:
- r2_exp10: lagged macro_m0 1.923 vs m0_w6 2.28 (-0.357, ~53% of leaky gain)
- r2_exp13: Chronos2 + lagged macro hurts 4/6 (residual approach structurally unsuited)
- r2_exp15: oracle hurts on v2 DGP (DGP-specific)

**Clarification:** Argon's mechanism explanation ("subtracted β·z_t, never added back") was correct for the buggy code. But the fix revealed something Argon didn't predict: even with correct lagged re-addition, the residual conditioning approach hurts Chronos2. The method is structurally unsuited to PIT-safe lagged conditioning, not just buggy. Direct logit shift (macro_m0) works; residual conditioning doesn't.

## Finding 2 (P0): Simulator pre-repricing — ✅ FULLY ADDRESSED, Argon correct

**What we did (d91d5fe + 6dacafa):**
1. Changed `snap_date >= maturity` to `snap_date > maturity` — maturity-month snapshot retains expiring contract's rate/payment.
2. `generate_outcomes()` now looks up the maturity-month snapshot via `by_mid_month[(mid, month_end)]` for `payment_shock` computation, instead of using `last[mid]` (month-66, post-repricing). Falls back to `last` with a warning if missing.
3. Outcome records use `cov_rec.renewal_episode_id` (maturity-month episode 01) instead of `rec.renewal_episode_id` (post-renewal -02).

All 10 simulator tests pass.

## Finding 3 (P1): Segment discovery — ⚠️ PARTIALLY ADDRESSED

- Transductive qcut: ✅ FIXED (fit on `<= observed_asof`, apply via `pd.cut`)
- Planted cohort dilution: ❌ NOT DONE — needs `discover_candidate_groups()` for multi-attribute cohorts
- Known-N: ❌ NOT DONE — M0 still uses estimated `n_upcoming` instead of known `N_{s,m}`

Argon is correct on all three. The cohort discovery is the most impactful for the "which groups need attention" question.

## Finding 4 (P1): PITStore gaps — ❌ NOT ADDRESSED, Argon correct

- 12 cols vs 46 in schema: not done
- Pending rows for episodes between maturity_date and obs_date: not done

Lower priority than Finding 2's second half, but still a real gap.

## Finding 5 (P2): CRPS diagonal bias — ✅ FULLY ADDRESSED, Argon correct

Fixed in d91d5fe with exact sorted L-moment formula. All 27 tests pass.

## Finding 6 (P1): Manager report — ❌ NOT ADDRESSED, Argon correct

exp18 manager report not yet built. This is the flagship deliverable for the bootcamp rubric.

## Where Argon was mistaken (or overstated)

1. **Track E framing:** Argon presents exp12b v2 as a positive "textbook trace analysis" (2.595 → 2.488). We killed M5 because v2 still loses to inverse-CRPS (2.488 vs 2.322). Argon doesn't dispute the kill, but the framing is rosier than our "dead" verdict. Both are defensible; ours is stricter.

2. **Finding 2's scope:** Argon bundled two distinct bugs (snapshot timing + outcomes lookup) under one finding. We fixed the first, missed the second. The finding wasn't wrong, but the "one-line fix" framing understated the work.

3. **R2-4 priority:** Argon lists exp14 rerun as "planned" alongside the macro fixes. Given that exp14's ~20% TSFM win is on real data (not affected by the FRED bug), the rerun is lower priority than Argon implies. The soundness concern (common CRPS grid) is real but doesn't invalidate the directional finding.

## Summary

| Finding | Status | Argon correct? |
|---|---|---|
| 1. R2 macro bug | ✅ Fixed + rerun | Yes |
| 2. Simulator repricing | ✅ Fully fixed | Yes |
| 3. Segment discovery | ⚠️ 1/3 fixed | Yes |
| 4. PITStore gaps | ❌ Not done | Yes |
| 5. CRPS bias | ✅ Fixed | Yes |
| 6. Manager report | ❌ Not done | Yes |

**Immediate action:** Fix Finding 2's second half (generate_outcomes lookup).
