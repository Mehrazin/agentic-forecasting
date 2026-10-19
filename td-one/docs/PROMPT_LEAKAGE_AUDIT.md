# Prompt Leakage Audit — LLM integration ladder (exp55–exp57)

**Auditor:** Muse (subagent of the TD-One orchestrator)  
**Date:** 2026-10-18  
**Scope:** every template field of `td_one/llm_rungs/prompts.py`, used by
`experiments/exp55_run_llm.py`, `exp56_run_llm.py`, `exp57_run_llm.py`.

**Convention.** For origin index `orig`, `train = values[:orig]` (strictly
`<= origin`, i.e. the actual for this origin is `values[orig]` and is never
visible), `as_of = date[orig-1]` (last observed month), `forecast date =
date[orig]`. Origins = `range(n-60, n-1)` on `benchmarks/food_cpi/data.csv`
(n=200; as_of 2021-08-01 … 2026-07-01).

## Verdict

✅ **No leakage.** Every template field is a pure function of (dataset,
`as_of`, history `<= as_of`). No actuals, no post-origin values, no post-origin
news, and no statistics computed past `as_of` appear in any prompt. Details
below.

## Field-by-field trace

### `serialize_history(values, dates, precision=2)`

| Field | Data source | ≤ origin proof |
|---|---|---|
| Each `YYYY-MM: value` line | `benchmarks/food_cpi/data.csv`, sliced `[:orig]` in the run scripts | The slice excludes `values[orig]` (the actual) and everything after by construction; the audit asserts this mechanically in the scripts (`train = values[:orig]`) |

### `system_prompt_forecaster()` / `system_prompt_calibrator()`

Timeless instructions (JSON shape, monotone quantiles, same units). No data fields at all. ✅

### `user_prompt_generic` (rung 2, exp55 — exp53 replication)

| Field | Data source | ≤ origin proof |
|---|---|---|
| `task_line` | Hardcoded: "Canada Food CPI 1-month-ahead forecast." | Timeless task description, no numbers |
| `series_desc` / `units` / `frequency` | Hardcoded: "Canada Food CPI (headline), monthly index (source: StatCan)" / "index" / "MS" | Series metadata; identifies the series, contains no observations |
| `history_str` | `serialize_history` of `train` | ≤ origin (above) |
| `forecast_start` / `forecast_end` | `date[orig]` formatted | This is the *forecast target* date, not an observation; knowing the target month is inherent to the forecasting task (Vector's prompt does the same) |

**Adaptation vs Vector's `_build_user_prompt`:** identical shape (task line, meta
block, history, forecast instruction). Two deliberate changes: (1) "MS" frequency
rendered as a "Forecast the next 1 value" line instead of their f-string `"{n}
{frequency} values"` ("1 MS values" reads badly); (2) output contract expressed
as an inline JSON shape instead of Vector's provider-side JSON-schema object,
because our direct Groq client only supports `json_object` mode — the field set
(q05…q95), array-length rule, and monotonicity rule are unchanged, and
`runner.parse_grid` sorts each returned row ascending to enforce monotonicity
exactly as Vector's `_quantile_grid_from_response` does.

### `user_prompt_calibrator` (rung 3, exp56)

Adds one block: the classical 9-quantile grid rendered by
`calibrator_grid_text(grid, levels)` as `level: value` lines.

| Field | Data source | ≤ origin proof |
|---|---|---|
| Classical 9-quantile grid | `td_one.agent.tools.data.naive_forecasts(train, horizon=1, period=12).seasonal_naive[0]` | `train = values[:orig]`; seasonal-naive point = `train[-12]`, sigma = std of `diff(train)` — all ≤ origin. The exp56 script asserts `median == values[orig-12]` and `sigma == std(diff(train))` on the first origin before running |
| Everything else | Same as `user_prompt_generic` | Same proofs |

The calibrator system prompt frames the LLM as adjusting a statistical grid
rather than forecasting from scratch; output contract is identical to the
forecaster's.

### `user_prompt_v1` (rung 4 V1, exp57) — "Recent behavior (in words)"

Adds the `words_stats_block` (rendered via the `suffix` seam, mirroring
Vector's `user_prompt_suffix` design):

| Field | Data source | ≤ origin proof |
|---|---|---|
| Last observed value + date | `train[-1]`, `as_of` | ≤ origin by construction |
| 12-month % change | `train[-1] / value@date12 − 1`, where `date12` is the observation dated exactly 12 calendar months before `as_of` (date-indexed lookup, not positional); the reference month is printed as "(vs YYYY-MM)" | Both operands ≤ origin; the series skips 2025-10-01, so positional `train[-13]` would be a 13-month span for origins ≥ 190 — the calendar lookup avoids this. If the exact month is missing, the latest earlier month is used and printed (affects only the 3m change for as_of 2026-01-01, labeled "vs 2025-09") |
| 3-month % change | Same, `date3` = 3 calendar months before `as_of` | Same |
| Max / min of last 12 calendar months | `max/min` over observations with `date12 < date ≤ as_of` | Same window, ≤ origin |

**Data quirk (documented):** `benchmarks/food_cpi/data.csv` has no 2025-10-01
row (2025-09-01 → 2025-11-01, the only >35-day gap). All 12m/3m lookups are
date-indexed, so no origin's window is misaligned; for our origin range
(as_of 2021-08-01 … 2026-07-01) the 12m-ago and 3m-ago months always exist.
The exp53 generic replication is unaffected (it uses the full serialized
history, no window statistics).

No numbers are invented: the block only restates what the history already
contains, in words the LLM can use without re-deriving.

### `user_prompt_v2` (rung 4 V2, exp57) — V1 + macro context

Adds `macro_context_block(facts, as_of)`. Every fact carries its own as-of
date; the block states "nothing here is newer than {as_of}".

| Fact | Value source | Fact as-of | ≤ origin proof |
|---|---|---|---|
| Bank of Canada policy rate | `data/fred/IRSTCB01CAM156N.parquet` — last row with `date <= as_of` (strict inequality on timestamps; never after) | That row's own date | By construction of the lookup; the fact's as-of date is printed in the prompt |
| Canada Food CPI 12-month inflation | Computed `train[-1]/train[-13] − 1` from the pre-origin history | `as_of` | Same as the V1 12m change |

**Known limitation (documented, not a leak):** `IRSTCB01CAM156N` ends
2023-12-01, so origins after 2023-12 reuse the 2023-12-01 value with its true
as-of date printed and a "(latest available; treat as stale)" marker. This is
PIT-safe (the fact is genuinely ≤ origin) but stale — the alternative, live
BoC data, is not in the repo. The staleness is disclosed to the LLM in-prompt
and here.

## Macro lookup table (versioned)

| Fact | Source file | Coverage in repo | Lookup rule | As-of labeling |
|---|---|---|---|---|
| BoC policy rate | `td-one/data/fred/IRSTCB01CAM156N.parquet` | 1955-01 … 2023-12 (NaN at series start) | `max{date ≤ as_of}` | Fact's own date, printed in prompt; stale marker when >6 months old |
| Food CPI 12m inflation | `td-one/benchmarks/food_cpi/data.csv` (same series) | 2010-01 … 2026-09 | computed from `train` | `as_of` |

## Explicit negative statements

1. **No actuals.** The forecast target `values[orig]` is loaded for scoring
   only; it is never passed to any prompt builder. `train` slices end at
   `orig-1`.
2. **No post-origin values.** History serialization, all V1 statistics, and the
   classical grid are computed on `values[:orig]`. The BoC lookup uses
   `date <= as_of` (timestamp comparison, not month rounding).
3. **No news.** No news corpus, articles, or headlines are referenced by any
   rung-4 template; the V2 macro block contains only the two tabled facts.
4. **No dataset statistics computed over the full sample.** Means, ranges,
   and changes are all windowed on `train`; nothing like "the series' all-time
   max" enters a prompt.
5. **No model internals.** Prompts contain no CRPS values, no win rates, no
   mention of other experiments.

## How to re-verify

- `experiments/exp56_run_llm.py` prints a snaive parity check (median ==
  `y[t-12]`, sigma == `std(diff(train))`) on origin 0 before any LLM call.
- `experiments/exp57_run_llm.py` asserts the 12-month window alignment
  (`date[orig-13]` is exactly 12 months before `as_of`) for every origin.
- All three scripts are resume-safe: caches are rewritten after every origin,
  so any origin's prompt inputs can be recomputed deterministically from
  (data.csv, origin index) and re-checked.

---
*Signed: Muse (TD-One LLM-rungs subagent), 2026-10-18.*
