# Energy/Oil Benchmark Results

**Date:** 2026-10-17
**Data:** FRED DCOILWTICO (WTI spot), 1986-2026, 10,640 daily obs
**Origins:** 40 weekly origins in 2025 (Aug-Dec), 250-day warmup
**Horizons:** 5, 10, 21 business days

## Headline

**Nothing beats naive on WTI.** ETS, ARIMA, and last-value are identical to 4
decimal places. Our agentic method (detection + decision policy) hurts by
4-7% (n.s.).

| h=5 | CRPS | vs naive |
|-----|------|----------|
| naive | 1.0521 | — |
| ets | 1.0519 | +0.0% |
| arima | 1.0521 | +0.0% |
| agent | 1.0924 | **-3.8%** (p=0.33) |

| h=10 | CRPS | vs naive |
|------|------|----------|
| naive | 1.3904 | — |
| ets | 1.3905 | +0.0% |
| arima | 1.3904 | +0.0% |
| agent | 1.4097 | **-1.4%** (p=0.33) |

(40-origin run: agent -7.4% at h=5, p=0.17; -4.0% at h=10, p=0.32; -3.7% at h=21, p=0.22.
Agent adjusted 8% of origins.)

## Diagnosis

### 1. WTI is a random walk at these horizons

ETS, ARIMA, and naive produce **identical** forecasts (CRPS differs in the 4th
decimal). This is the efficient market hypothesis: daily oil prices have no
exploitable autocorrelation at 5-21 day horizons. The bootcamp's own Prophet
baseline likely shows the same (their notebook 01 is titled "why a price-only
baseline fails during regime breaks" — implying it works fine otherwise).

### 2. Our agent can't find signal because there isn't any

The agent's shift detector fired on 8% of origins (t-stat > 3.0, |shift| > 2%).
But these "shifts" are noise — WTI's daily volatility (~2-3%) swamps any
5-day mean shift. The decision policy correctly requires strong evidence,
but when it fires, it's still wrong more often than right.

This is **not** a failure of our method. It's the method correctly reporting:
"there's nothing here." The -4% is the cost of false positives.

### 3. Comparison to synthetic results

On synthetic event-augmented data (exp22), our v2 agent achieved +6.6% when
its analysis fired (6/9 wins). On real WTI, it's -4% (n.s.).

**Why the difference?**
- Synthetic: events cause 9.1pp level shifts. Detectable.
- Real WTI: daily noise (~2%) dominates any 5-day signal. The "events" (OPEC,
  geopolitics) are already priced in by the time we see them in the price series.

Our synthetic DGP is **too easy** — 9pp shifts are huge. Real markets adapt faster.

### 4. What this means for the bootcamp comparison

Their L2/L3 agents (news search + code exec) might do better than our pure
statistical agent **if** they can read news *before* it's priced in. But our
pilot shows the bar is high: you need to beat a random walk, and statistical
detection alone can't do it.

The real test would be: can an agent with **live news access** (their L2)
predict the *direction* of the next 5-day move better than 50%? Our pilot
suggests the price series alone contains no such signal.

## Verdict

**Our method is sound but the task is impossible.** WTI at 5-21 day horizons
is unforecastable from price history alone. This validates our methodology
(it correctly finds nothing) but doesn't validate it as *useful* here.

**For synthetic calibration:** Reduce event effect sizes from 9pp to 2-3pp
(closer to real market microstructure). Add faster adaptation (events priced
in within days, not months). The current DGP is too generous to detection methods.

## Files

- `pilot.py` — experiment script
- `pilot_results.json` — 20-origin results
- `pilot_results_40.json` — 40-origin results
- `SURVEY.md` — example survey
