# Tool Call Audit: Frequency, Patterns, Failures

**Date:** 2026-10-18
**Scope:** exp21–exp47, benchmarks (boc_rates, smm, food_cpi, m4)
**Method:** Static analysis (grep) — the `tool_tracker.py` singleton was built but never wired into any tool, so no runtime logs exist.

---

## 1. Tool Frequency Table

| Tool | Module | Files using | Call sites | T-namespace? |
|---|---|---|---|---|
| `detect_changepoints` | regimes.py | 9 | 10 | ✅ |
| `test_stationarity` | analysis.py | 8 | 8 | ✅ |
| `decompose_series` | analysis.py | 6 | 6 | ✅ |
| `analyze_residuals` | analysis.py | 6 | 4 | ✅ |
| `compare_models` | evaluation.py | 3 | 5 | ✅ |
| `backtest_event_impact` | regimes.py | 4 | 3 | ✅ |
| `cross_validate` | evaluation.py | 3 | 3 | ✅ |
| `feature_importance` | analysis.py | 3 | 2 | ✅ |
| `plot_diagnostics` | plotting.py | 2 | **0** | ✅ |
| `generate_synthetic_scenario` | scenarios.py | 2 | **0** | ✅ |

**Key finding:** `plot_diagnostics` and `generate_synthetic_scenario` are in the T-namespace and mentioned in experiment comments, but have **zero actual call sites**. They are dead weight.

---

## 2. Usage by Experiment

| Experiment | Tools used |
|---|---|
| exp21 (feature discovery) | analyze_residuals, detect_changepoints, cross_validate, compare_models |
| exp22 (M2 v2) | detect_changepoints, test_stationarity |
| exp23 (model selection) | detect_changepoints, test_stationarity, decompose_series |
| exp24 (regime aware) | detect_changepoints, test_stationarity, decompose_series, backtest_event_impact |
| exp33 (SMM agent) | analyze_residuals, detect_changepoints, test_stationarity |
| exp34 (SMM events) | backtest_event_impact |
| exp35/36 (Food CPI) | **all 10** (only experiment to use the full set) |
| exp37 (M4 selection) | detect_changepoints, test_stationarity, decompose_series |
| exp42 (FFORMA) | detect_changepoints, test_stationarity, decompose_series |
| exp46 (news dense) | feature_importance |

**The "diagnostic trio"** (`detect_changepoints` + `test_stationarity` + `decompose_series`) appears together in 6/10 experiments. This is the standard exploration pattern.

---

## 3. Model-Written Code Patterns

The agents in these experiments are **scripted policies**, not live LLMs. But the embedded code strings reveal what a real LLM would need to write:

### Pattern A: Data loading boilerplate (every experiment)
```python
df = _pd.read_csv("__CSV__", parse_dates=["date"])
df = df[df["date"] <= "__ORIGIN__"]
x = df["__COL__"].to_numpy()
```
**Appears in:** 100% of agent code blocks. **Should be a tool.**

### Pattern B: Quantile forecast construction (every experiment)
```python
def _q(point, sigma):
    return {0.1: point - Z * sigma, 0.5: point, 0.9: point + Z * sigma}
```
**Appears in:** exp36, exp35, exp33. **Should be a tool** (`calibrated_quantiles` exists in decision_policy.py but isn't in the T-namespace).

### Pattern C: Naive forecast baselines (repeated)
```python
def naive(train, hh):
    s = float(_np.std(_np.diff(train[-25:]))) * _np.sqrt(hh)
    return _q(float(train[-1]), s)
```
**Appears in:** exp36, exp35. Every agent reimplements naive/seasonal-naive/trail-mean. **Should be a tool.**

### Pattern D: One-step residual loop (diagnostics)
```python
for i in range(len(x) - 24, len(x)):
    tr = x[:i]
    yt.append(float(x[i]))
    yp.append(float(one_step(fns[winner], tr)))
ra = T.analyze_residuals(yt, yp)
```
**Appears in:** exp36. **Partially covered** by `analyze_residuals`, but the loop itself is boilerplate.

### Libraries imported by model code
- `pandas` (100%), `numpy` (100%), `json` (100%)
- `statsmodels` — **never directly** (always via tools). Good — tools are the right abstraction.
- `sklearn` — **never**. The meta-learning experiments (exp41/42) used it in driver scripts, not agent code.

---

## 4. Failure Analysis

**No runtime failure logs exist** (tracker not wired). From code inspection:

| Tool | Potential issue |
|---|---|
| `test_stationarity` | KPSS can be slow on long series; no timeout guard in tool itself (relies on sandbox 30s) |
| `decompose_series` | STL requires `period >= 2` and `len >= 2*period`; degenerate input → exception (caught? check) |
| `detect_changepoints` | Binary segmentation is O(n log n); fine for our sizes |
| `plot_diagnostics` | Returns PNG path; **nothing consumes it** in any experiment — dead output |
| `generate_synthetic_scenario` | **Never called** — speculative tool with no user |

**Sandbox:** 30s default timeout, 1GB memory. No evidence of timeouts in practice (experiments use 60-120s for agent code blocks, tools themselves are fast).

---

## 5. The Tracker Gap

`src/td_one/agent/tool_tracker.py` (created 2026-10-17) is **fully implemented but never imported** by any tool or experiment. Zero call sites reference it.

**Impact:** We have no data on:
- Actual wall-clock time per tool call
- Failure rates in production
- Which tools correlate with good outcomes
- Input patterns (what arguments are actually passed)

**This audit is static-only as a result.**
