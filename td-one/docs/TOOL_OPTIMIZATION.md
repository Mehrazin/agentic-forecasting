# Tool Optimization: Recommendations

**Date:** 2026-10-18
**Source:** `docs/TOOL_AUDIT.md`
**Status:** Proposed — implement before final report

---

## 1. New Tools to Add

### 1a. `load_series` (HIGH PRIORITY)
**Why:** Every agent code block starts with 3 lines of pandas boilerplate to load and filter by origin date. This is 100% repeated.

**Signature:**
```python
def load_series(csv_path: str, date_col: str = "date", value_col: str,
                origin: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Load time series, optionally filtered to <= origin. Returns (values, date_strings)."""
```

**Impact:** Eliminates ~5 lines from every agent program. Reduces copy-paste errors (date filtering bugs are silent and catastrophic for PIT safety).

### 1b. `naive_forecasts` (HIGH PRIORITY)
**Why:** Every experiment reimplements naive, seasonal-naive, and trailing-mean from scratch. These are 15-20 lines each, repeated across exp35, exp36, exp33.

**Signature:**
```python
def naive_forecasts(train: np.ndarray, horizon: int, period: int = 12) -> dict[str, dict[float, float]]:
    """Return {model_name: {quantile: value}} for naive, seasonal-naive, trailing-mean."""
```

**Impact:** Standardizes baselines. Currently each reimplementation has slightly different sigma estimation — this is a comparability bug.

### 1c. `combine_quantiles` (MEDIUM PRIORITY)
**Why:** `calibrated_quantiles()` exists in `decision_policy.py` but is NOT in the T-namespace. Agents can't call it. exp42 implemented Vincentized combination manually.

**Action:** Expose existing function to T-namespace. Zero new code, just wiring.

**Signature:** (already exists)
```python
def calibrated_quantiles(point: float, sigma: float, ...) -> dict[float, float]
```

### 1d. `regime_probabilities` (MEDIUM PRIORITY)
**Why:** exp43's hand-rule regime classifier is now our best combination method. It's currently embedded in `exp43_regime_router.py`, not reusable.

**Signature:**
```python
def regime_probabilities(features: dict[str, float]) -> dict[str, float]:
    """Map diagnostic features to P(regime). Regimes: seasonal, trending, noisy."""
```

**Impact:** Makes exp43's winning method available to future agents.

---

## 2. Tools to Optimize

### 2a. Wire in `tool_tracker.py`
**Problem:** The tracker exists but is never called. We have zero runtime data.

**Fix:** Add 3 lines to each tool's entry point:
```python
from td_one.agent.tool_tracker import tracker
import time
_start = time.perf_counter()
# ... existing logic ...
tracker.record("tool_name", (time.perf_counter() - _start) * 1000, {...}, str(result)[:500], success=True)
```

**Better:** Use a decorator to avoid repetition:
```python
@track_tool("detect_changepoints")
def detect_changepoints(...): ...
```

### 2b. `test_stationarity` — add fast path
**Problem:** Runs both ADF and KPSS. KPSS is slow on long series (>500 obs).

**Fix:** Add `quick: bool = False` parameter. When True, run ADF only (fast, sufficient for most agent decisions).

### 2c. `decompose_series` — validate inputs
**Problem:** STL fails cryptically on short series or invalid period.

**Fix:** Add explicit validation with clear error messages:
```python
if len(series) < 2 * period:
    raise ValueError(f"Need >= {2*period} observations for period={period}, got {len(series)}")
```

---

## 3. Tools to Deprecate

### 3a. `plot_diagnostics` — REMOVE
**Evidence:** Zero call sites. The one experiment that lists it (exp36) notes "PNG path only; nothing consumes it downstream."

**Action:** Remove from T-namespace. Keep the module for manual use, but agents don't need it.

### 3b. `generate_synthetic_scenario` — REMOVE
**Evidence:** Zero call sites. Listed in exp36 as "what-if shocks; not needed for point backtest."

**Action:** Remove from T-namespace. This was speculative; no experiment found a use.

### 3c. `feature_importance` — KEEP BUT SIMPLIFY
**Evidence:** Only 2 call sites (exp46, exp35). But exp46 used it meaningfully for news feature analysis.

**Action:** Keep, but document when to use it (post-hoc analysis, not in-loop).

---

## 4. Missing Capabilities

What agents try to do but can't:

| Need | Current workaround | Proposed tool |
|---|---|---|
| Load data by origin | 3-line pandas boilerplate | `load_series` (1a) |
| Get baseline forecasts | Reimplement naive (20 lines) | `naive_forecasts` (1b) |
| Combine quantile forecasts | Manual Vincentization | `combine_quantiles` (1c) |
| Classify regime | Copy exp43 code | `regime_probabilities` (1d) |
| ACF/PACF features | Not available | Add to `decompose_series` output |
| Entropy, Hurst, lumpiness | Not available | New `tsfeatures` tool (see exp43 gap analysis) |

---

## 5. Implementation Priority

| Priority | Action | Effort | Impact |
|---|---|---|---|
| P0 | Wire `tool_tracker` decorator | 1 hour | Enables all future optimization |
| P0 | Remove `plot_diagnostics`, `generate_synthetic_scenario` | 15 min | Cleaner namespace, less confusion |
| P1 | Add `load_series` | 1 hour | Eliminates #1 boilerplate |
| P1 | Add `naive_forecasts` | 2 hours | Standardizes baselines |
| P1 | Expose `calibrated_quantiles` to T-namespace | 15 min | Zero new code |
| P2 | Add `regime_probabilities` | 2 hours | Reusable exp43 |
| P2 | `test_stationarity` fast path | 30 min | Speed |
| P2 | `decompose_series` validation | 30 min | Robustness |
| P3 | `tsfeatures` tool (ACF, entropy, Hurst) | 4 hours | Closes exp43 gap analysis |

**Total:** ~11 hours for P0-P2. P3 is optional.

---

## 6. T-Namespace After Optimization

**Keep (8):**
- `analyze_residuals`
- `detect_changepoints`
- `test_stationarity`
- `decompose_series`
- `backtest_event_impact`
- `cross_validate`
- `compare_models`
- `feature_importance`

**Remove (2):**
- `plot_diagnostics` ❌
- `generate_synthetic_scenario` ❌

**Add (4):**
- `load_series` ✅
- `naive_forecasts` ✅
- `combine_quantiles` ✅ (existing, just expose)
- `regime_probabilities` ✅

**Net:** 10 → 12 tools, but 2 removed are dead weight and 4 added eliminate boilerplate.
