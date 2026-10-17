# Phase 2 Build Summary: Code Execution + Sandboxing + 10 Tools

**Date:** 2026-10-17
**Status:** Complete — 66 tests pass, ruff clean, mypy strict clean. Not committed.

## What was built

### A. `src/td_one/agent/sandbox.py` (new, ~400 lines)
Ported from AI2's `metaharness/sandbox.py` with TD-One hardening:
- **Subprocess backend**: wall-clock timeout (30s default), POSIX resource ceilings
  (1GB address space, 60s CPU, 256MB files, 64 fds), per-call scratch dir under /tmp
- **Docker backend**: `--network none`, read-only rootfs, memory/cpu caps (when available)
- **Secret stripping**: proxy vars + anything matching KEY/TOKEN/SECRET/PASSWORD/CREDENTIAL
  removed from child env; verified child cannot see parent secrets (test)
- **Path traversal guards**: only /tmp and ~/workspace allowed
- **Byte caps**: 256KB on stdout/stderr (bloat/injection defense)
- **Audit logging**: every run logged with SHA-256 of source, timing, outcome
- Key fix during build: 512MB default was too tight for statsmodels import chain → 1024MB

### B. `src/td_one/agent/tools/` (new package, 10 tools)
| # | Tool | Module | Notes |
|---|---|---|---|
| 1 | `analyze_residuals` | analysis.py | BP heteroskedasticity test, top-3 residual drivers |
| 2 | `detect_changepoints` | regimes.py | Binary segmentation, BIC penalty (no ruptures dep) |
| 3 | `feature_importance` | analysis.py | Permutation importance, model-agnostic (no sklearn/SHAP) |
| 4 | `cross_validate` | evaluation.py | PIT-safe rolling-origin CV, CRPS ± SE |
| 5 | `compare_models` | evaluation.py | Same splits + Diebold-Mariano with HLN correction |
| 6 | `plot_diagnostics` | plotting.py | 4-panel PNG (resid vs time, QQ, ACF, pred vs actual) |
| 7 | `test_stationarity` | analysis.py | ADF + KPSS, combined verdict |
| 8 | `decompose_series` | analysis.py | STL + trend/seasonal strength metrics |
| 9 | `backtest_event_impact` | regimes.py | Per-event ATE vs pre-event baseline |
| 10 | `generate_synthetic_scenario` | scenarios.py | 4 shock types for counterfactuals |

Each tool: Pydantic-validated I/O, PIT-safety where temporal, clear errors, unit tests.

### C. `src/td_one/agent/react_v2.py` (new, ~250 lines)
Code-executing ReAct loop:
- New `code_execute` action runs Python in the sandbox
- `T` namespace pre-imports all 10 tools + numpy/pandas
- v1 tools (data_segment/evidence/forecasting) still work for backward compat
- Default max_steps 10 (vs v1's 6) since analysis takes more steps

## Test results
- **47 new tests**: 17 sandbox + 21 tools + 9 react_v2 — all pass
- **19 existing tests** (test_agent, test_exp3_m2) — still pass
- **ruff check**: clean | **ruff format**: clean | **mypy strict**: clean

## Files
- `src/td_one/agent/sandbox.py` (new)
- `src/td_one/agent/react_v2.py` (new)
- `src/td_one/agent/tools/__init__.py` (new)
- `src/td_one/agent/tools/analysis.py` (new)
- `src/td_one/agent/tools/regimes.py` (new)
- `src/td_one/agent/tools/evaluation.py` (new)
- `src/td_one/agent/tools/plotting.py` (new)
- `src/td_one/agent/tools/scenarios.py` (new)
- `tests/test_sandbox.py` (new)
- `tests/test_agent_tools.py` (new)
- `tests/test_react_v2.py` (new)

## Ready for
- **exp21**: Agent feature discovery (residuals → propose → CV-test loop)
- **exp22**: M2 v2 vs v1 comparison (code-executing ReAct vs 3-tool ReAct)
- **exp23**: Diagnostic-based model selection
- **exp24**: Full pipeline on event-augmented data
