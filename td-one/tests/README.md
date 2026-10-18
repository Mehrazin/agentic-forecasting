# `tests/` — test suite

TDD is the rule; every bug fix gets a regression test. Test philosophy (from
the repo's agent instructions): test non-obvious logic, defensive contracts,
and error paths — not Pydantic construction or trivial behavior.

```
.venv/bin/pytest tests/ -x -q        # full suite
.venv/bin/pytest tests/test_events.py -q   # one file
```

`conftest.py` holds shared fixtures. Up: [`td-one/`](../README.md).
