"""Tests for the shared Groq chat client (no network; budget logic only)."""

from collections.abc import Callable
from email.message import Message

import pytest

from td_one.agent.groq_client import (
    GROQ_MODEL_DEFAULT,
    GroqChatClient,
    GroqEmptyContentError,
)


def test_default_model_is_gpt_oss_120b() -> None:
    assert GROQ_MODEL_DEFAULT == "openai/gpt-oss-120b"
    assert GroqChatClient().model == GROQ_MODEL_DEFAULT


def test_call_budget_must_be_positive() -> None:
    with pytest.raises(ValueError, match="call_budget"):
        GroqChatClient(call_budget=0)


def test_budget_exhaustion_raises_before_any_network() -> None:
    """The (budget+1)-th call raises without touching the network."""
    client = GroqChatClient(call_budget=2)
    client.calls = 2  # simulate two prior calls (no network in tests)
    with pytest.raises(RuntimeError, match="budget.*exhausted"):
        client.complete("hello")


def test_calls_counter_starts_at_zero() -> None:
    assert GroqChatClient(call_budget=5).calls == 0


# --- Fake HTTP layer (no network) -------------------------------------------


class _FakeSleeper:
    """Stand-in for the time module: records sleeps, never really waits."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def sleep(self, s: float) -> None:
        self.delays.append(s)


def _install_fake_http(
    monkeypatch: pytest.MonkeyPatch,
    behavior: Callable[[int], str | BaseException],
) -> dict[str, int]:
    """Fake the Groq HTTP layer.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        behavior: maps 1-based request number -> reply content string,
            or an exception to raise as the transport error.

    Returns:
        Mutable request counter (``state["n"]``).

    """
    import sys
    import types
    import urllib.request

    state = {"n": 0}

    def read_json_response(resp):
        state["n"] += 1
        outcome = behavior(state["n"])
        if isinstance(outcome, BaseException):
            raise outcome
        return {"choices": [{"message": {"content": outcome}}]}

    fake_dc = types.ModuleType("dynamic_credentials")
    fake_dc.add_surrogate_to_request = lambda req, *a, **k: None
    fake_dc.read_json_response = read_json_response
    monkeypatch.setitem(sys.modules, "dynamic_credentials", fake_dc)

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req, timeout=None):
            return _Resp()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **k: _Opener())
    return state


def _http_error(code: int, retry_after: str | None = None) -> BaseException:
    """Build a urllib HTTPError with optional Retry-After header."""
    import urllib.error

    hdrs = Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://api.groq.com/", code, "err", hdrs, None)


def _stub_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> _FakeSleeper:
    """Replace the retry module's sleep with a recorder (fast, deterministic)."""
    import td_one.agent.retry as retry_mod

    sleeper = _FakeSleeper()
    monkeypatch.setattr(retry_mod, "time", sleeper)
    return sleeper


def test_empty_content_is_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gpt-oss empty replies are transient: retried, then succeed."""
    sleeper = _stub_retry_sleep(monkeypatch)
    state = _install_fake_http(monkeypatch, lambda n: "" if n <= 2 else "done")
    client = GroqChatClient(call_budget=5)
    assert client._do_complete("hello", None, False, 5.0) == "done"
    assert state["n"] == 3
    assert len(sleeper.delays) == 2  # two backoffs before the success


def test_empty_content_exhaustion_reraises_dedicated_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent empty content surfaces GroqEmptyContentError (retryable)."""
    sleeper = _stub_retry_sleep(monkeypatch)
    _install_fake_http(monkeypatch, lambda n: "")
    client = GroqChatClient(call_budget=5)
    with pytest.raises(GroqEmptyContentError, match="empty content"):
        client._do_complete("hello", None, False, 5.0)
    assert len(sleeper.delays) == 5  # all 6 attempts exhausted


def test_retry_after_header_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server's Retry-After beats the 60s floor when larger."""
    sleeper = _stub_retry_sleep(monkeypatch)
    _install_fake_http(monkeypatch, lambda n: _http_error(429, retry_after="90"))
    client = GroqChatClient(call_budget=5)
    with pytest.raises(RuntimeError, match="429"):
        client._do_complete("hello", None, False, 5.0)
    assert len(sleeper.delays) == 5  # 6 attempts -> 5 backoffs
    for d in sleeper.delays:
        assert 67.5 <= d <= 112.5, d  # 90s +/- 25% jitter


def test_throttle_spaces_calls_process_wide(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 2.5s throttle spaces calls; the lock is held while waiting."""
    from td_one.agent import groq_client as gc

    sleeps: list[float] = []
    now = [1000.0]

    class _Clock:
        def monotonic(self) -> float:
            return now[0]

        def sleep(self, s: float) -> None:
            # The throttle must hold its process-wide lock while waiting.
            assert gc._throttle_lock.locked()
            sleeps.append(s)

    monkeypatch.setattr(gc, "time", _Clock())
    monkeypatch.setattr(gc, "_throttle_last", [1000.0])

    gc._throttle()  # no time elapsed -> full 2.5s wait
    assert sleeps == [2.5]
    now[0] = 1001.0  # 1s later -> 1.5s remains
    gc._throttle()
    assert sleeps == [2.5, 1.5]
    now[0] = 1004.0  # 3s later -> no wait
    gc._throttle()
    assert sleeps == [2.5, 1.5]
