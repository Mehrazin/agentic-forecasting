"""Tests for td_one/agent/retry.py: backoff, rate-limit niceness.

Ported from the AI2 harness (tests/test_llm_retry.py) so both repos
verify the same retry sophistication. No network; sleeps are stubbed.
"""

from __future__ import annotations

import pytest

from td_one.agent.retry import (
    RetryConfig,
    _backoff_delay,
    _is_rate_limited,
    _is_transient,
    _retry_after_s,
    with_retry,
)


class RateLimitError(Exception):
    """RateLimitError."""

    def __init__(self, retry_after=None):  # type: ignore[no-untyped-def]
        """Initialize."""
        self.headers = {"retry-after": str(retry_after)} if retry_after else {}


class ServerError(Exception):
    """ServerError."""

    status_code = 500


class BadRequestError(Exception):
    """Simulates a non-retryable 400 Bad Request error."""

    status_code = 400


def test_rate_limit_detected_by_name_and_status() -> None:
    """Verifies rate limit detected by name and status."""
    assert _is_rate_limited(RateLimitError())
    assert _is_rate_limited(BadRequestError()) is False


def test_transient_covers_5xx_and_rate_limit() -> None:
    """Transient covers 5xx and rate limit."""
    cfg = RetryConfig()
    assert _is_transient(ServerError(), cfg)
    assert _is_transient(RateLimitError(), cfg)
    assert not _is_transient(BadRequestError(), cfg)
    assert not _is_transient(ValueError("nope"), cfg)


def test_backoff_honors_retry_after() -> None:
    """Verifies backoff honors retry after."""
    cfg = RetryConfig(min_rate_limit_delay_s=5.0)
    d = _backoff_delay(0, RateLimitError(retry_after=30), cfg)
    assert 22.0 <= d <= 38.0  # 30s +/- 25% jitter


def test_backoff_floors_rate_limit_without_header() -> None:
    """Verifies backoff floors rate limit without header."""
    cfg = RetryConfig(min_rate_limit_delay_s=5.0, jitter_ratio=0.0)
    d = _backoff_delay(0, RateLimitError(), cfg)
    assert d >= 5.0


def test_backoff_grows_exponentially() -> None:
    """Transient (non-rate-limit) backoff doubles per attempt."""
    cfg = RetryConfig(base_delay_s=1.0, jitter_ratio=0.0, max_delay_s=1000.0)
    d0 = _backoff_delay(0, ServerError(), cfg)
    d1 = _backoff_delay(1, ServerError(), cfg)
    assert d0 == pytest.approx(1.0)
    assert d1 == pytest.approx(2.0)


def test_retry_succeeds_after_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient failure then success returns the value; sleeps stubbed."""
    monkeypatch.setattr("td_one.agent.retry.time.sleep", lambda s: None)
    calls = {"n": 0}

    @with_retry(RetryConfig(max_attempts=3, jitter_ratio=0.0))
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ServerError("boom")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_non_transient_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 raises on the first attempt with no sleep."""
    slept = {"n": 0}
    monkeypatch.setattr("td_one.agent.retry.time.sleep", lambda s: slept.update(n=slept["n"] + 1))

    @with_retry(RetryConfig(jitter_ratio=0.0))
    def bad() -> str:
        raise BadRequestError("bad request")

    with pytest.raises(BadRequestError):
        bad()
    assert slept["n"] == 0


def test_exhaustion_reraises_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """After max_attempts transient failures, the last error propagates."""
    monkeypatch.setattr("td_one.agent.retry.time.sleep", lambda s: None)
    calls = {"n": 0}

    @with_retry(RetryConfig(max_attempts=3, jitter_ratio=0.0))
    def always_fails() -> str:
        calls["n"] += 1
        raise ServerError(f"fail {calls['n']}")

    with pytest.raises(ServerError, match="fail 3"):
        always_fails()
    assert calls["n"] == 3


def test_urlerror_is_transient() -> None:
    """Urllib raises URLError (not ConnectionError) for DNS/connect failures."""
    import urllib.error

    cfg = RetryConfig()
    assert _is_transient(urllib.error.URLError("dns failure"), cfg)
    assert "URLError" in cfg.transient_names


def test_empty_content_error_is_transient() -> None:
    """GroqEmptyContentError retries (known transient for gpt-oss)."""
    from td_one.agent.groq_client import GroqEmptyContentError

    assert _is_transient(GroqEmptyContentError("empty"), RetryConfig())


def test_retry_after_s_reads_attribute() -> None:
    """_retry_after_s honors a retry_after attribute carried on the error."""
    err = RuntimeError("429")
    err.retry_after = 42.0  # carried by clients (e.g. groq_client)
    assert _retry_after_s(err) == 42.0


def test_retry_after_attribute_drives_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """A carried Retry-After sets the delay (not just the floor)."""
    delays: list[float] = []
    monkeypatch.setattr("td_one.agent.retry.time.sleep", delays.append)
    calls = {"n": 0}

    @with_retry(RetryConfig(max_attempts=2, jitter_ratio=0.0, min_rate_limit_delay_s=5.0))
    def limited() -> str:
        calls["n"] += 1
        err = RateLimitError()
        err.status_code = 429
        err.retry_after = 30.0
        raise err

    with pytest.raises(RateLimitError):
        limited()
    assert calls["n"] == 2
    assert delays == [30.0]  # honored exactly (jitter off), not the 5s floor
