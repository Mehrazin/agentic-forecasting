"""Retry with exponential backoff for LLM calls, nice to rate limits.

Ported from the AI2 harness (``infra/llm/retry.py``) so both repos share
the same sophistication. All provider ``complete()`` implementations go
through :func:`with_retry`. Transient failures (timeouts, 5xx, 429) are
retried with exponential backoff plus jitter. Rate-limit responses are
honored: a ``Retry-After`` header (seconds) sets the delay directly, and
a 429 always backs off at least ``min_rate_limit_delay_s`` even without
the header. Non-retryable errors (4xx other than 429, auth failures)
raise immediately.

Usage::

    from td_one.agent.retry import RetryConfig, with_retry

    _RETRY = RetryConfig(max_attempts=6, min_rate_limit_delay_s=60.0)

    class MyClient:
        @with_retry(_RETRY)
        def _do_complete(self, prompt: str) -> str:
            ...
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Tunable retry policy."""

    max_attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    min_rate_limit_delay_s: float = 5.0
    jitter_ratio: float = 0.25
    # Exception types treated as transient (checked by name so providers
    # don't need to import every SDK's error hierarchy).
    transient_names: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "Timeout",
                "TimeoutError",
                "ConnectionError",
                # urllib raises URLError (not ConnectionError) for
                # DNS/connect failures; without this a network blip fails
                # the step immediately.
                "URLError",
                "ServiceUnavailable",
                "InternalServerError",
                "RateLimitError",
                "APISConnectionError",
                "APITimeoutError",
                # td_one.agent.groq_client.GroqEmptyContentError: the model
                # returned no usable content (known transient for gpt-oss).
                "GroqEmptyContentError",
            }
        )
    )


def _is_rate_limited(exc: BaseException) -> bool:
    """Detect rate-limit errors across SDKs without importing them."""
    name = type(exc).__name__
    if "RateLimit" in name or "TooManyRequests" in name:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status == 429


def _is_transient(exc: BaseException, config: RetryConfig) -> bool:
    """Decide whether the exception merits a retry."""
    if _is_rate_limited(exc):
        return True
    name = type(exc).__name__
    if name in config.transient_names:
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return isinstance(status, int) and 500 <= status < 600


def _retry_after_s(exc: BaseException) -> float | None:
    """Extract Retry-After seconds from an exception, if present."""
    headers = getattr(exc, "headers", None)
    if isinstance(headers, dict):
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw is not None:
            try:
                return max(0.0, float(raw))
            except (TypeError, ValueError):
                return None
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(retry_after, (int, float)):
        return max(0.0, float(retry_after))
    return None


def _backoff_delay(attempt: int, exc: BaseException, config: RetryConfig) -> float:
    """Compute the delay before the next attempt.

    Rate limits honor Retry-After (or a floor); other transient errors
    use exponential backoff with jitter.
    """
    if _is_rate_limited(exc):
        delay = _retry_after_s(exc)
        if delay is None:
            delay = config.min_rate_limit_delay_s * (2**attempt)
        delay = max(delay, config.min_rate_limit_delay_s)
    else:
        delay = config.base_delay_s * (2**attempt)
    delay = min(delay, config.max_delay_s)
    jitter = delay * config.jitter_ratio
    return delay + random.uniform(-jitter, jitter)


def with_retry(
    config: RetryConfig | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate an LLM call site with retry + backoff.

    Args:
        config: Retry policy; defaults to :class:`RetryConfig`.

    """
    cfg = config or RetryConfig()

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> T:
            last: BaseException | None = None
            for attempt in range(cfg.max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    # Intentional: classify ALL failures as transient or
                    # permanent; non-transient re-raise immediately.
                    last = exc
                    if not _is_transient(exc, cfg):
                        raise
                    if attempt == cfg.max_attempts - 1:
                        logger.debug(
                            "retry %s: exhausted after %d attempts (%r)",
                            getattr(fn, "__name__", "call"),
                            cfg.max_attempts,
                            exc,
                        )
                        raise
                    delay = _backoff_delay(attempt, exc, cfg)
                    logger.debug(
                        "retry %s: attempt %d/%d failed (%r); sleeping %.1fs before next attempt",
                        getattr(fn, "__name__", "call"),
                        attempt + 1,
                        cfg.max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(max(0.0, delay))
            assert last is not None  # unreachable; keeps mypy honest
            raise last

        return wrapper

    return decorator
