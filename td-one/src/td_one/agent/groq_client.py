"""Groq chat client via the vault surrogate (the API key never enters this process).

Retry behavior mirrors the AI2 harness (``infra/llm/groq_vault_provider.py``):
patient retry for the Groq free tier (30 RPM) — on 429, wait a full minute,
because Groq's window is per-minute and short exponential backoff just burns
attempts. A module-level throttle spaces calls 2.5s apart process-wide.

Groq sits behind Cloudflare, so every request carries a browser User-Agent
(a bare client gets a 403/1010 block that looks like a key rejection).
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import threading
import time
import urllib.error
import urllib.request

from td_one.agent.retry import RetryConfig, with_retry

logger = logging.getLogger(__name__)

#: Verified default model (best on complex agentic tasks in prior testing).
GROQ_MODEL_DEFAULT = "openai/gpt-oss-120b"

#: Valid reasoning-effort levels for GPT-OSS models on Groq
#: (per console.groq.com/docs/reasoning). Higher effort = deeper reasoning
#: but more tokens consumed (the "reasoning tax" against the TPM budget).
REASONING_EFFORT_LEVELS = ("low", "medium", "high")

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

#: Location of the vault-surrogate helper (Muse VM connector layout).
#: Hoisted here (once, de-duplicated) instead of on every request so the
#: import below never accumulates duplicate sys.path entries.
_CONNECTOR_BIN = "/opt/muse/skills/custom-connector-setup/bin"
if _CONNECTOR_BIN not in sys.path:
    sys.path.insert(0, _CONNECTOR_BIN)


class GroqEmptyContentError(RuntimeError):
    """The model returned no usable content.

    Known transient failure mode of gpt-oss (reasoning tax vs the
    ``max_tokens`` cap). Classified transient in ``td_one.agent.retry`` so
    empty replies are retried instead of burning a ReAct step.
    """


#: Patient retry for the Groq free tier (mirrors AI2's _VAULT_RETRY).
_GROQ_RETRY = RetryConfig(
    max_attempts=6,
    base_delay_s=2.0,
    max_delay_s=120.0,
    min_rate_limit_delay_s=60.0,
)

#: Client-side throttle: space calls at least 2.5s apart process-wide
#: (module-level so all client instances share it).
_throttle_lock = threading.Lock()
_throttle_last: list[float] = [0.0]
_MIN_INTERVAL_S = 2.5


def _retry_after_s(exc: urllib.error.HTTPError) -> float | None:
    """Parse Retry-After seconds from an HTTPError's headers, if present."""
    raw = exc.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects on authenticated requests."""

    def redirect_request(self, *args):  # type: ignore[no-untyped-def]
        """Refuse all redirects."""
        raise RuntimeError("redirect refused, verify the endpoint")


def _throttle() -> None:
    with _throttle_lock:
        now = time.monotonic()
        wait = _MIN_INTERVAL_S - (now - _throttle_last[0])
        if wait > 0:
            time.sleep(wait)
        _throttle_last[0] = time.monotonic()


class GroqChatClient:
    """Minimal Groq chat-completion client.

    Implements the ``complete(prompt) -> str`` protocol consumed by
    :class:`td_one.agent.react.ReActAgent`, plus the ``(prompt, system,
    json_mode, timeout_s)`` superset used by the M3 delta estimator.
    Retries transient failures (429/5xx/timeouts) with the patient
    free-tier policy; the per-run call budget still bounds total spend.
    """

    def __init__(
        self,
        model: str = GROQ_MODEL_DEFAULT,
        call_budget: int = 10,
        max_tokens: int = 4000,
        reasoning_effort: str = "medium",
    ) -> None:
        """Initialize.

        Args:
            model: Groq model id.
            call_budget: Maximum ``complete`` calls before raising; bounds
                spend per experiment run.
            max_tokens: Per-call token cap. Kept under the ~8000 TPM
                free-tier ceiling (8192 provoked HTTP 413 in the wild).
            reasoning_effort: GPT-OSS reasoning depth — "low", "medium",
                or "high". Higher effort improves complex-task quality but
                consumes more of the TPM budget (the "reasoning tax").

        """
        if call_budget < 1:
            raise ValueError("call_budget must be >= 1")
        if reasoning_effort not in REASONING_EFFORT_LEVELS:
            raise ValueError(
                f"reasoning_effort must be one of {REASONING_EFFORT_LEVELS}, "
                f"got {reasoning_effort!r}"
            )
        self.model = model
        self.call_budget = call_budget
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.calls = 0

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        timeout_s: float = 120.0,
    ) -> str:
        """Send one chat completion; return the assistant's text content.

        Wall-clock note: the budget bounds Groq *spend*, not time. One
        logical call may attempt up to 6 HTTP tries with backoff (5 x up
        to 120s on repeated 429s is ~10 minutes of sleeping alone, plus
        per-attempt timeouts), so a 20-call run can theoretically run for
        hours under sustained rate limiting.
        """
        if self.calls >= self.call_budget:
            raise RuntimeError(f"Groq call budget ({self.call_budget}) exhausted")
        self.calls += 1
        _throttle()
        return self._do_complete(prompt, system, json_mode, timeout_s)

    @with_retry(_GROQ_RETRY)
    def _do_complete(
        self,
        prompt: str,
        system: str | None,
        json_mode: bool,
        timeout_s: float,
    ) -> str:
        """Single completion attempt (retry wrapper handles transients)."""
        body: dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system or "You are a mortgage risk analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        raw = json.dumps(body).encode()

        from dynamic_credentials import (  # type: ignore[import-not-found]
            add_surrogate_to_request,
            read_json_response,
        )

        req = urllib.request.Request(_GROQ_CHAT_URL, data=raw)
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", _BROWSER_UA)
        req.add_header("Accept", "application/json")
        add_surrogate_to_request(req, "custom.groq", allowed_hosts=["api.groq.com"])
        try:
            with urllib.request.build_opener(NoRedirect()).open(req, timeout=timeout_s) as resp:
                data = read_json_response(resp)
        except urllib.error.HTTPError as exc:
            # Preserve the status for the retry wrapper (_is_rate_limited
            # checks status_code); include Groq's body for 4xx diagnostics.
            # Carry Retry-After through as an attribute so _retry_after_s
            # honors the server's requested delay instead of only the floor.
            detail = ""
            with contextlib.suppress(Exception):
                detail = exc.read().decode()[:500]
            err = RuntimeError(f"Groq request failed: HTTP Error {exc.code}: {detail}")
            err.status_code = exc.code  # type: ignore[attr-defined]
            retry_after = _retry_after_s(exc)
            if retry_after is not None:
                err.retry_after = retry_after  # type: ignore[attr-defined]
            raise err from exc
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Groq returned no choices")
        content = choices[0]["message"]["content"]
        if not content:
            raise GroqEmptyContentError("Groq returned empty content")
        return str(content)
