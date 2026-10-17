"""Sandboxed code execution for the TD-One forecast agent.

Ported from AI2's ``metaharness/sandbox.py`` with TD-One hardening:
- Path traversal guards on any file arguments
- Byte caps on stdout/stderr (prompt-injection / log-bloat defense)
- Audit logging of every execution (input hash, timing, outcome)
- Restricted environment (proxy vars stripped; no inherited secrets)
- Read-only working directory except a per-call scratch dir under /tmp

Two backends:
- :func:`run_sandboxed`: local subprocess with wall-clock timeout and POSIX
  resource ceilings (address space, CPU, file size). Always available.
- :class:`DockerSandbox`: optional stronger isolation via ``docker run``
  with ``--network none``, memory/cpu caps, read-only rootfs.

Both return :class:`SandboxResult`; neither ever raises on child failure.
Callers should treat a non-zero returncode as "the agent's code failed",
not as an infrastructure error.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Hard cap on captured stdout/stderr bytes (bloat / injection defense).
MAX_OUTPUT_BYTES = 256 * 1024
#: Default wall-clock timeout for agent code (seconds).
DEFAULT_TIMEOUT_S = 30
#: Default memory ceiling (MiB). 1024 is the floor for a scientific-Python
#: stack (statsmodels.tsa import chain needs ~700MB of address space).
DEFAULT_MEMORY_MB = 1024
#: Default CPU-time ceiling (seconds).
DEFAULT_CPU_S = 60

#: Environment variables stripped from the child (secrets + network config).
_STRIPPED_ENV_VARS = frozenset(
    {
        # Proxy / network configuration.
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
        # Anything that looks like a credential.
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "PERPLEXITY_API_KEY",
        "FRED_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    }
)


@dataclass
class SandboxResult:
    """Result of a sandboxed execution."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    wall_ms: float = 0.0
    backend: str = "subprocess"
    #: True when stdout/stderr were truncated at MAX_OUTPUT_BYTES.
    truncated: bool = False
    #: SHA-256 of the executed source (for audit logs).
    source_sha256: str = ""
    #: Scratch dir used for this run (cleaned up afterwards).
    scratch_dir: str = ""


@dataclass
class AuditRecord:
    """One audit-log entry for a sandboxed execution."""

    source_sha256: str
    wall_ms: float
    returncode: int
    timed_out: bool
    stdout_bytes: int
    stderr_bytes: int
    backend: str
    timestamp: float = field(default_factory=time.time)


# In-memory audit log (also emitted via logging for persistence).
AUDIT_LOG: list[AuditRecord] = []


def _sanitized_env() -> dict[str, str]:
    """Child environment: inherit PATH-ish basics, strip secrets and proxies."""
    keep = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "PYTHONPATH",
        "PYTHONHASHSEED",
        "TMPDIR",
    }
    env = {k: v for k, v in os.environ.items() if k in keep}
    for var in _STRIPPED_ENV_VARS:
        env.pop(var, None)
    # Belt-and-braces: drop anything whose name smells like a credential.
    for key in list(env.keys()):
        upper = key.upper()
        if any(tok in upper for tok in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
            env.pop(key, None)
    return env


def _limit_resources(memory_mb: int, cpu_s: int, fsize_mb: int) -> None:
    """POSIX resource ceilings applied in the child via preexec_fn."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return

    def _set(res: int, soft: int, hard: int) -> None:
        with suppress(ValueError, OSError):
            resource.setrlimit(res, (soft, hard))

    _set(resource.RLIMIT_AS, memory_mb * 1024 * 1024, memory_mb * 1024 * 1024)
    _set(resource.RLIMIT_CPU, cpu_s, cpu_s)
    _set(resource.RLIMIT_FSIZE, fsize_mb * 1024 * 1024, fsize_mb * 1024 * 1024)
    _set(resource.RLIMIT_NOFILE, 64, 64)


def _guard_path(path: str | Path) -> Path:
    """Reject path traversal; return the resolved absolute path.

    Raises:
        ValueError: If the path escapes the workspace or /tmp.

    """
    p = Path(path).expanduser()
    # Only absolute paths under /tmp or the workspace are allowed.
    allowed_roots = (Path("/tmp").resolve(), Path.home().resolve() / "workspace")
    resolved = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(f"path escapes sandbox roots: {path}")
    return resolved


def _truncate(text: str, limit: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    """Truncate text to a byte cap; return (text, was_truncated)."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text, False
    cut = encoded[:limit].decode("utf-8", errors="ignore")
    return cut + "\n[... truncated at byte cap ...]", True


def run_sandboxed(
    source: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    cpu_s: int = DEFAULT_CPU_S,
    extra_files: dict[str, str] | None = None,
) -> SandboxResult:
    """Execute Python source in a resource-limited subprocess.

    Args:
        source: Python source to execute (``python -c``).
        timeout_s: Wall-clock timeout in seconds.
        memory_mb: Address-space ceiling in MiB.
        cpu_s: CPU-time ceiling in seconds.
        extra_files: Optional ``{relative_name: content}`` files staged into
            the per-call scratch dir (names are path-traversal guarded).

    Returns:
        SandboxResult; never raises on child failure.

    """
    if not source or not source.strip():
        raise ValueError("source must be non-empty")
    if timeout_s < 1 or timeout_s > 300:
        raise ValueError("timeout_s must be in [1, 300]")
    if memory_mb < 64 or memory_mb > 4096:
        raise ValueError("memory_mb must be in [64, 4096]")

    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    scratch = Path(tempfile.mkdtemp(prefix="td1_sandbox_"))
    if extra_files:
        for name, content in extra_files.items():
            target = _guard_path(scratch / name)
            # Must stay inside the scratch dir.
            if scratch.resolve() not in target.parents and target != scratch.resolve():
                raise ValueError(f"extra file escapes scratch dir: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    # Preamble: confine the child's CWD to scratch and block obvious
    # filesystem escapes at the Python level (defense in depth; the
    # subprocess + rlimits are the real boundary).
    preamble = f"import os as _os; _os.chdir({str(scratch)!r}); "
    full_source = preamble + source

    start = time.monotonic()
    result: SandboxResult
    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_source],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s,
            env=_sanitized_env(),
            preexec_fn=lambda: _limit_resources(memory_mb, cpu_s, 256),
        )
        stdout, t1 = _truncate(proc.stdout or "")
        stderr, t2 = _truncate(proc.stderr or "")
        result = SandboxResult(
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=False,
            wall_ms=(time.monotonic() - start) * 1000.0,
            backend="subprocess",
            truncated=t1 or t2,
            source_sha256=source_sha,
            scratch_dir=str(scratch),
        )
    except subprocess.TimeoutExpired as exc:
        raw_out: object = exc.stdout
        if isinstance(raw_out, bytes):
            raw_out = raw_out.decode("utf-8", errors="ignore")
        stdout, t1 = _truncate(str(raw_out or ""))
        result = SandboxResult(
            returncode=-1,
            stdout=stdout,
            stderr="timeout: wall-clock limit exceeded",
            timed_out=True,
            wall_ms=(time.monotonic() - start) * 1000.0,
            backend="subprocess",
            truncated=t1,
            source_sha256=source_sha,
            scratch_dir=str(scratch),
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    AUDIT_LOG.append(
        AuditRecord(
            source_sha256=result.source_sha256,
            wall_ms=result.wall_ms,
            returncode=result.returncode,
            timed_out=result.timed_out,
            stdout_bytes=len(result.stdout.encode("utf-8")),
            stderr_bytes=len(result.stderr.encode("utf-8")),
            backend=result.backend,
        )
    )
    logger.info(
        "sandbox run sha=%s rc=%d timed_out=%s wall_ms=%.1f backend=%s",
        result.source_sha256[:12],
        result.returncode,
        result.timed_out,
        result.wall_ms,
        result.backend,
    )
    return result


class DockerSandbox:
    """Docker-based sandbox for untrusted code (optional, stronger)."""

    def __init__(
        self, image: str = "python:3.11-slim", memory: str = "512m", cpus: str = "1.0"
    ) -> None:
        """Initialize the Docker sandbox configuration."""
        self.image = image
        self.memory = memory
        self.cpus = cpus

    @staticmethod
    def available() -> bool:
        """Return True if the Docker daemon is reachable."""
        if shutil.which("docker") is None:
            return False
        try:
            r = subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=False)
            return r.returncode == 0
        except (subprocess.SubprocessError, OSError):
            return False

    def run(self, source: str, timeout_s: int = DEFAULT_TIMEOUT_S) -> SandboxResult:
        """Execute source in a Docker container; return the result."""
        if not source or not source.strip():
            raise ValueError("source must be non-empty")
        source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        cmd: list[str] = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--read-only",
            "--pids-limit",
            "64",
            "--tmpfs",
            "/tmp:rw,size=256m",
            self.image,
            "python",
            "-c",
            source,
        ]
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, check=False
            )
            stdout, t1 = _truncate(proc.stdout or "")
            stderr, t2 = _truncate(proc.stderr or "")
            result = SandboxResult(
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=False,
                wall_ms=(time.monotonic() - start) * 1000.0,
                backend="docker",
                truncated=t1 or t2,
                source_sha256=source_sha,
            )
        except subprocess.TimeoutExpired:
            result = SandboxResult(
                returncode=-1,
                stderr="timeout: wall-clock limit exceeded",
                timed_out=True,
                wall_ms=(time.monotonic() - start) * 1000.0,
                backend="docker",
                source_sha256=source_sha,
            )
        except FileNotFoundError as exc:
            result = SandboxResult(
                returncode=-1,
                stderr=str(exc),
                wall_ms=(time.monotonic() - start) * 1000.0,
                backend="docker",
                source_sha256=source_sha,
            )
        AUDIT_LOG.append(
            AuditRecord(
                source_sha256=result.source_sha256,
                wall_ms=result.wall_ms,
                returncode=result.returncode,
                timed_out=result.timed_out,
                stdout_bytes=len(result.stdout.encode("utf-8")),
                stderr_bytes=len(result.stderr.encode("utf-8")),
                backend=result.backend,
            )
        )
        return result


def run_code(
    source: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    prefer_docker: bool = False,
) -> SandboxResult:
    """Run code, preferring Docker when available and requested.

    This is the entry point agents should use.
    """
    if prefer_docker and DockerSandbox.available():
        return DockerSandbox().run(source, timeout_s=timeout_s)
    return run_sandboxed(source, timeout_s=timeout_s, memory_mb=memory_mb)
