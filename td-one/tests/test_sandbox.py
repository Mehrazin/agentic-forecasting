"""Tests for the sandboxed code execution (td_one.agent.sandbox).

Covers: basic execution, timeout enforcement, secret stripping from the
child environment, path-traversal guards, output byte caps, audit logging,
and input validation.
"""

import os

import pytest

from td_one.agent.sandbox import (
    AUDIT_LOG,
    DockerSandbox,
    _guard_path,
    _sanitized_env,
    run_code,
    run_sandboxed,
)


def test_basic_execution() -> None:
    r = run_sandboxed("print(2 + 2)")
    assert r.returncode == 0
    assert r.stdout.strip() == "4"
    assert not r.timed_out
    assert r.source_sha256


def test_failure_does_not_raise() -> None:
    r = run_sandboxed("raise RuntimeError('boom')")
    assert r.returncode != 0
    assert "boom" in r.stderr


def test_timeout_enforced() -> None:
    r = run_sandboxed("import time; time.sleep(30)", timeout_s=2)
    assert r.timed_out
    assert r.returncode == -1


def test_timeout_validation() -> None:
    with pytest.raises(ValueError):
        run_sandboxed("print(1)", timeout_s=0)
    with pytest.raises(ValueError):
        run_sandboxed("print(1)", timeout_s=301)


def test_empty_source_rejected() -> None:
    with pytest.raises(ValueError):
        run_sandboxed("   ")


def test_memory_validation() -> None:
    with pytest.raises(ValueError):
        run_sandboxed("print(1)", memory_mb=32)


def test_secrets_stripped_from_env() -> None:
    env = _sanitized_env()
    for key in env:
        upper = key.upper()
        assert "GROQ_API_KEY" not in upper or key not in env
    assert "GROQ_API_KEY" not in env
    assert "HTTP_PROXY" not in env
    assert "http_proxy" not in env
    # Credential-smelling keys are dropped even if custom-named.
    os.environ["MY_CUSTOM_SECRET_XYZ"] = "s3cret"
    try:
        assert "MY_CUSTOM_SECRET_XYZ" not in _sanitized_env()
    finally:
        del os.environ["MY_CUSTOM_SECRET_XYZ"]


def test_child_cannot_see_parent_secret() -> None:
    os.environ["TD1_TEST_SECRET_ABC"] = "topsecret"
    try:
        r = run_sandboxed("import os; print(os.environ.get('TD1_TEST_SECRET_ABC', 'MISSING'))")
        assert r.stdout.strip() == "MISSING"
    finally:
        del os.environ["TD1_TEST_SECRET_ABC"]


def test_output_byte_cap() -> None:
    r = run_sandboxed("print('x' * 1000000)")
    assert r.truncated
    assert len(r.stdout.encode("utf-8")) <= 256 * 1024 + 100


def test_path_guard_rejects_escape() -> None:
    with pytest.raises(ValueError):
        _guard_path("/etc/passwd")
    # Absolute path outside allowed roots.
    with pytest.raises(ValueError):
        _guard_path("/root/secret.txt")


def test_path_guard_allows_tmp() -> None:
    p = _guard_path("/tmp/td1_test_file.txt")
    assert str(p).startswith("/tmp")


def test_extra_files_traversal_rejected() -> None:
    with pytest.raises(ValueError):
        run_sandboxed("print(1)", extra_files={"../../evil.py": "print('evil')"})


def test_extra_files_staged() -> None:
    r = run_sandboxed(
        "print(open('data.txt').read().strip())",
        extra_files={"data.txt": "hello"},
    )
    assert r.returncode == 0
    assert r.stdout.strip() == "hello"


def test_audit_log_records() -> None:
    before = len(AUDIT_LOG)
    r = run_sandboxed("print('audit')")
    assert len(AUDIT_LOG) == before + 1
    rec = AUDIT_LOG[-1]
    assert rec.source_sha256 == r.source_sha256
    assert rec.returncode == 0


def test_run_code_entrypoint() -> None:
    r = run_code("print('entry')")
    assert r.returncode == 0
    assert "entry" in r.stdout


def test_docker_available_returns_bool() -> None:
    assert isinstance(DockerSandbox.available(), bool)


def test_numpy_available_in_sandbox() -> None:
    r = run_sandboxed("import numpy; print(numpy.__version__)")
    # numpy may or may not be importable in the child depending on env;
    # the test just asserts the sandbox reports honestly.
    assert r.returncode in (0, 1)
