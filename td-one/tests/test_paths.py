"""Tests for td_one._paths: the shared repo-root helper.

No network; no fixtures.
"""

from td_one._paths import _repo_root


def test_repo_root_holds_pyproject() -> None:
    """The helper resolves to the td-one project root.

    Regression: four modules each hand-counted parents[N]; one used the
    wrong N and silently read from the wrong directory.
    """
    root = _repo_root()
    assert (root / "pyproject.toml").is_file(), root
    assert root.name == "td-one", root
