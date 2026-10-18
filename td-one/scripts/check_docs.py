"""Check td-one docs hygiene: no orphaned files, no missing READMEs.

- Every .md must be a README/index itself or linked from another .md.
- Every .png in docs/plots must be referenced from a .md.
- Every directory must have a README.md or be allowlisted.

Run: .venv/bin/python scripts/check_docs.py   (from td-one/)
Exit 0 = clean, 1 = problems listed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWLIST_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".hypothesis", ".agent_state",
    # weight blobs and raw data shards carry their parent's README
    "models/weights/chronos2", "models/weights/timesfm25",
    "data/news_dist", "data/scale_up", "data/schemas", "data/synthetic",
    "data/freddie/raw", "data/freddie/staging",
}

LINK_RE = re.compile(r"\]\(([^)#\s]+\.mdc?)\)|]\(([^)#\s]+\.png)\)")


def main() -> int:
    problems: list[str] = []

    mds = [p for p in ROOT.rglob("*.md") if ".venv" not in p.parts]
    # map of every link target -> set of sources
    linked_md: set[str] = set()
    linked_png: set[str] = set()
    for src in mds:
        try:
            text = src.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in LINK_RE.finditer(text):
            target = m.group(1) or m.group(2)
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (src.parent / target).resolve()
            try:
                rel = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                problems.append(f"LEAK: {src.relative_to(ROOT)} links outside td-one: {target}")
                continue
            if target.endswith(".png"):
                linked_png.add(rel)
            else:
                linked_md.add(rel)

    # 1. orphaned markdown
    index_names = {"README.md", "CONTRIBUTING.md"}
    for p in mds:
        rel = p.relative_to(ROOT).as_posix()
        if p.name in index_names or rel in linked_md:
            continue
        problems.append(f"ORPHAN MD: {rel} (not linked from any doc)")

    # 2. unreferenced plots
    for p in (ROOT / "docs" / "plots").glob("*.png"):
        rel = p.relative_to(ROOT).as_posix()
        if rel not in linked_png:
            problems.append(f"UNREFERENCED PNG: {rel}")

    # 3. directories without README
    for d in sorted(p for p in ROOT.rglob("*") if p.is_dir() and ".venv" not in p.parts):
        rel = d.relative_to(ROOT).as_posix()
        if rel in ALLOWLIST_DIRS or any(
            rel == a or rel.startswith(a + "/") for a in ALLOWLIST_DIRS
        ):
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in d.parts):
            continue
        if not (d / "README.md").exists():
            problems.append(f"NO README: {rel}/")

    for prob in problems:
        print(prob)
    print(f"\n{len(problems)} problem(s) across {len(mds)} markdown files.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
