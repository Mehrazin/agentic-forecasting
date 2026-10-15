"""Bootstrap all TSFM model weights for TD-One.

Downloads (pinned revisions):
  - amazon/chronos-2 (120M params, primary)
  - google/timesfm-2.5-200m-pytorch (200M params, secondary)

Writes:
  - td-one/models/weights/<model>/  (raw snapshots)
  - td-one/models/MANIFEST.json      (revisions, sizes, timestamps, SHA)

Usage:
  cd td-one
  uv run --with huggingface_hub python scripts/bootstrap_models.py

The script is idempotent: existing snapshots are verified, not re-downloaded.
Requires: HF_TOKEN env var (or ~/.cache/huggingface/token) for gated repos.

Environment note: if your shell exports no_proxy/NO_PROXY with IPv6 bracket
entries (e.g. "[::1]"), huggingface_hub raises "Invalid port". Sanitize first:
  export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

MODELS = [
    {
        "key": "chronos2",
        "repo_id": "amazon/chronos-2",
        "revision": "main",  # pinned at download time, recorded in manifest
        "allow_patterns": ["*.safetensors", "*.json", "*.txt"],
    },
    {
        "key": "timesfm25",
        "repo_id": "google/timesfm-2.5-200m-pytorch",
        "revision": "main",
        "allow_patterns": ["*.safetensors", "*.pt", "*.bin", "*.json", "*.txt"],
    },
]


def sha256_of_dir(path: Path) -> str:
    """Hash of sorted file listing + sizes (fast, no full read)."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(path))
            h.update(rel.encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]


def dir_size_mb(path: Path) -> float:
    """Total size in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1024 / 1024, 1)


def main() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.", file=sys.stderr)
        print(
            "Run: uv run --with huggingface_hub "
            "python scripts/bootstrap_models.py",
            file=sys.stderr,
        )
        return 1

    root = Path(__file__).resolve().parent.parent  # td-one/
    weights_dir = root / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "bootstrap_date": datetime.now(UTC).isoformat(),
        "models": {},
    }

    for spec in MODELS:
        key = spec["key"]
        repo_id = spec["repo_id"]
        dest = weights_dir / key

        print(f"\n{'='*60}")
        print(f"Downloading {repo_id} -> {dest}")
        print(f"{'='*60}")

        try:
            snapshot_download(
                repo_id=repo_id,
                revision=spec["revision"],
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                allow_patterns=spec["allow_patterns"],
                resume_download=True,
            )
        except Exception as e:  # noqa: BLE001 - record any download failure
            print(f"FAILED to download {repo_id}: {e}", file=sys.stderr)
            manifest["models"][key] = {
                "repo_id": repo_id,
                "status": "failed",
                "error": str(e),
            }
            continue

        # Resolve actual commit hash
        actual_revision = spec["revision"]
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.repo_info(repo_id=repo_id, revision=spec["revision"])
            actual_revision = info.sha
        except Exception:  # noqa: BLE001 - offline/no-token: keep requested rev
            print(
                f"WARNING: could not resolve SHA for {repo_id}; "
                'recording "UNRESOLVED" — manifest is NOT reproducible',
            )
            actual_revision = "UNRESOLVED (offline or no HF_TOKEN)"

        size_mb = dir_size_mb(dest)
        dir_hash = sha256_of_dir(dest)

        manifest["models"][key] = {
            "repo_id": repo_id,
            "revision_requested": spec["revision"],
            "revision_resolved": actual_revision,
            "local_path": str(dest.relative_to(root)),
            "size_mb": size_mb,
            "dir_hash": dir_hash,
            "status": "ok",
        }
        print(f"  OK: {size_mb} MB, hash={dir_hash}, rev={actual_revision[:12]}")

    manifest_path = root / "models" / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    ok = sum(1 for m in manifest["models"].values() if m.get("status") == "ok")
    print(f"  {ok}/{len(MODELS)} models downloaded successfully")
    for key, m in manifest["models"].items():
        status = m.get("status", "?")
        size = m.get("size_mb", "?")
        print(f"  {key}: {status} ({size} MB)")

    return 0 if ok == len(MODELS) else 1


if __name__ == "__main__":
    sys.exit(main())
