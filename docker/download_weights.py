#!/usr/bin/env python3
"""Download AuK model weights into ckpts/ on first container start.

Downloads (all three):
  • tencent/AuK              → ckpts/AuK/
  • tencent/AuK-Flash        → ckpts/AuK-Flash/     (optional, skip on failure)
  • Qwen/Qwen2.5-Omni-3B     → ckpts/Qwen2.5-Omni-3B/

Set SKIP_AUK_FLASH=1 to skip AuK-Flash, or AUTO_DOWNLOAD_WEIGHTS=false
to disable auto-download entirely (handled by entrypoint.sh).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


CKPTS_DIR = Path("/app/ckpts")

MODELS: list[dict[str, object]] = [
    {
        "repo_id": "tencent/AuK",
        "local_dir": str(CKPTS_DIR / "AuK"),
        "required": True,
        "files": ["auk_base.safetensors", "config.yaml", "vae.safetensors"],
        "glob": None,
    },
    {
        "repo_id": "tencent/AuK-Flash",
        "local_dir": str(CKPTS_DIR / "AuK-Flash"),
        "required": os.environ.get("SKIP_AUK_FLASH", "0") != "1",
        "files": ["auk_flash.safetensors", "config.yaml", "vae.safetensors"],
        "glob": None,
    },
    {
        "repo_id": "Qwen/Qwen2.5-Omni-3B",
        "local_dir": str(CKPTS_DIR / "Qwen2.5-Omni-3B"),
        "required": True,
        "files": ["config.json"],
        "glob": "*.safetensors",
    },
]


def _is_complete(dest: Path, files: list[str], glob: str | None) -> bool:
    """True only if every required file exists non-empty (plus a weight glob match)."""
    for name in files:
        p = dest / name
        if not (p.is_file() and p.stat().st_size > 0):
            return False
    if glob:
        if not any(p.is_file() and p.stat().st_size > 0 for p in dest.glob(glob)):
            return False
    return True


def download_model(repo_id: str, local_dir: str, files: list[str], glob: str | None) -> bool:
    """Download a single repo, return True on success (resumes partial downloads)."""
    dest = Path(local_dir)
    if _is_complete(dest, files, glob):
        print(f"[download] {repo_id} already present at {dest} – skipping")
        return True
    if dest.exists() and any(dest.iterdir()):
        print(f"[download] {repo_id} incomplete at {dest} – resuming …", flush=True)

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()

        print(f"[download] Downloading {repo_id} → {dest} …", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
        )
        print(f"[download] {repo_id} download complete.")
        return True
    except Exception as exc:
        print(f"[download] WARNING: failed to download {repo_id}: {exc}", file=sys.stderr)
        return False


def main() -> None:
    any_failed = False
    for model in MODELS:
        repo_id = str(model["repo_id"])
        local_dir = str(model["local_dir"])
        required = bool(model["required"])
        files_raw = model["files"]
        files = [str(f) for f in files_raw] if isinstance(files_raw, list) else []
        glob = model["glob"] if isinstance(model["glob"], str) else None

        ok = download_model(repo_id, local_dir, files, glob)
        if not ok and required:
            print(f"[download] ERROR: required model {repo_id} could not be downloaded.", file=sys.stderr)
            any_failed = True

    if any_failed:
        sys.exit(1)
    print("[download] All models ready.")


if __name__ == "__main__":
    main()
