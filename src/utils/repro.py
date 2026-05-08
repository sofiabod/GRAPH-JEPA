"""reproducibility metadata capture.

every result json should be tied to:
- the commit it was produced from (git_sha)
- the dataset file content (dataset_sha256)
- the config file content (config_sha256)
- library versions (torch, numpy, python)

this lets a future reader replay any number in the paper by checking out
git_sha, ensuring the dataset hash matches, and rerunning the same script.

inside modal containers there is no .git tree, so commit_sha is read from
an env var GIT_SHA that the local entrypoint writes when launching.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _read_git_sha() -> Optional[str]:
    """try git rev-parse first; fall back to GIT_SHA env (set by local entrypoint)."""
    env_sha = os.environ.get("GIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, OSError):
        pass
    return None


def _read_git_dirty() -> Optional[bool]:
    """returns True if working tree has uncommitted changes, False if clean,
    None if git unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=False,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (FileNotFoundError, OSError):
        pass
    return None


def file_sha256(path: str | Path) -> Optional[str]:
    """sha256 of a file (None if not readable)."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lib_versions() -> dict:
    """capture versions of the core libraries that affect numerical results."""
    versions: dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in ("torch", "numpy", "scipy", "torch_geometric", "omegaconf"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def capture_metadata(*, dataset_paths: list[str | Path] | None = None,
                     config_path: str | Path | None = None,
                     extra: dict | None = None) -> dict:
    """capture all reproducibility metadata into a single dict.

    pass dataset_paths for the graphs.pt + meta.json pair, and config_path
    for the cfg yaml. result is a dict suitable for embedding in any
    result json under a `_repro` key.
    """
    md: dict = {
        "git_sha": _read_git_sha(),
        "git_dirty": _read_git_dirty(),
        "lib_versions": lib_versions(),
        "cuda": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic": _torch_deterministic_state(),
        },
        "argv": list(sys.argv),
    }
    if dataset_paths:
        md["dataset_sha256"] = {
            str(p): file_sha256(p) for p in dataset_paths
        }
    if config_path is not None:
        md["config_path"] = str(config_path)
        md["config_sha256"] = file_sha256(config_path)
    if extra:
        md.update(extra)
    return md


def _torch_deterministic_state() -> Optional[bool]:
    try:
        import torch
        return bool(torch.are_deterministic_algorithms_enabled())
    except Exception:
        return None
