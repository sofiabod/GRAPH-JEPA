"""mirror the modal `tgjepa-results` volume to local `results/_modal_volume/`.

ensures every checkpoint, eval json, and summary produced on modal is saved
locally so nothing is lost if the volume is purged or the modal account is
unreachable.

usage:
    python scripts/pull_modal_results.py
    python scripts/pull_modal_results.py --paths /baci_gravity /tgbn_trade
    python scripts/pull_modal_results.py --dest results/_modal_volume

what gets pulled (default, top-level dirs on the volume):
    /<dataset>/seed{N}/checkpoint.pt
    /<dataset>/seed{N}/checkpoint_best.pt
    /sequential-ablation/seed{N}/checkpoint.pt
    /eval/<dataset>/seed{N}/eval_summary.json
    /eval2/seed{N}/eval2.json
    (plus any other top-level dirs on the volume)

uses the modal cli (`modal volume get`) which is the most reliable cross-version
api. requires `modal` installed and authenticated locally.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


VOLUME_NAME = "tgjepa-results"


def list_volume_top_level() -> list[str]:
    """run `modal volume ls` and return top-level dir names on the volume."""
    proc = subprocess.run(
        ["modal", "volume", "ls", VOLUME_NAME],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"modal volume ls failed:\n{proc.stderr}", file=sys.stderr)
        return []
    # output has a header line and a list; keep non-empty lines that look like names
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    # strip out anything that looks like a header / table border
    names = []
    for ln in lines:
        # modal cli emits names as leaf strings in its newer ls; older versions used tables.
        # be liberal: anything without spaces or shell metacharacters is a candidate.
        if any(ch in ln for ch in ("│", "─", "┌", "└", "├", " " * 4)):
            continue
        if ln.lower() in ("filename", "name", "type"):
            continue
        names.append(ln)
    return names


def pull_path(remote: str, local_dest: Path) -> bool:
    """run `modal volume get` to download a remote path into local_dest.

    returns True on success.
    """
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    # --force overwrites existing files; without it, modal errors with
    # "Is a directory" when the destination dir already exists from a prior pull.
    cmd = ["modal", "volume", "get", "--force", VOLUME_NAME, remote, str(local_dest)]
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # modal cli sometimes returns non-zero even on partial success; print but continue
        print(f"  warn ({proc.returncode}): {proc.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="results/_modal_volume",
                        help="local directory to mirror into")
    parser.add_argument("--paths", nargs="+", default=None,
                        help="specific remote paths to pull (default: all top-level dirs)")
    parser.add_argument("--clean", action="store_true",
                        help="remove the local mirror dir before pulling")
    args = parser.parse_args()

    dest = Path(args.dest)
    if args.clean and dest.exists():
        print(f"cleaning {dest} ...")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.paths:
        paths = [p if p.startswith("/") else f"/{p}" for p in args.paths]
    else:
        names = list_volume_top_level()
        if not names:
            print("could not list volume top-level dirs; pass --paths explicitly", file=sys.stderr)
            sys.exit(1)
        paths = [f"/{n}" for n in names]

    print(f"mirroring {VOLUME_NAME} -> {dest}")
    print(f"pulling {len(paths)} path(s): {paths}")

    failures = []
    for remote in paths:
        local = dest / remote.lstrip("/")
        ok = pull_path(remote, local)
        if not ok:
            failures.append(remote)

    print()
    print(f"done. mirrored to {dest}")
    if failures:
        print(f"  {len(failures)} path(s) had warnings: {failures}")


if __name__ == "__main__":
    main()
