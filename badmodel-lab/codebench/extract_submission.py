#!/usr/bin/env python3
"""Filter an agent's raw /work output into a gradeable submission.

Drops anything matching the task's private/test_manifest.json glob patterns before the
result ever reaches run_grade.sh — the agent may have written its own copy of a visible
test file (or tried to plant a fake one), but that copy must never be trusted. What
actually gets graded is always the canonical file from tasks/<id>/private/tests/.

SECURITY: this script runs on the HOST (not inside a container) — /work is a real host
bind mount (see run_agent_sandbox.sh), so any symlink the agent creates there points at a
REAL host path the instant it's written, independent of the container's own filesystem
view. shutil.copy2's default follow_symlinks=True would dereference it and copy the
TARGET's actual content into the submission tree, verified live: a one-line run_python
action (`os.symlink('/home/user/.ssh/id_rsa', 'leak')`) is enough to exfiltrate any file
readable by whoever runs this harness. Symlinks are refused outright, never followed —
a legitimate coding-benchmark submission has no legitimate reason to contain one.

Usage: extract_submission.py --raw-dir OUT/raw --manifest tasks/<id>/private/test_manifest.json --out OUT/submission
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
from pathlib import Path


def load_manifest(path: Path) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    patterns = data.get("test_file_globs", data if isinstance(data, list) else [])
    if not isinstance(patterns, list):
        raise ValueError(f"{path}: expected a list or {{'test_file_globs': [...]}}")
    return [str(p) for p in patterns]


def is_dropped(rel_path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pat) for pat in patterns)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if not args.raw_dir.is_dir():
        print(f"extract_submission: raw dir missing: {args.raw_dir}", file=sys.stderr)
        return 1

    patterns = load_manifest(args.manifest)
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    kept, dropped, rejected_symlinks = 0, 0, 0
    for src in sorted(args.raw_dir.rglob("*")):
        # Checked BEFORE is_dir(): is_dir()/is_file() both follow symlinks, so a symlinked
        # directory would otherwise get silently walked into (or, for a symlinked file,
        # copied straight through by shutil.copy2 below) instead of being caught here.
        if src.is_symlink():
            rejected_symlinks += 1
            continue
        if src.is_dir():
            continue
        rel = src.relative_to(args.raw_dir).as_posix()
        if is_dropped(rel, patterns):
            dropped += 1
            continue
        dst = args.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)
        kept += 1

    if rejected_symlinks:
        print(f"extract_submission: WARNING rejected {rejected_symlinks} symlink(s) in raw output "
              f"(never followed, never copied)", file=sys.stderr)
    print(f"extract_submission: kept={kept} dropped={dropped} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
