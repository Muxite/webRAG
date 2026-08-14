#!/usr/bin/env python3
"""
Audit every environment variable the codebase reads.

Greps the `services/` and `agent/` trees for env-var access (`os.environ.get("X")`,
`os.getenv("X")`, `os.environ["X"]`) and prints each variable, its read sites, and a default if a
literal one is given at the call site. Keeps docs/CONFIGURATION.md honest and catches drift over
time.

Usage:
    python scripts/list_env_vars.py            # grouped table
    python scripts/list_env_vars.py --names    # bare sorted names (for diffing against docs)
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_DIRS = [os.path.join(ROOT, "services"), os.path.join(ROOT, "agent")]

# os.environ.get("X"[, "default"])  |  os.getenv("X"[, "default"])  |  os.environ["X"]
_PATTERNS = [
    re.compile(r'os\.environ\.get\(\s*["\']([A-Z][A-Z0-9_]*)["\']\s*(?:,\s*([^)]*))?\)'),
    re.compile(r'os\.getenv\(\s*["\']([A-Z][A-Z0-9_]*)["\']\s*(?:,\s*([^)]*))?\)'),
    re.compile(r'os\.environ\[\s*["\']([A-Z][A-Z0-9_]*)["\']\s*\]'),
]


def scan() -> dict[str, dict]:
    found: dict[str, dict] = defaultdict(lambda: {"files": set(), "default": ""})
    for scan_dir in SCAN_DIRS:
        for dirpath, _dirs, files in os.walk(scan_dir):
            if "/.venv" in dirpath or "/__pycache__" in dirpath or "/node_modules" in dirpath:
                continue
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    text = open(path, encoding="utf-8").read()
                except (OSError, UnicodeDecodeError):
                    continue
                rel = os.path.relpath(path, ROOT)
                for pat in _PATTERNS:
                    for m in pat.finditer(text):
                        name = m.group(1)
                        found[name]["files"].add(rel)
                        default = (m.group(2) if m.lastindex and m.lastindex >= 2 else "") or ""
                        if default and not found[name]["default"]:
                            found[name]["default"] = default.strip()
    return found


def main() -> None:
    found = scan()
    names = sorted(found)
    if "--names" in sys.argv:
        print("\n".join(names))
        return
    test_only = [n for n in names if n.startswith("IDEA_TEST_")]
    other = [n for n in names if not n.startswith("IDEA_TEST_")]
    print(f"{len(names)} env vars ({len(other)} runtime, {len(test_only)} IDEA_TEST_* benchmark-only)\n")
    for n in other:
        info = found[n]
        default = f"  (default {info['default']})" if info["default"] else ""
        print(f"  {n}{default}\n      {', '.join(sorted(info['files']))}")
    print(f"\n  IDEA_TEST_* ({len(test_only)}): " + ", ".join(test_only))


if __name__ == "__main__":
    main()
