"""Shell tool — a READ-ONLY, allow-listed command DSL.

The model never authors a command string. It picks a narrow intent (COUNT_LINES,
DISK_USAGE, FIND_FILE, …) and fills typed slots; code builds the exact ``argv`` from a
fixed allow-list (no ``shell=True``, no pipes, no metacharacter expansion), runs it with
``cwd`` = the workdir under a timeout, and returns a SUMMARIZED result. The container is
the real boundary; this allow-list + path confinement is defence in depth.

Only read-only executables are permitted; writes/network/deletes are simply not in the
allow-list, so they cannot be constructed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Tool, ToolContext, ToolResult, confine

_ALLOWED = {"wc", "grep", "du", "find", "head"}   # read-only only
_TIMEOUT = 10


class ShellTool:
    name = "shell"

    def _file_under(self, ctx: ToolContext, ref: Optional[str]) -> Optional[Path]:
        if not ref:
            return None
        try:
            target = Path(ref).resolve()
            base = ctx.workdir.resolve()
        except Exception:
            return None
        return target if (target == base or base in target.parents) and target.is_file() else None

    def execute(self, request: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        op = request.get("op")
        wd = ctx.workdir
        argv: Optional[List[str]] = None

        if op == "count_lines":
            target = self._file_under(ctx, request.get("file"))
            if target is None:
                return ToolResult(False, "count_lines: file not under workdir", error="path")
            pattern = request.get("pattern")
            argv = ["grep", "-c", "--", str(pattern), str(target)] if pattern \
                else ["wc", "-l", str(target)]
        elif op == "word_count":
            target = self._file_under(ctx, request.get("file"))
            if target is None:
                return ToolResult(False, "word_count: file not under workdir", error="path")
            argv = ["wc", "-w", str(target)]
        elif op == "head":
            target = self._file_under(ctx, request.get("file"))
            if target is None:
                return ToolResult(False, "head: file not under workdir", error="path")
            n = max(1, min(200, int(request.get("n", 10) or 10)))
            argv = ["head", "-n", str(n), str(target)]
        elif op == "disk_usage":
            d = confine(wd, request.get("dir", ".") or ".")
            if d is None or not d.exists():
                return ToolResult(False, "disk_usage: dir not under workdir", error="path")
            argv = ["du", "-sh", str(d)]
        elif op == "find":
            name = str(request.get("name", "")).strip()
            if not name:
                return ToolResult(False, "find: no name", error="missing")
            argv = ["find", str(wd.resolve()), "-name", name, "-type", "f"]
        else:
            return ToolResult(False, f"unknown shell op {op!r}", error="unknown_op")

        if argv[0] not in _ALLOWED:
            return ToolResult(False, f"command {argv[0]!r} not allow-listed", error="not_allowed")
        try:
            proc = subprocess.run(argv, cwd=str(wd), capture_output=True, text=True, timeout=_TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(False, f"shell error: {exc}", error="exec_error")

        out = (proc.stdout or "").strip()
        # grep exit 1 == "no matches", a valid result not an error
        ok = proc.returncode == 0 or (argv[0] == "grep" and proc.returncode == 1)
        summary = out[:200].replace("\n", " / ") or "(no output)"
        return ToolResult(ok, f"{op}: {summary}", data={"stdout": out, "rc": proc.returncode})
