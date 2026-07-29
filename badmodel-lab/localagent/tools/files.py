"""File tool — read/write/move/list confined to the sandbox workdir.

Each file *action* (READ_FILE, WRITE_FILE, MOVE_FILE, LIST_DIR) maps to this one tool
with ``op`` injected by the action's build_request, so the model only ever picks a
narrow intent, never ``file(op=...)``. All paths are confined via ``confine()`` even
though the container is the real boundary — defence in depth. Output is summarized.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .base import Tool, ToolContext, ToolResult, confine


class FileTool:
    name = "file"

    def execute(self, request: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        op = request.get("op")
        if op == "list":
            return self._list(request, ctx)
        if op == "read":
            return self._read(request, ctx)
        if op == "write":
            return self._write(request, ctx)
        if op == "move":
            return self._move(request, ctx)
        return ToolResult(False, f"unknown file op {op!r}", error="unknown_op")

    def _list(self, request: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        rel = request.get("path", ".") or "."
        target = confine(ctx.workdir, rel)
        if target is None or not target.exists():
            return ToolResult(False, f"list: {rel} not found or outside workdir", error="not_found")
        if not target.is_dir():
            return ToolResult(False, f"list: {rel} is a file, not a directory (try READ_FILE)",
                              error="not_a_dir")
        entries = sorted(p for p in target.iterdir())
        new_entities = []
        for p in entries:
            kind = "dir" if p.is_dir() else "file"
            new_entities.append((kind, p.name, str(p)))
        names = ", ".join(p.name for p in entries) or "(empty)"
        return ToolResult(True, f"listed {rel}: {len(entries)} items ({names})",
                          data={"count": len(entries)}, new_entities=new_entities)

    def _read(self, request: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ref = request.get("file")  # resolved abs path from an entity id
        if not ref:
            return ToolResult(False, "read: no file", error="missing")
        try:
            target = Path(ref).resolve()
            base = ctx.workdir.resolve()
        except Exception:
            return ToolResult(False, "read: bad path", error="bad_path")
        if not (target == base or base in target.parents):
            return ToolResult(False, "read: outside workdir", error="path_escape")
        if not target.is_file():
            return ToolResult(False, "read: not a file", error="not_found")
        text = target.read_text(errors="replace")
        head = text[:400].replace("\n", " ")
        return ToolResult(True, f"read {target.name}: {len(text)} chars — “{head[:120]}…”",
                          data={"chars": len(text), "text": text})

    def _write(self, request: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        rel = request.get("path")
        content = request.get("content", "")
        if not rel:
            return ToolResult(False, "write: no path", error="missing_path")
        target = confine(ctx.workdir, rel)
        if target is None:
            return ToolResult(False, f"write: {rel} escapes workdir", error="path_escape")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content))
        return ToolResult(True, f"wrote {rel} ({len(str(content))} chars)",
                          data={"path": str(target)},
                          new_artifacts=[("file", rel, str(target))])

    def _move(self, request: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        src_ref = request.get("src")
        dst_rel = request.get("dst")
        if not src_ref or not dst_rel:
            return ToolResult(False, "move: need src and dst", error="missing_arg")
        src = Path(src_ref)
        dst = confine(ctx.workdir, dst_rel)
        src_ok = src.exists() and str(src.resolve()).startswith(str(ctx.workdir.resolve()))
        if not src_ok or dst is None:
            return ToolResult(False, "move: src/dst invalid or outside workdir", error="bad_path")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)
        return ToolResult(True, f"moved {src.name} -> {dst_rel}",
                          data={"dst": str(dst)}, new_artifacts=[("file", dst_rel, str(dst))])
