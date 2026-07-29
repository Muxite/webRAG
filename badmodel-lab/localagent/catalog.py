"""Default action catalog — wires narrow, typed action specs to their tools, grouped
for state-dependent exposure (a task enables only the groups it needs, so a weak model
never sees more than a handful of legal actions).
"""
from __future__ import annotations

from typing import Dict, Optional

from .actions import ActionRegistry, ActionSpec, Slot
from .tools.base import Tool
from .tools.files import FileTool
from .tools.memory import MemoryTool
from .tools.shell import ShellTool


def build_default_registry(max_active: int = 8) -> ActionRegistry:
    reg = ActionRegistry(max_active=max_active)

    # --- files ----------------------------------------------------------
    reg.register(ActionSpec(
        "LIST_DIR", "list the files in a directory", tool="file", group="file",
        slots=[Slot("path", "string", required=False, default=".")],
        build_request=lambda f, s: {"op": "list", "path": f.get("path", ".")}))
    reg.register(ActionSpec(
        "READ_FILE", "read a known file's contents", tool="file", group="file",
        slots=[Slot("file", "entity", entity_kind="file")],
        precondition=lambda s: bool(s.entities_of("file")),
        build_request=lambda f, s: {"op": "read", "file": s.get(f["file"]).ref}))
    reg.register(ActionSpec(
        "WRITE_FILE", "create or overwrite a file", tool="file", group="file", read_only=False,
        slots=[Slot("path", "string"), Slot("content", "string")],
        build_request=lambda f, s: {"op": "write", "path": f["path"], "content": f.get("content", "")}))
    reg.register(ActionSpec(
        "MOVE_FILE", "move or rename a known file", tool="file", group="file", read_only=False,
        slots=[Slot("src", "entity", entity_kind="file"), Slot("dst", "string")],
        precondition=lambda s: bool(s.entities_of("file")),
        build_request=lambda f, s: {"op": "move", "src": s.get(f["src"]).ref, "dst": f["dst"]}))

    # --- shell (read-only allow-listed DSL) -----------------------------
    reg.register(ActionSpec(
        "COUNT_LINES", "count lines in a file (optionally matching a pattern)", tool="shell",
        group="shell", slots=[Slot("file", "entity", entity_kind="file"),
                              Slot("pattern", "string", required=False, default=None)],
        precondition=lambda s: bool(s.entities_of("file")),
        build_request=lambda f, s: {"op": "count_lines", "file": s.get(f["file"]).ref,
                                    "pattern": f.get("pattern")}))
    reg.register(ActionSpec(
        "WORD_COUNT", "count words in a file", tool="shell", group="shell",
        slots=[Slot("file", "entity", entity_kind="file")],
        precondition=lambda s: bool(s.entities_of("file")),
        build_request=lambda f, s: {"op": "word_count", "file": s.get(f["file"]).ref}))
    reg.register(ActionSpec(
        "HEAD_FILE", "show the first N lines of a file", tool="shell", group="shell",
        slots=[Slot("file", "entity", entity_kind="file"), Slot("n", "int", required=False, default=10)],
        precondition=lambda s: bool(s.entities_of("file")),
        build_request=lambda f, s: {"op": "head", "file": s.get(f["file"]).ref, "n": f.get("n", 10)}))
    reg.register(ActionSpec(
        "DISK_USAGE", "total size of a directory", tool="shell", group="shell",
        slots=[Slot("dir", "string", required=False, default=".")],
        build_request=lambda f, s: {"op": "disk_usage", "dir": f.get("dir", ".")}))
    reg.register(ActionSpec(
        "FIND_FILE", "find files by name under the workdir", tool="shell", group="shell",
        slots=[Slot("name", "string")],
        build_request=lambda f, s: {"op": "find", "name": f["name"]}))

    # --- web (read-only) ------------------------------------------------
    reg.register(ActionSpec(
        "WEB_SEARCH", "search the web; returns pages as U-ids", tool="web", group="web",
        slots=[Slot("query", "string")],
        build_request=lambda f, s: {"op": "search", "query": f["query"]}))
    reg.register(ActionSpec(
        "WEB_READ", "read a found web page by its U-id", tool="web", group="web",
        slots=[Slot("url", "entity", entity_kind="url")],
        precondition=lambda s: bool(s.entities_of("url")),
        build_request=lambda f, s: {"op": "read", "url": s.get(f["url"]).ref}))

    # --- memory ---------------------------------------------------------
    reg.register(ActionSpec(
        "REMEMBER", "save a fact to durable memory for later", tool="memory", group="memory",
        read_only=False, slots=[Slot("text", "string")],
        build_request=lambda f, s: {"op": "remember", "text": f["text"]}))
    reg.register(ActionSpec(
        "RECALL", "look up a fact from durable memory", tool="memory", group="memory",
        slots=[Slot("query", "string")],
        build_request=lambda f, s: {"op": "recall", "query": f["query"]}))

    # --- core -----------------------------------------------------------
    reg.register(ActionSpec(
        "FINISH", "give the final answer and stop", tool="finish", group="core",
        slots=[Slot("answer", "string")],
        build_request=lambda f, s: {"answer": f["answer"]}))
    return reg


def build_default_tools(web_tool: Optional[Tool] = None, include_shell: bool = True) -> Dict[str, Tool]:
    tools: Dict[str, Tool] = {"file": FileTool(), "memory": MemoryTool()}
    if include_shell:
        tools["shell"] = ShellTool()
    if web_tool is not None:
        tools["web"] = web_tool
    return tools
