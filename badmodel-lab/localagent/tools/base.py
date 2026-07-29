"""Tool contract. A tool is a narrow, deterministic capability the orchestrator runs
after the model has SELECTED and the code has VALIDATED an action. Tools never see
model-authored paths/ids directly — the action's build_request resolved entity ids to
real refs first. Every tool returns a COMPACT summary (code summarizes, not the model)
so a low-context local model isn't flooded with raw output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple


@dataclass
class ToolContext:
    """Everything a tool may touch. ``workdir`` is the sandbox scratch root; tools MUST
    confine filesystem access to it (belt-and-suspenders with the container itself)."""
    workdir: Path
    memory: Any = None            # a MemoryStore (see tools/memory.py), optional
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    summary: str                                  # one compact line -> state.recent_tool_summaries
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    # (kind, label, ref) tuples the loop registers as new entities the model can refer to
    new_entities: List[Tuple[str, str, Any]] = field(default_factory=list)
    # (kind, label, ref) produced outputs recorded as artifacts (files written, answers)
    new_artifacts: List[Tuple[str, str, Any]] = field(default_factory=list)


class Tool(Protocol):
    name: str

    def execute(self, request: Dict[str, Any], ctx: ToolContext) -> ToolResult: ...


def confine(workdir: Path, rel: str) -> Optional[Path]:
    """Resolve ``rel`` under ``workdir`` and refuse anything that escapes it (``..``,
    absolute paths, sym/hardlink traversal). Returns None on escape — the caller turns
    that into a ToolResult error, never an exception."""
    try:
        base = workdir.resolve()
        target = (base / rel).resolve()
        if target == base or base in target.parents:
            return target
        return None
    except Exception:
        return None
