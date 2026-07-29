"""Typed working-state blackboard for the local-agent orchestrator.

Doctrine (see plan): keep an EXTERNAL typed state so the noisy semantic controller
never has to rediscover the world from chat history every turn. Deterministic code
owns this object; the model only reads a compact rendered *slice* and refers to
things by short ids (F1, U2, M3) — it never reproduces long paths, URLs, or UUIDs.
The program resolves an id back to the real ref.

Pure Python, no LLM, no I/O — unit-testable in isolation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# id prefix per entity kind — the model outputs these short ids, code resolves them.
_KIND_PREFIX = {"file": "F", "url": "U", "memory": "M", "dir": "D", "artifact": "A"}
_DEFAULT_PREFIX = "E"


@dataclass
class Entity:
    """A thing the model may refer to by id. ``ref`` is the real value (path/url/…)
    the model must NEVER have to reproduce — code resolves ``id -> ref``."""
    id: str
    kind: str
    label: str            # short human/model-facing description
    ref: Any              # the real underlying value (abs path, full url, doc id…)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Budget:
    """Hard stop conditions the orchestrator (not the model) enforces."""
    max_calls: int = 40
    used_calls: int = 0
    max_seconds: Optional[float] = None
    started_at: float = field(default_factory=time.time)

    def call(self) -> None:
        self.used_calls += 1

    def exhausted(self) -> bool:
        if self.used_calls >= self.max_calls:
            return True
        if self.max_seconds is not None and (time.time() - self.started_at) >= self.max_seconds:
            return True
        return False

    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.used_calls)


@dataclass
class AgentState:
    """The blackboard. Everything the model needs to act is here; the prompt renders
    only the relevant slice of it."""
    task_goal: str = ""
    current_phase: str = "start"
    entities: Dict[str, Entity] = field(default_factory=dict)
    artifacts: Dict[str, Entity] = field(default_factory=dict)     # produced outputs (files written, answers)
    completed_steps: List[str] = field(default_factory=list)       # compact one-line summaries
    open_questions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    recent_tool_summaries: List[str] = field(default_factory=list)
    enabled_groups: Optional[set] = None          # None = all action groups legal (state-gating)
    suppressed: set = field(default_factory=set)   # action names temporarily removed (anti-repeat)
    executed_actions: set = field(default_factory=set)  # action names that ran successfully (finish gates)
    evidence: str = ""                             # last read page/text — carried into the finalizer
    finish_blocked: bool = False                   # FINISH hidden for one step after a gated finish
    budget: Budget = field(default_factory=Budget)
    finished: bool = False
    final_answer: Optional[str] = None
    _counters: Dict[str, int] = field(default_factory=dict)

    # --- entity bookkeeping ------------------------------------------------
    def add_entity(self, kind: str, label: str, ref: Any, *, meta: Optional[dict] = None) -> str:
        prefix = _KIND_PREFIX.get(kind, _DEFAULT_PREFIX)
        n = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = n
        eid = f"{prefix}{n}"
        self.entities[eid] = Entity(id=eid, kind=kind, label=label, ref=ref, meta=meta or {})
        return eid

    def add_artifact(self, kind: str, label: str, ref: Any, *, meta: Optional[dict] = None) -> str:
        aid = self.add_entity(kind, label, ref, meta=meta)
        self.artifacts[aid] = self.entities[aid]
        return aid

    def get(self, eid: str) -> Optional[Entity]:
        return self.entities.get(eid)

    def entities_of(self, kind: str) -> List[Entity]:
        return [e for e in self.entities.values() if e.kind == kind]

    # --- step / tool logging ----------------------------------------------
    def record_step(self, summary: str) -> None:
        self.completed_steps.append(summary)

    def record_tool(self, summary: str, *, keep: int = 8) -> None:
        """Store a COMPACT fact from a tool result (code summarizes, not the model),
        capped so context stays small for a low-context local model."""
        self.recent_tool_summaries.append(summary)
        if len(self.recent_tool_summaries) > keep:
            self.recent_tool_summaries = self.recent_tool_summaries[-keep:]

    def finish(self, answer: str) -> None:
        self.finished = True
        self.final_answer = answer

    # --- prompt rendering (compact slice) ---------------------------------
    def render_slice(self, *, max_entities: int = 12, max_steps: int = 6) -> str:
        lines: List[str] = [f"GOAL: {self.task_goal}", f"PHASE: {self.current_phase}"]
        if self.constraints:
            lines.append("CONSTRAINTS: " + "; ".join(self.constraints))
        ents = list(self.entities.values())[-max_entities:]
        if ents:
            lines.append("ENTITIES (refer to these by id):")
            for e in ents:
                lines.append(f"  {e.id} = {e.label} [{e.kind}]")
        if self.recent_tool_summaries:
            lines.append("RECENT RESULTS:")
            for s in self.recent_tool_summaries[-max_steps:]:
                lines.append(f"  - {s}")
        if self.open_questions:
            lines.append("OPEN: " + "; ".join(self.open_questions))
        lines.append(f"BUDGET: {self.budget.used_calls}/{self.budget.max_calls} calls used")
        return "\n".join(lines)
