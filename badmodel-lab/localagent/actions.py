"""The action IR — a small, closed, typed action space (the doctrine's core).

Instead of exposing a flat catalog of tools with unconstrained JSON args, the model
picks ONE action class from a state-filtered set of 3-8, then fills a few typed
slots. Each slot is an enum / entity-id / scalar — never a free-form path or nested
object. Deterministic code resolves entity ids to real refs and builds the tool
request. This module holds only the *specification*; parsing/validation is in ir.py.

Pure Python, no LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .state import AgentState

SlotType = str  # "enum" | "entity" | "string" | "int" | "bool"


@dataclass
class Slot:
    name: str
    type: SlotType
    required: bool = True
    enum_values: Optional[List[str]] = None      # for type == "enum"
    entity_kind: Optional[str] = None            # for type == "entity": restrict to this kind
    default: Any = None                          # filled programmatically when omitted
    help: str = ""

    def describe(self) -> str:
        if self.type == "enum":
            opts = "|".join(self.enum_values or [])
            return f"{self.name} (one of: {opts})"
        if self.type == "entity":
            k = f" {self.entity_kind}" if self.entity_kind else ""
            return f"{self.name} (a{k} entity id, e.g. F1)"
        d = f", default {self.default}" if (not self.required and self.default is not None) else ""
        return f"{self.name} ({self.type}{d})"


@dataclass
class ActionSpec:
    name: str                                    # UPPER_SNAKE, the token the router picks
    summary: str                                 # one line shown to the model
    tool: str                                    # tool name the executor dispatches to
    slots: List[Slot] = field(default_factory=list)
    read_only: bool = True                       # gates best-of-N: read-only accepts first valid candidate
    group: str = "core"                          # for state-dependent exposure (file|shell|web|memory|core)
    precondition: Callable[[AgentState], bool] = lambda s: True   # state-gating for legal_actions
    # Map validated slot values -> a concrete tool request. Default resolves entity
    # ids to their real refs so the tool never sees a model-authored path.
    build_request: Optional[Callable[[Dict[str, Any], AgentState], Dict[str, Any]]] = None

    def slot(self, name: str) -> Optional[Slot]:
        for s in self.slots:
            if s.name == name:
                return s
        return None

    def required_slots(self) -> List[Slot]:
        return [s for s in self.slots if s.required]

    def to_request(self, filled: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        if self.build_request is not None:
            return self.build_request(filled, state)
        # default: resolve entity ids -> refs, pass scalars through
        req: Dict[str, Any] = {}
        for s in self.slots:
            if s.name not in filled:
                continue
            val = filled[s.name]
            if s.type == "entity":
                ent = state.get(val)
                req[s.name] = ent.ref if ent else val
                req[s.name + "_id"] = val
            else:
                req[s.name] = val
        return req

    def describe(self) -> str:
        if not self.slots:
            return f"{self.name} — {self.summary}"
        slotdesc = "; ".join(s.describe() for s in self.slots)
        return f"{self.name} — {self.summary}  [slots: {slotdesc}]"


class ActionRegistry:
    """Holds all action specs; exposes only the currently-LEGAL ones per state."""

    def __init__(self, max_active: int = 8) -> None:
        self._specs: Dict[str, ActionSpec] = {}
        self.max_active = max_active

    def register(self, spec: ActionSpec) -> "ActionRegistry":
        self._specs[spec.name] = spec
        return self

    def get(self, name: str) -> Optional[ActionSpec]:
        return self._specs.get(name)

    def all(self) -> List[ActionSpec]:
        return list(self._specs.values())

    def legal_actions(self, state: AgentState) -> List[ActionSpec]:
        """State-dependent tool exposure: only actions in the state's enabled groups
        whose precondition holds, capped at ``max_active`` so a weak model never sees
        30 options at once. ``enabled_groups=None`` means all groups."""
        groups = getattr(state, "enabled_groups", None)
        suppressed = getattr(state, "suppressed", set())

        def base_ok(sp):
            return (groups is None or sp.group in groups) and _safe_precond(sp, state)

        # core actions (e.g. FINISH) are ALWAYS legal — never stripped by the cap OR by
        # suppression, so the agent can always stop. Suppression hides only non-core actions.
        core = [sp for sp in self._specs.values() if sp.group == "core" and base_ok(sp)]
        # after a gated finish, hide FINISH for one step so the model must take a productive action
        if getattr(state, "finish_blocked", False):
            core = [sp for sp in core if sp.name != "FINISH"]
        non_core = [sp for sp in self._specs.values()
                    if sp.group != "core" and base_ok(sp) and sp.name not in suppressed]
        return non_core[: max(0, self.max_active - len(core))] + core

    def render_menu(self, state: AgentState) -> str:
        legal = self.legal_actions(state)
        return "\n".join(f"  {sp.describe()}" for sp in legal)


def _safe_precond(spec: ActionSpec, state: AgentState) -> bool:
    try:
        return bool(spec.precondition(state))
    except Exception:
        return False
