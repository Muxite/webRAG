"""Parse → route → validate → repair for the tiny action language.

The model emits a compact, line-based action instead of JSON:

    ACTION: READ_FILE
    file: F1

or the terse pipe form:  ``read_file|F1``

This module turns that into a validated tool request, or into a *typed* error the
repairer can act on one field at a time. It is the doctrine's "convert one brittle
generation problem into several high-accuracy classification problems": routing is a
closed-enum pick, each slot is a small typed choice, and every failure is a compiler-
like signal (``unknown_entity``, ``invalid_enum``, …), never a stack trace.

Pure, deterministic, LLM-free — the whole point is that this layer cannot be fooled
by a noisy model and is exhaustively unit-testable with hand-written completions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .actions import ActionRegistry, ActionSpec, Slot
from .state import AgentState

# error codes (typed, machine-readable — the model gets these, not tracebacks)
UNKNOWN_ACTION = "unknown_action"
PARSE_ERROR = "parse_error"
MISSING_REQUIRED_SLOT = "missing_required_slot"
INVALID_ENUM = "invalid_enum"
UNKNOWN_ENTITY = "unknown_entity"
WRONG_ENTITY_KIND = "wrong_entity_kind"
TYPE_ERROR = "type_error"


@dataclass
class TypedError:
    code: str
    message: str
    slot: Optional[str] = None
    allowed: List[str] = field(default_factory=list)

    def line(self) -> str:
        base = f"reason={self.code}"
        if self.slot:
            base += f" slot={self.slot}"
        if self.allowed:
            base += f" allowed=[{','.join(map(str, self.allowed))}]"
        return base


def _norm_action(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", name.strip()).upper()


def parse_block(text: str) -> Tuple[Optional[str], Dict[str, str], List[str]]:
    """Parse the raw model completion into (action_name, kv_slots, positional_slots).

    Tolerant by design: ignores fences/blank lines, accepts an ``ACTION:`` prefix or a
    bare first line, and the ``name|a|b`` pipe form (positional). Never raises.
    """
    kv: Dict[str, str] = {}
    positional: List[str] = []
    action: Optional[str] = None
    for raw in (text or "").splitlines():
        line = raw.strip().strip("`").strip()
        if not line:
            continue
        if action is None:
            # first meaningful line names the action
            head = line
            if ":" in head and head.split(":", 1)[0].strip().lower() in ("action", "act", "tool"):
                head = head.split(":", 1)[1].strip()
            if "|" in head:
                parts = [p.strip() for p in head.split("|")]
                action = _norm_action(parts[0])
                positional = [p for p in parts[1:] if p != ""]
            else:
                action = _norm_action(head)
            continue
        # subsequent "key: value" lines fill named slots
        if ":" in line:
            k, v = line.split(":", 1)
            kv[k.strip().lower()] = v.strip()
        elif "|" in line and not positional:
            positional = [p.strip() for p in line.split("|") if p.strip()]
    return action, kv, positional


def route(action_name: Optional[str], registry: ActionRegistry, state: AgentState
          ) -> Tuple[Optional[ActionSpec], Optional[TypedError]]:
    legal = registry.legal_actions(state)
    legal_names = [sp.name for sp in legal]
    if not action_name:
        return None, TypedError(PARSE_ERROR, "no action found", allowed=legal_names)
    spec = registry.get(action_name)
    if spec is None or spec.name not in legal_names:
        return None, TypedError(UNKNOWN_ACTION, f"'{action_name}' is not a legal action here",
                                allowed=legal_names)
    return spec, None


def _coerce_and_check(slot: Slot, raw: str, state: AgentState) -> Tuple[Any, Optional[TypedError]]:
    v = raw.strip()
    if slot.type == "enum":
        for opt in slot.enum_values or []:
            if v.lower() == opt.lower():
                return opt, None
        return None, TypedError(INVALID_ENUM, f"{v!r} not a valid {slot.name}",
                                slot=slot.name, allowed=list(slot.enum_values or []))
    if slot.type == "entity":
        ent = state.get(v.upper())
        if ent is None:
            allowed = [e.id for e in (state.entities_of(slot.entity_kind) if slot.entity_kind
                                      else state.entities.values())]
            return None, TypedError(UNKNOWN_ENTITY, f"{v!r} is not a known entity id",
                                    slot=slot.name, allowed=allowed)
        if slot.entity_kind and ent.kind != slot.entity_kind:
            allowed = [e.id for e in state.entities_of(slot.entity_kind)]
            return None, TypedError(WRONG_ENTITY_KIND,
                                    f"{ent.id} is a {ent.kind}, need a {slot.entity_kind}",
                                    slot=slot.name, allowed=allowed)
        return ent.id, None
    if slot.type == "int":
        m = re.search(r"-?\d+", v)
        if not m:
            return None, TypedError(TYPE_ERROR, f"{v!r} is not an integer", slot=slot.name)
        return int(m.group()), None
    if slot.type == "bool":
        return v.lower() in ("y", "yes", "true", "1", "on"), None
    return v, None  # string


def fill_and_validate(spec: ActionSpec, kv: Dict[str, str], positional: List[str],
                      state: AgentState) -> Tuple[Dict[str, Any], List[TypedError]]:
    """Fill only this action's slots. Positional args map to slots in declared order;
    named args win. Missing optional slots take their programmatic default. Returns
    (filled, errors) — errors is empty iff the request is fully valid."""
    filled: Dict[str, Any] = {}
    errors: List[TypedError] = []
    # positional → slot names in order
    pos_map = {spec.slots[i].name: positional[i] for i in range(min(len(positional), len(spec.slots)))}
    for slot in spec.slots:
        raw = kv.get(slot.name.lower(), pos_map.get(slot.name))
        if raw is None or raw == "":
            if slot.required and slot.default is None:
                if slot.type == "entity":
                    allowed = [e.id for e in (state.entities_of(slot.entity_kind) if slot.entity_kind
                                              else state.entities.values())]
                else:
                    allowed = list(slot.enum_values or [])
                errors.append(TypedError(MISSING_REQUIRED_SLOT, f"{slot.name} is required",
                                         slot=slot.name, allowed=allowed))
            elif slot.default is not None:
                filled[slot.name] = slot.default
            continue
        val, err = _coerce_and_check(slot, raw, state)
        if err is not None:
            errors.append(err)
        else:
            filled[slot.name] = val
    return filled, errors


def repair_prompt(spec: ActionSpec, errors: List[TypedError], filled: Dict[str, Any]
                  ) -> Tuple[str, Optional[str]]:
    """ONE-field repair, phrased as a POSITIVE direct question — not an ``INVALID_ACTION`` error
    dump, which weak models misread as a system error and answer conversationally about. Asks for
    just the single slot value; the caller grammar-constrains enum/entity slots via err.allowed.
    Returns (prompt, target_slot)."""
    if not errors:
        return "", None
    err = errors[0]
    target = err.slot
    if not target:
        return f"Choose one action to use for {spec.name}. Reply with only the action name.", None
    slot = spec.slot(target)
    hint = (slot.help or "") if slot else ""
    lines = [f"For the {spec.name} action, provide the `{target}`."]
    if err.allowed:
        lines.append(f"It must be one of: {', '.join(map(str, err.allowed))}.")
    elif err.code == INVALID_ENUM or err.code == TYPE_ERROR:
        lines.append(f"({err.message})")
    if hint:
        lines.append(hint)
    lines.append(f"Reply with ONLY the {target} value — nothing else.")
    return "\n".join(lines), target
