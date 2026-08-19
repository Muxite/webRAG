"""The family registry: what exists, what each one asks, and where its SHIPPED text lives.

One place that answers "which families are there", so ``runner`` and ``analyze``
stop carrying their own hardcoded dictionaries and a new family becomes one entry
rather than four edits.

SHIPPED IS RESOLVED FROM SETTINGS FIRST
---------------------------------------
Three of the four shipped prompts live in ``agent/app/idea_dag_settings.json``,
not in source, and the engine prefers the settings value::

    # actions.py:2304
    system_content = self.settings.get("verify_system_prompt") or self._DEFAULT_SYSTEM_PROMPT

v1's SHIPPED arm imported the class constant alone. The two texts happen to agree
today (modulo an em-dash and brace escaping), so v1's numbers stand -- but nothing
checked that, and editing a JSON file would have moved what the engine sends while
leaving the benchmark arm behind. ``shipped_prompt`` resolves the same way the
engine does.

Settings values are ``.format()`` templates, so their JSON examples carry doubled
braces (``{{"verdict": ...}}``). Those are un-doubled here; shipping literal ``{{``
to a model would spike parse failures and read as a model deficit.

NOT EVERY FAMILY HAS A SHIPPED ARM
----------------------------------
``select`` has none. v1 ran SHIPPED x select and discarded every cell in analysis,
because a four-way truth verdict cannot name one of five candidate marathons. Here
``shipped_key`` is ``None`` and the runner never places those calls, so the waste
is avoided rather than cleaned up afterwards.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.app.promptbench.availability import Item
from agent.app.promptbench.items import build_select_items, build_verify_items, load_all_specs
from agent.app.promptbench.items_engine import build_followup_items, build_goal_achieved_items
from agent.app.promptbench.items_keystone import build_keystone_claim_items

SETTINGS_PATH = Path("agent/app/idea_dag_settings.json")


def _settings() -> Mapping[str, Any]:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _undouble_braces(text: str) -> str:
    return text.replace("{{", "{").replace("}}", "}")


def _verify_fallback() -> str:
    from agent.app.idea_policies.actions import VerifyLeafAction
    return VerifyLeafAction._DEFAULT_SYSTEM_PROMPT


def _followup_fallback() -> str:
    """``got_operations.check_needs_followup``'s inline default.

    Inline in a ``settings.get(key, default)`` call rather than a module constant,
    so it cannot be imported. It is present in the settings JSON, which is what the
    engine reads in practice; this fallback exists for the case where it is not, and
    the parity test asserts the two agree.
    """
    return (
        "You are a follow-up detector in a Graph-of-Thought research system. A leaf "
        "task has just completed and produced a result. Decide whether that result "
        "reveals a GENUINE, concrete follow-up investigation required to satisfy the "
        "parent goal and not already covered by existing sibling tasks. Return JSON: "
        '{"needs_followup": boolean, "reason": string}.'
    )


def _merge_fallback() -> str:
    return (
        "You are the Aggregate operation in a Graph-of-Thought system. Combine child "
        "node results into a coherent summary. Remove redundancy, extract key findings. "
        "Evaluate if the original goal has been achieved. Return JSON: {summary: string, "
        "key_findings: [string, ...], goal_achieved: boolean, goal_evaluation: string, "
        "missing_requirements: [string, ...]}."
    )


def _step_confidence_fallback() -> str:
    """``GoTOperations.judge_step_confidence``'s system prompt.

    The only one of the four with no settings key at all -- it is a literal in
    ``got_operations.py``. Copied verbatim; the parity test asserts the copy still
    matches the source text so it cannot drift unnoticed.
    """
    return (
        "You are a step-level verifier in a Graph-of-Thought research system. A single "
        "sub-task has just produced an output. Estimate, on a 0-1 scale, how confident you "
        "are that this step's output is CORRECT and ON-TRACK toward the overall task, using "
        "ONLY what is visible below. You do NOT know the ground-truth answer and must not "
        "assume one; judge from the resolved content alone. Return JSON: "
        '{"confidence": number between 0 and 1, "reason": string}.'
    )


def shipped_prompt(key: Optional[str], fallback: Callable[[], str]) -> str:
    """Resolve a shipped prompt the way the engine does: settings first."""
    if key:
        value = _settings().get(key)
        if isinstance(value, str) and value.strip():
            return _undouble_braces(value)
    return fallback()


@dataclasses.dataclass(frozen=True)
class Family:
    """One decision point, rendered as items.

    ``shipped_aliases`` maps the ENGINE's output vocabulary onto the family's option
    vocabulary. ``None`` marks a genuine abstention rather than a parse failure.
    v1 needed this because its ``verify`` family asked SATISFIES/VIOLATES while the
    shipped prompt answers TRUE/FALSE -- well-formed completions scored a 92% parse
    failure until the two were aligned. Families added since speak the engine's own
    vocabulary, so their alias tables are identity or empty.
    """

    name: str
    build: Callable[[Sequence[Dict[str, Any]]], List[Item]]
    choices: Tuple[str, ...]
    shipped_key: Optional[str]
    shipped_fallback: Optional[Callable[[], str]]
    shipped_aliases: Mapping[str, Optional[str]] = dataclasses.field(default_factory=dict)

    @property
    def has_shipped(self) -> bool:
        return self.shipped_fallback is not None

    def shipped(self) -> str:
        if self.shipped_fallback is None:
            raise ValueError(f"family {self.name!r} has no shipped arm")
        return shipped_prompt(self.shipped_key, self.shipped_fallback)


# The engine's verify vocabulary, mapped onto SATISFIES/VIOLATES.
_VERIFY_ALIASES: Mapping[str, Optional[str]] = {
    "TRUE": "SATISFIES",
    "PARTIALLY_TRUE": "SATISFIES",
    "FALSE": "VIOLATES",
    "UNVERIFIABLE": None,
}

# keystone_claim already asks in the engine's own vocabulary; only the
# four-way-to-two-way narrowing needs declaring.
_CLAIM_ALIASES: Mapping[str, Optional[str]] = {
    "PARTIALLY_TRUE": "TRUE",
    "UNVERIFIABLE": None,
}

_FAMILIES = (
    Family("verify", build_verify_items, ("SATISFIES", "VIOLATES"),
           "verify_system_prompt", _verify_fallback, _VERIFY_ALIASES),
    Family("select", build_select_items, (),
           None, None, {}),
    Family("keystone_claim", build_keystone_claim_items, ("TRUE", "FALSE"),
           "verify_system_prompt", _verify_fallback, _CLAIM_ALIASES),
    Family("followup", build_followup_items, ("YES", "NO"),
           "got_reexpand_followup_system_prompt", _followup_fallback,
           {"TRUE": "YES", "FALSE": "NO"}),
    Family("goal_achieved", build_goal_achieved_items, ("ACHIEVED", "NOT_ACHIEVED"),
           "merge_system_prompt", _merge_fallback,
           {"TRUE": "ACHIEVED", "FALSE": "NOT_ACHIEVED"}),
    # Same items as ``verify``; the difference is that the arms ask for a
    # confidence number, and the metric layer scores it. Kept a separate family so
    # the accuracy comparison and the calibration comparison never share a cell.
    Family("calibration", build_verify_items, ("SATISFIES", "VIOLATES"),
           None, _step_confidence_fallback, _VERIFY_ALIASES),
)

REGISTRY: Dict[str, Family] = {f.name: f for f in _FAMILIES}
ALL_FAMILIES: Tuple[str, ...] = tuple(REGISTRY)


def build(names: Sequence[str], specs: Optional[Sequence[Dict[str, Any]]] = None
          ) -> Dict[str, List[Item]]:
    """Items for the named families, in registry order."""
    unknown = set(names) - set(REGISTRY)
    if unknown:
        raise KeyError(f"unknown families: {sorted(unknown)}")
    # Unfiltered on purpose: each builder decides what it can use. Narrowing here
    # would silently starve any family not built on candidate sets.
    specs = load_all_specs() if specs is None else specs
    return {name: REGISTRY[name].build(specs) for name in ALL_FAMILIES if name in set(names)}
