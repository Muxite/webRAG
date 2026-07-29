"""
Plan-library adapter — the ONLY module the engine talks to for retrieved plans.

A filled :class:`~agent.app.plan_library.schema.PlanTemplate` is action-agnostic (it is
``compiled_plan.py``'s shape: instruction/expect prose per leaf). This module maps it onto the
native Graph-of-Thought vocabulary the engine actually executes, so a library-sourced subtree
behaves — to contract checking, re-expansion, telemetry and the confidence judge — exactly like
an LLM-invented one:

  * each filled leaf becomes a ``search`` candidate whose ``details.query`` is built from the
    leaf's *structured slot values* (see ``schema._build_query``), its ``details.intent`` is the
    filled instruction, and its ``details.expect`` is the filled contract, carried through
    unchanged so ``contract_satisfaction.py`` applies identically;
  * the template's ``aggregation`` becomes the PARENT node's ``details.intent``, which is the
    channel ``MergeLeafAction`` reads as ``parent_intent`` when it synthesizes children
    (``actions.py``). Emitting a literal ``merge`` candidate would be pointless: the engine
    strips LLM-authored merge candidates because merge nodes are system-managed
    (``idea_engine._handle_expansion_node``), so parent details are how an organic plan's
    aggregation guidance reaches the merge too;
  * ``depends_on`` chains ride the engine's EXISTING dependency mechanism — ``details.requires_data
    = {type, source_node_id}``, which both blocks a leaf until its source is DONE-and-ready
    (``idea_engine._has_required_data``) and forces sequential execution
    (``idea_sequencing.detect_state_dependencies`` / the "DEPENDENCY DETECTED" branch). Because
    ``source_node_id`` is a real graph node id, it can only be written AFTER ``graph.expand()``
    has minted the children — hence :func:`link_dependencies`, which the caller runs on the
    freshly created nodes. Until then the dependency rides along as blueprint ids in
    ``details._plan_library_depends_on``.

Pure module: no I/O, no LLM, no engine state. Retrieval decides *which* template; this decides
what the engine gets to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Union

from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.data_contracts import ContractRegistry, default_contract_registry
from agent.app.plan_library.schema import (
    FilledPlan,
    PlanTemplate,
    TemplateValidationError,
    bind_template,
)
from agent.app.testing import compiled_plan

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaNode

#: Where a library-sourced subtree came from — mirrored into the contract log's ``origin``.
ORIGIN_AUTO = "plan_library_auto"
ORIGIN_ACTION = "plan_library_action"

#: Private candidate/node detail keys. Underscore-prefixed like the engine's other internal
#: markers (``_got_reexpand_count``, ``_got_reexpanded``) so they never collide with the
#: LLM-authored ``details`` vocabulary in :class:`DetailKey`.
PLAN_LIBRARY_TEMPLATE_ID = "_plan_library_template_id"
PLAN_LIBRARY_ORIGIN = "_plan_library_origin"
PLAN_LIBRARY_LEAF_ID = "_plan_library_leaf_id"
PLAN_LIBRARY_DEPENDS_ON = "_plan_library_depends_on"
PLAN_LIBRARY_DEPENDS_ON_NODES = "_plan_library_depends_on_nodes"
#: SINGLE-USE hand-off of an already-resolved expansion, from the on-demand path's completion
#: hook to ``_handle_expansion_node`` (which pops it). The ON-DEMAND action has its plan in
#: hand the moment it completes, but growing children is the engine's job — routing it back
#: through the one expansion entry point is what gives a library-sourced subtree the same
#: duplicate filter, beam width, child metadata, contract-log rows and post-expansion hooks
#: whichever call site produced it, with nothing re-implemented per call site.
PLAN_LIBRARY_PENDING = "_plan_library_pending"

_MAX_TITLE_CHARS = 120


@dataclass(frozen=True)
class PlanLibraryExpansion:
    """What a retrieval hit contributes to one expansion step."""

    #: ``graph.expand()``-shaped candidate dicts, one per filled leaf, in wave-friendly
    #: declaration order.
    candidates: List[Dict[str, Any]]
    #: Details to merge onto the PARENT node before/after expanding (aggregation guidance +
    #: the expansion meta the LLM path would have supplied).
    parent_details: Dict[str, Any]
    aggregation: str
    template_id: str
    origin: str
    #: The filled plan in ``compiled_plan`` shape — for logging/telemetry and offline diffing
    #: against hand-authored plans via ``compiled_plan.plan_structure``.
    plan: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.candidates)


def candidates_from_template(
    template: Union[PlanTemplate, Mapping[str, Any]],
    slot_values: Mapping[str, Any],
    *,
    origin: str = ORIGIN_AUTO,
    contracts: Optional[ContractRegistry] = None,
) -> PlanLibraryExpansion:
    """Fill ``template`` with ``slot_values`` and adapt it into GoT candidates.

    :raises TemplateValidationError: if the template is invalid or the slot values do not fill
        it — the caller (retrieval) treats that as "no match" and falls back to organic
        expansion, matching ``contract_satisfaction.py``'s fail-toward-silence philosophy.
    """
    return candidates_from_filled_plan(
        bind_template(template, slot_values), origin=origin, contracts=contracts
    )


def candidates_from_filled_plan(
    filled: FilledPlan,
    *,
    origin: str = ORIGIN_AUTO,
    contracts: Optional[ContractRegistry] = None,
) -> PlanLibraryExpansion:
    """Adapt an already-filled template into GoT expansion candidates.

    Takes :class:`FilledPlan` rather than the bare ``{"leaves": [...], "aggregation": ...}``
    dict because the structured per-leaf bindings (the search query derived from slot VALUES)
    are exactly what that dict has thrown away — re-deriving a query from the filled prose is
    the LLM call this design exists to avoid.
    """
    if not isinstance(filled, FilledPlan):
        raise TemplateValidationError(
            f"expected a FilledPlan (see schema.bind_template), got {type(filled).__name__}"
        )
    registry = contracts or default_contract_registry()
    provides = registry.contract_for_action(IdeaActionType.SEARCH.value)

    candidates: List[Dict[str, Any]] = []
    for leaf in filled.leaves:
        title = _title_for(leaf.instruction, leaf.id)
        details: Dict[str, Any] = {
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.QUERY.value: leaf.query,
            DetailKey.INTENT.value: leaf.instruction,
            DetailKey.JUSTIFICATION.value: (
                f"Plan library template '{filled.template_id}' leaf '{leaf.id}'"
            ),
            DetailKey.EXPECT.value: leaf.expect,
            # ``_parse_candidates`` derives goal/original_goal from the candidate title when
            # the model supplies no explicit goal; mirror that so downstream prompts and the
            # children-metadata threading see the same shape.
            DetailKey.GOAL.value: title,
            DetailKey.ORIGINAL_GOAL.value: title,
            PLAN_LIBRARY_TEMPLATE_ID: filled.template_id,
            PLAN_LIBRARY_ORIGIN: origin,
            PLAN_LIBRARY_LEAF_ID: leaf.id,
            PLAN_LIBRARY_DEPENDS_ON: list(leaf.depends_on),
        }
        if provides is not None:
            details[DetailKey.PROVIDES_DATA.value] = {"type": provides.name}
        candidates.append({"title": title, "details": details, "score": None})

    has_deps = any(leaf.depends_on for leaf in filled.leaves)
    parent_details = {
        # MergeLeafAction reads this as ``parent_intent`` (directly, or via the merge node's
        # fallback to its parent) — the aggregation discipline is the template's real payload.
        DetailKey.INTENT.value: filled.aggregation,
        DetailKey.EXPANSION_META.value: {DetailKey.EXECUTE_ALL_CHILDREN.value: not has_deps},
        PLAN_LIBRARY_TEMPLATE_ID: filled.template_id,
        PLAN_LIBRARY_ORIGIN: origin,
    }
    return PlanLibraryExpansion(
        candidates=candidates,
        parent_details=parent_details,
        aggregation=filled.aggregation,
        template_id=filled.template_id,
        origin=origin,
        plan=filled.to_plan(),
    )


def link_dependencies(nodes: Sequence["IdeaNode"]) -> int:
    """Resolve blueprint-level ``depends_on`` into real ``requires_data`` node references.

    Run this on the children ``graph.expand()`` just created (node ids do not exist before
    that). Each node carrying an unsatisfied dependency gets
    ``details.requires_data = {"type": <producer contract>, "source_node_id": <sibling id>}``,
    which is the engine's existing gate: ``_has_required_data`` blocks the leaf until the
    source is DONE with a usable result, ``_handle_active_node`` schedules the source first,
    and ``detect_state_dependencies`` drops the batch out of parallel execution.

    ``requires_data`` names a SINGLE source, so a fan-in leaf points at its *deepest*
    dependency (largest topological wave index, ties broken by declared order). The shallower
    ones still run first — they are in earlier waves of the same batch — and every resolved
    sibling id is recorded in ``_plan_library_depends_on_nodes`` for the contract log.

    :returns: how many nodes had a dependency written.
    """
    tagged = [n for n in nodes if isinstance(n.details.get(PLAN_LIBRARY_LEAF_ID), str)]
    if not tagged:
        return 0

    node_by_leaf: Dict[str, "IdeaNode"] = {}
    for node in tagged:
        node_by_leaf.setdefault(str(node.details[PLAN_LIBRARY_LEAF_ID]), node)

    depth = _wave_depths(tagged)
    registry = default_contract_registry()
    linked = 0
    for node in tagged:
        deps = node.details.get(PLAN_LIBRARY_DEPENDS_ON) or []
        if not isinstance(deps, list):
            continue
        sources = [node_by_leaf[str(d)] for d in deps if str(d) in node_by_leaf]
        if not sources:
            continue
        node.details[PLAN_LIBRARY_DEPENDS_ON_NODES] = [s.node_id for s in sources]
        deepest = max(
            sources,
            key=lambda s: depth.get(str(s.details.get(PLAN_LIBRARY_LEAF_ID)), 0),
        )
        contract = registry.contract_for_action(
            deepest.details.get(DetailKey.ACTION.value)
        ) or registry.contract_for_action(IdeaActionType.SEARCH.value)
        node.details[DetailKey.REQUIRES_DATA.value] = {
            "type": contract.name if contract else "urls_from_search",
            "source_node_id": deepest.node_id,
        }
        linked += 1
    return linked


def _wave_depths(nodes: Sequence["IdeaNode"]) -> Dict[str, int]:
    """Topological wave index per leaf id, via ``compiled_plan.topological_waves``.

    Fails open (all-zero) on a malformed dependency set: the filled plan was already validated
    at fill time, so this can only trip if a caller hand-assembled the details.
    """
    leaves = [
        {
            "id": str(node.details.get(PLAN_LIBRARY_LEAF_ID)),
            "depends_on": [str(d) for d in (node.details.get(PLAN_LIBRARY_DEPENDS_ON) or [])],
        }
        for node in nodes
    ]
    known = {leaf["id"] for leaf in leaves}
    for leaf in leaves:
        leaf["depends_on"] = [d for d in leaf["depends_on"] if d in known and d != leaf["id"]]
    try:
        waves = compiled_plan.topological_waves(leaves)
    except compiled_plan.PlanValidationError:
        return {}
    return {leaf_id: index for index, wave in enumerate(waves) for leaf_id in wave}


def _title_for(instruction: str, fallback: str) -> str:
    """A short node title from a leaf's instruction (its first sentence, clipped)."""
    text = " ".join(str(instruction or "").split())
    first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0] if text else ""
    title = first or text
    if len(title) > _MAX_TITLE_CHARS:
        clipped = title[: _MAX_TITLE_CHARS - 3]
        title = (clipped.rsplit(" ", 1)[0] if " " in clipped else clipped) + "..."
    return title or fallback
