"""Sequential reordering and state-dependency helpers from IdeaDagEngine.

Reorders children so data-producing work (search, visits with URLs) runs before
data-consuming work (think, save, merge). Detects state dependencies among
siblings to decide between parallel and sequential execution.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from agent.app.idea_dag import IdeaDag, IdeaNode
from agent.app.idea_policies import DetailKey
from agent.app.idea_policies.action_constants import NodeDetailsExtractor


def reorder_for_sequential(
    graph: IdeaDag,
    selected: IdeaNode,
    eligible: List[str],
    step_index: int,
) -> Optional[IdeaNode]:
    selected_action = NodeDetailsExtractor.get_action(selected.details) or ""
    # If a visit node has no explicit URL, it often depends on a sibling search node
    # to provide URLs. In sequential mode, enforce search-before-visit regardless
    # of score so we don't execute a visit prematurely and fail with "missing URL".
    if selected_action.lower() == "visit":
        url = (
            selected.details.get("optional_url")
            or selected.details.get(DetailKey.URL.value)
            or selected.details.get(DetailKey.LINK.value)
            or selected.details.get("url")
            or selected.details.get("link")
        )
        has_url = isinstance(url, str) and url.startswith(("http://", "https://"))
        if not has_url:
            search_candidates: List[IdeaNode] = []
            for nid in eligible:
                if nid == selected.node_id:
                    continue
                child = graph.get_node(nid)
                if not child or child.status.value in ("done", "failed", "skipped"):
                    continue
                child_action = NodeDetailsExtractor.get_action(child.details) or ""
                if child_action.lower() == "search":
                    search_candidates.append(child)
            if search_candidates:
                # Prefer the highest-scored search (or first if unscored).
                best_search = max(search_candidates, key=lambda n: n.score if n.score is not None else float("-inf"))
                return best_search

    data_consuming = {"think", "save", "merge"}
    if selected_action.lower() not in data_consuming:
        return None

    data_producing_candidates: List[IdeaNode] = []
    for nid in eligible:
        if nid == selected.node_id:
            continue
        child = graph.get_node(nid)
        if not child or child.status.value == "done":
            continue
        child_action = NodeDetailsExtractor.get_action(child.details) or ""
        if child_action.lower() == "search":
            data_producing_candidates.append(child)
        elif child_action.lower() == "visit":
            url = child.details.get("optional_url") or child.details.get("url") or child.details.get("link") or ""
            has_url = isinstance(url, str) and url.startswith(("http://", "https://"))
            has_link_idea = bool(child.details.get("link_idea"))
            if has_url or has_link_idea:
                data_producing_candidates.append(child)

    if not data_producing_candidates:
        return None

    for candidate in data_producing_candidates:
        url = candidate.details.get("optional_url") or candidate.details.get("url") or candidate.details.get("link") or ""
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return candidate

    return data_producing_candidates[0]


def detect_state_dependencies(
    graph: IdeaDag,
    candidate_ids: List[str],
    logger: logging.Logger,
) -> bool:
    from agent.app.idea_policies.action_constants import NodeDetailsExtractor
    from agent.app.idea_policies.base import IdeaActionType

    has_search = False
    has_visit = False
    visit_needs_url = False
    has_data_dependencies = False

    for node_id in candidate_ids:
        node = graph.get_node(node_id)
        if not node:
            continue

        action = NodeDetailsExtractor.get_action(node.details)
        if action == IdeaActionType.SEARCH.value:
            has_search = True
        elif action == IdeaActionType.VISIT.value:
            has_visit = True
            url = node.details.get(DetailKey.URL.value) or node.details.get(DetailKey.LINK.value) or node.details.get("url") or node.details.get("link")
            if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
                visit_needs_url = True

            requires_data = node.details.get(DetailKey.REQUIRES_DATA.value)
            if requires_data and isinstance(requires_data, dict):
                source_node_id = requires_data.get("source_node_id")
                if source_node_id and source_node_id in candidate_ids:
                    has_data_dependencies = True
                    logger.info(f"[DEPENDENCY] Node {node_id} requires data from sibling {source_node_id} - forcing sequential")

    if has_search and has_visit and visit_needs_url:
        return True

    if has_data_dependencies:
        return True

    return False


def siblings_are_independent(
    graph: IdeaDag,
    candidate_ids: List[str],
    mandate: IdeaNode,
    logger: logging.Logger,
) -> Tuple[bool, str]:
    """Check if candidates can safely execute in parallel (deterministic, no LLM).

    Applies ordered rules: state dependencies, unresolved slots, a shared race
    group, concrete URLs, mixed search/visit with URLs, disjoint searches, else
    no independence.

    :returns: (independent, reason) with reason for debugging.
    """
    if detect_state_dependencies(graph, candidate_ids, logger):
        return False, "state_dependency"

    from agent.app.idea_policies.dataflow import unresolved_slots

    nodes: List[IdeaNode] = []
    for node_id in candidate_ids:
        node = graph.get_node(node_id)
        if node is not None:
            nodes.append(node)

    for node in nodes:
        if unresolved_slots(node.details, title=node.title):
            return False, "unresolved_slot"

    # An authored race group IS a declaration of independence: its members are different
    # routes to the SAME fact, so none can consume another's output. The four heuristics
    # below only recognise independence through URL/query shape, and a realistic race pairs
    # (say) a SEARCH with a THINK, which matches none of them — the group would wrongly
    # serialize and the concurrency the mechanism exists for would never happen. Checked
    # after the two safety gates above (a state dependency or an unresolved slot still wins:
    # those are facts about dispatch, the label is only an authoring hint) and before the
    # heuristics. Requires the WHOLE batch to share one label, so a mixed batch of racers and
    # unrelated siblings is not waved through on the racers' evidence.
    if len(nodes) >= 2:
        labels = {
            (n.details.get(DetailKey.RACE_GROUP.value) or "").strip()
            for n in nodes
        }
        if len(labels) == 1 and next(iter(labels)):
            return True, "race_group"

    reason = disjoint_approach_reason(nodes)
    if reason:
        return True, reason

    return False, "no_independence_evidence"


def concrete_url(details) -> Optional[str]:
    """The first http(s) URL this node carries, under any of the five spellings in use."""
    for key in ("optional_url", DetailKey.URL.value, DetailKey.LINK.value, "url", "link"):
        value = details.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def disjoint_approach_reason(nodes: List[IdeaNode]) -> Optional[str]:
    """Do these siblings take demonstrably DIFFERENT approaches? Which evidence says so?

    The three URL/query shape heuristics :func:`siblings_are_independent` ends with, lifted
    out verbatim so a second consumer can reuse them without a graph or a mandate:
    ``alternative_branch.infer_race_groups`` needs exactly this "different route" half of
    the race definition. Necessary-but-not-sufficient for a race on its own — a breadth
    fan-out over six unrelated entities also has disjoint searches — so that caller pairs it
    with a same-target check.

    :returns: ``"concrete_urls"`` / ``"mixed_search_visit"`` / ``"disjoint_searches"``, or
        ``None`` when no heuristic matches.
    """
    from agent.app.idea_policies.base import IdeaActionType

    if not nodes:
        return None

    if all(
        NodeDetailsExtractor.get_action(n.details) == IdeaActionType.VISIT.value
        and concrete_url(n.details)
        for n in nodes
    ):
        return "concrete_urls"

    actions = {NodeDetailsExtractor.get_action(n.details) for n in nodes}
    if actions == {IdeaActionType.SEARCH.value, IdeaActionType.VISIT.value} and all(
        concrete_url(n.details)
        for n in nodes
        if NodeDetailsExtractor.get_action(n.details) == IdeaActionType.VISIT.value
    ):
        return "mixed_search_visit"

    if all(
        NodeDetailsExtractor.get_action(n.details) == IdeaActionType.SEARCH.value
        for n in nodes
    ):
        queries = set()
        for n in nodes:
            query = NodeDetailsExtractor.get_query(n.details, fallback_title=n.title)
            if isinstance(query, str) and query.strip():
                queries.add(" ".join(query.strip().lower().split()))
        if len(queries) >= 2:
            return "disjoint_searches"

    return None


def defer_unresolved_slot_candidates(
    graph: IdeaDag,
    ready_children: List[str],
    step_index: int,
    logger: logging.Logger,
) -> List[str]:
    """Drop slot-bearing candidates from this step's ready set.

    Defers only one candidate per step if other siblings can proceed. Guarantees
    forward progress by never deferring a candidate twice.

    :param ready_children: Node ids filtered to this step's action-ready set.
    :returns: ready_children with deferred candidates removed.
    """
    from agent.app.idea_policies.dataflow import unresolved_slots

    deferrable: List[tuple] = []
    for candidate_id in ready_children:
        node = graph.get_node(candidate_id)
        if node is None:
            continue
        if node.details.get("__dataflow_deferred_step") is not None:
            continue
        slots = unresolved_slots(node.details, title=node.title)
        if slots:
            deferrable.append((candidate_id, slots))

    if not deferrable or len(deferrable) >= len(ready_children):
        return ready_children

    deferred_ids = {candidate_id for candidate_id, _ in deferrable}
    kept = [cid for cid in ready_children if cid not in deferred_ids]
    for candidate_id, slots in deferrable:
        node = graph.get_node(candidate_id)
        node.details["__dataflow_deferred_step"] = step_index
        logger.info(
            f"[STEP {step_index}] DATAFLOW DEFER: node {candidate_id} has an unresolved "
            f"slot in {slots}; deferring to next step ({len(kept)} sibling(s) proceed)"
        )
    return kept
