"""Sequential reordering / state-dependency helpers from :class:`IdeaDagEngine`.

Behavior-preserving helpers that (a) reorder a selected child so data-producing
work (search / URL-bearing visit) runs before data-consuming work
(think / save / merge, and URL-less visits that need a sibling search), and
(b) detect implicit state dependencies among sibling candidates so the engine
falls back to sequential execution. State is passed in explicitly so the engine
remains the sole stateful orchestrator.
"""

from __future__ import annotations

import logging
from typing import List, Optional

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
