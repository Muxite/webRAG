"""
Post-expansion hooks.

A `PostExpansionHook` runs after the LLM expansion policy produces children
for a parent node. Hooks can inspect those children and inject additional
nodes the planner might have missed — for example, mandate-required visit or
search actions.

Hooks are task-specific policy and should NOT live in the engine. The four
built-in mandate-enforcement helpers used by the web-research workflow are
implemented here as a pair of hooks, bundled into `WEB_POST_EXPANSION_HOOKS`
for convenience.

Custom action packs ship their own hooks; pass them to `IdeaDagEngine` via
the `post_expansion_hooks` constructor parameter.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional, Protocol

from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.action_constants import NodeDetailsExtractor
from agent.app.idea_policies.data_contracts import URLS_FROM_SEARCH
from agent.app.idea_policies.mandate_requirements import (
    MandateRequirements,
    clean_extracted_url,
    parse_mandate_requirements,
)

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag


class PostExpansionHook(Protocol):
    """Runs after children are added to a node during expansion.

    Hooks MUST be idempotent: an expansion may re-fire on the same node and
    each hook should detect already-injected children before adding new ones.
    """

    def apply(
        self,
        graph: "IdeaDag",
        node_id: str,
        step_index: int,
        mandate: str,
        logger: logging.Logger,
        telemetry: Optional[Any] = None,
    ) -> None: ...


def _record_enforce(telemetry: Optional[Any], step_index: int, node_id: str,
                    what: str, reason: str) -> None:
    """Record an enforcement injection on the decision trace (best-effort)."""
    if telemetry is None:
        return
    rec = getattr(telemetry, "record_decision", None)
    if callable(rec):
        rec(stage="enforce", node_id=node_id, chosen=what, rationale=reason,
            metadata={"step": step_index})


def extract_mandate(graph: "IdeaDag", node_id: str) -> str:
    """Resolve the mandate text starting from a node, falling back to root."""
    node = graph.get_node(node_id)
    if not node:
        return ""
    mandate = node.details.get("mandate") or ""
    if mandate:
        return mandate
    root = graph.get_node(graph.root_id())
    if not root:
        return ""
    if root.node_id == node_id:
        return root.title or ""
    return root.details.get("mandate") or root.title or ""


def _best_url_for_title(title: str, urls: List[str]) -> Optional[str]:
    """Best mandate URL for a URL-less visit child, matched by title overlap.

    Single mandate URL -> that URL. Otherwise score each URL by how many slug
    tokens of its last path segment appear in the title.
    """
    if not urls:
        return None
    if len(urls) == 1:
        return urls[0]
    title_l = (title or "").lower()
    best_url, best_score = None, 0
    for u in urls:
        slug = u.rstrip("/").rsplit("/", 1)[-1]
        tokens = [t for t in re.split(r"[_\-%]+", slug.lower()) if len(t) > 2]
        score = sum(1 for t in tokens if t in title_l)
        if score > best_score:
            best_url, best_score = u, score
    return best_url if best_score > 0 else None


class MandateUrlInjectionHook:
    """If the mandate text contains explicit URLs and no visit child covers
    one, inject a visit node for each missing URL. Also repairs a URL-less visit
    child in place (the planner sometimes names the page but drops the URL)."""

    def apply(
        self,
        graph: "IdeaDag",
        node_id: str,
        step_index: int,
        mandate: str,
        logger: logging.Logger,
        telemetry: Optional[Any] = None,
    ) -> None:
        node = graph.get_node(node_id)
        if not node or not mandate:
            return

        mandate_urls = parse_mandate_requirements(mandate).named_urls
        if not mandate_urls:
            return

        covered_urls = set()
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            if NodeDetailsExtractor.get_action(child.details) != IdeaActionType.VISIT.value:
                continue
            child_url = child.details.get(DetailKey.URL.value) or child.details.get("optional_url") or ""
            if not child_url:
                # Self-heal a URL-less visit child from a mandate URL named in its title,
                # instead of leaving a zombie node that can only fail.
                repaired = _best_url_for_title(child.title or "", mandate_urls)
                if repaired:
                    child.details[DetailKey.URL.value] = repaired
                    child.details["optional_url"] = repaired
                    child_url = repaired
                    logger.info(
                        f"[STEP {step_index}] ENFORCE: repaired URL-less visit child "
                        f"{child_id} -> {repaired[:60]}"
                    )
            if child_url:
                covered_urls.add(child_url.rstrip("/"))

        missing_urls = [u for u in mandate_urls if u.rstrip("/") not in covered_urls]
        if not missing_urls:
            return

        for url in missing_urls:
            visit_node = graph.add_child(
                parent_id=node_id,
                title=f"Visit {url[:60]}",
                details={
                    DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                    DetailKey.URL.value: url,
                    "optional_url": url,
                    DetailKey.IS_LEAF.value: True,
                    DetailKey.JUSTIFICATION.value: "Mandate requires visiting this URL",
                    DetailKey.GOAL.value: f"Visit and extract information from {url}",
                },
            )
            logger.info(
                f"[STEP {step_index}] ENFORCE: Injected visit node {visit_node.node_id} "
                f"for mandate URL {url[:60]}"
            )
            _record_enforce(telemetry, step_index, visit_node.node_id,
                            f"visit {url[:60]}", "mandate names this URL")


class MandatePhraseEnforcementHook:
    """If the mandate uses explicit phrases like 'must visit' or 'must search'
    and no child action satisfies that requirement, inject the missing action.
    A visit injected after a search is wired with REQUIRES_DATA pointing at
    the search node, so it waits for results."""

    def apply(
        self,
        graph: "IdeaDag",
        node_id: str,
        step_index: int,
        mandate: str,
        logger: logging.Logger,
        telemetry: Optional[Any] = None,
    ) -> None:
        node = graph.get_node(node_id)
        if not node or not mandate:
            return

        req = parse_mandate_requirements(mandate)
        requires_visit = req.must_visit
        requires_search = req.must_search
        if not requires_visit and not requires_search:
            return

        has_visit = False
        has_search = False
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            child_action = NodeDetailsExtractor.get_action(child.details)
            if child_action == IdeaActionType.VISIT.value:
                has_visit = True
            elif child_action == IdeaActionType.SEARCH.value:
                has_search = True

        search_node_id = None
        if requires_search and not has_search:
            logger.warning(
                f"[STEP {step_index}] ENFORCE: Mandate requires search but no search node created. "
                f"Injecting search node."
            )
            search_node = graph.add_child(
                parent_id=node_id,
                title="Search for required information",
                details={
                    DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                    DetailKey.QUERY.value: mandate[:200],
                    DetailKey.IS_LEAF.value: True,
                    DetailKey.JUSTIFICATION.value: "Mandate explicitly requires search action",
                    DetailKey.GOAL.value: f"Search as required by mandate: {mandate[:100]}",
                },
            )
            search_node_id = search_node.node_id
            has_search = True
            logger.info(
                f"[STEP {step_index}] ENFORCE: Injected search node {search_node_id} "
                f"for mandate requirement"
            )
            _record_enforce(telemetry, step_index, search_node_id,
                            "search", "mandate phrase requires a search")
        elif has_search:
            for child_id in node.children:
                child = graph.get_node(child_id)
                if child and NodeDetailsExtractor.get_action(child.details) == IdeaActionType.SEARCH.value:
                    search_node_id = child_id
                    break

        if requires_visit and not has_visit:
            logger.warning(
                f"[STEP {step_index}] ENFORCE: Mandate requires visit but no visit node created. "
                f"Injecting visit node."
            )
            visit_node = graph.add_child(
                parent_id=node_id,
                title="Visit required URL",
                details={
                    DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                    DetailKey.IS_LEAF.value: True,
                    DetailKey.JUSTIFICATION.value: "Mandate explicitly requires visit action - will extract URL from search results or use link_idea",
                    DetailKey.GOAL.value: f"Visit URL as required by mandate: {mandate[:100]}",
                    "link_idea": "URL from search results or mandate",
                    "link_count": 1,
                },
            )
            if search_node_id:
                visit_node.details[DetailKey.REQUIRES_DATA.value] = {
                    "type": URLS_FROM_SEARCH.name,
                    "source_node_id": search_node_id,
                }
                logger.info(
                    f"[STEP {step_index}] ENFORCE: Visit node {visit_node.node_id} "
                    f"depends on search node {search_node_id}"
                )
            logger.info(
                f"[STEP {step_index}] ENFORCE: Injected visit node {visit_node.node_id} "
                f"for mandate requirement"
            )
            _record_enforce(telemetry, step_index, visit_node.node_id,
                            "visit", "mandate phrase requires a visit")


class MandateNavigationHook:
    """Enforce *link-following* through-lines.

    When the mandate asks the agent to navigate by following hyperlinks toward a
    described destination, a weak/confident planner may answer from memory instead of
    opening the next page. This hook detects that a source page has already been visited
    (so its outgoing links exist) but a described ``nav_target`` has NOT yet been visited,
    and injects a follow-up visit node that uses the visited page's links to reach the
    target (``link_idea`` = the target phrase, ``link_count`` >= 1, plus an LLM link pick).

    Reuses ``VisitLeafAction``'s existing semantic link-following — no new action.
    Idempotent: it will not add a second follow-up for the same target.
    """

    def apply(
        self,
        graph: "IdeaDag",
        node_id: str,
        step_index: int,
        mandate: str,
        logger: logging.Logger,
        telemetry: Optional[Any] = None,
    ) -> None:
        node = graph.get_node(node_id)
        if not node or not mandate:
            return
        req = parse_mandate_requirements(mandate)
        if not req.navigation or not req.nav_targets:
            return

        # Has at least one page already been visited (links available to follow)?
        visited_any = _has_successful_visit(graph)
        if not visited_any:
            return  # nothing to follow yet; wait for the start page to load

        # Which targets already have a visit/link-follow node addressing them?
        addressed = _addressed_targets(graph)
        for target in req.nav_targets:
            tnorm = target.strip().lower()
            if any(tnorm in a or a in tnorm for a in addressed):
                continue
            visit_node = graph.add_child(
                parent_id=node_id,
                title=f"Follow link to {target[:50]}",
                details={
                    DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                    DetailKey.IS_LEAF.value: True,
                    "link_idea": target,
                    "link_count": int(_nav_link_count()),
                    DetailKey.JUSTIFICATION.value: (
                        f"Mandate requires navigating to '{target}' by following a link "
                        f"from an already-visited page (grounded follow-through)."
                    ),
                    DetailKey.GOAL.value: f"Follow a hyperlink to reach: {target}",
                },
            )
            logger.info(
                f"[STEP {step_index}] ENFORCE: Injected link-follow visit "
                f"{visit_node.node_id} toward '{target[:50]}'"
            )
            _record_enforce(telemetry, step_index, visit_node.node_id,
                            f"follow-link -> {target[:50]}",
                            "mandate navigation target not yet visited")


def _nav_link_count() -> int:
    """How many links the follow-up visit should open. Default 1: open the single best
    candidate (the visit action surfaces a wide candidate pool and the LLM picks the one
    matching the descriptive link_idea, e.g. 'rocket that launched the mission' -> Saturn V)."""
    import os
    try:
        return max(1, int(os.environ.get("IDEA_NAV_LINK_COUNT", "1")))
    except (TypeError, ValueError):
        return 1


def _has_successful_visit(graph: "IdeaDag") -> bool:
    for n in graph.iter_depth_first():
        ar = (n.details or {}).get(DetailKey.ACTION_RESULT.value) or {}
        if isinstance(ar, dict) and ar.get("action") == IdeaActionType.VISIT.value and ar.get("success"):
            return True
    return False


def _addressed_targets(graph: "IdeaDag") -> List[str]:
    """Lowercased link_idea / URL-slug text of existing visit nodes (to stay idempotent)."""
    out: List[str] = []
    for n in graph.iter_depth_first():
        if NodeDetailsExtractor.get_action(n.details) != IdeaActionType.VISIT.value:
            continue
        li = (n.details.get("link_idea") or "").strip().lower()
        if li:
            out.append(li)
        url = (n.details.get(DetailKey.URL.value) or n.details.get("optional_url") or "")
        if url:
            out.append(url.rsplit("/", 1)[-1].replace("_", " ").lower())
    return out


WEB_POST_EXPANSION_HOOKS: tuple[PostExpansionHook, ...] = (
    MandateUrlInjectionHook(),
    MandatePhraseEnforcementHook(),
    MandateNavigationHook(),
)


def default_post_expansion_hooks() -> List[PostExpansionHook]:
    """The in-tree default: web-research mandate enforcement."""
    return list(WEB_POST_EXPANSION_HOOKS)
