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
from typing import TYPE_CHECKING, List, Protocol

from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.action_constants import NodeDetailsExtractor
from agent.app.idea_policies.data_contracts import URLS_FROM_SEARCH

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
    ) -> None: ...


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


def clean_extracted_url(url: str) -> str:
    """Strip trailing punctuation while preserving balanced parens (Wikipedia URLs)."""
    strip_chars = ".,;:!?"
    url = url.rstrip(strip_chars)
    while url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    url = url.rstrip(strip_chars)
    return url


class MandateUrlInjectionHook:
    """If the mandate text contains explicit URLs and no visit child covers
    one, inject a visit node for each missing URL."""

    def apply(
        self,
        graph: "IdeaDag",
        node_id: str,
        step_index: int,
        mandate: str,
        logger: logging.Logger,
    ) -> None:
        node = graph.get_node(node_id)
        if not node or not mandate:
            return

        raw_urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', mandate)
        mandate_urls = [clean_extracted_url(u) for u in raw_urls]
        mandate_urls = [u for u in mandate_urls if u]
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


_VISIT_PHRASES = (
    "must visit",
    "you must visit",
    "must visit the",
    "required to visit",
    "need to visit",
    "should visit",
    "visit the url",
    "visit the page",
)
_SEARCH_PHRASES = (
    "must search",
    "you must search",
    "search for",
    "find and visit",
)


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
    ) -> None:
        node = graph.get_node(node_id)
        if not node or not mandate:
            return

        mandate_lower = mandate.lower()
        requires_visit = any(p in mandate_lower for p in _VISIT_PHRASES)
        requires_search = any(p in mandate_lower for p in _SEARCH_PHRASES)
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


WEB_POST_EXPANSION_HOOKS: tuple[PostExpansionHook, ...] = (
    MandateUrlInjectionHook(),
    MandatePhraseEnforcementHook(),
)


def default_post_expansion_hooks() -> List[PostExpansionHook]:
    """The in-tree default: web-research mandate enforcement."""
    return list(WEB_POST_EXPANSION_HOOKS)
