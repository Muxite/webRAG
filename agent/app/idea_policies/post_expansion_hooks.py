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
from agent.app.idea_policies.action_constants import ActionResultKey, NodeDetailsExtractor
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
                    # Injected with no URL at all (only a generic `link_idea`), so its `url`
                    # is exactly the field the resolved-value channel exists to fill.
                    "slot": DetailKey.URL.value,
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


class GroundingEvidenceEnforcementHook:
    """Enforce the plain-grounding contract ``evaluate_grounding`` already documents but
    that neither of the two hooks above cover: a mandate phrased as "do not guess" /
    "based on the page you open" (``requirements.grounding``) without an explicit
    "must visit" phrase (``MandatePhraseEnforcementHook``) or a navigation target
    (``MandateNavigationHook``) can otherwise be planned as search-only children, run
    to completion, and finalize with zero visited pages — the exact "0 visits, grounded:
    False" failure `evaluate_grounding` is meant to catch but that nothing upstream
    prevents. Once a search has completed, inject a visit seeded from its results;
    idempotent (skips once any visit child exists or a visit has already succeeded).
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
        # The other two hooks already handle these trigger phrasings; don't double-inject.
        if not req.grounding or req.must_visit or req.navigation:
            return

        if _has_successful_visit(graph):
            return
        for child_id in node.children:
            child = graph.get_node(child_id)
            if child and NodeDetailsExtractor.get_action(child.details) == IdeaActionType.VISIT.value:
                return  # a visit is already planned (or ran and failed); let it play out

        search_node_id = None
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            if (
                NodeDetailsExtractor.get_action(child.details) == IdeaActionType.SEARCH.value
                and child.details.get(DetailKey.ACTION_RESULT.value)
            ):
                search_node_id = child_id
                break
        if not search_node_id:
            return  # nothing to visit yet; wait for a search to complete

        visit_node = graph.add_child(
            parent_id=node_id,
            title="Visit a source page for grounded evidence",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                DetailKey.IS_LEAF.value: True,
                DetailKey.JUSTIFICATION.value: (
                    "Mandate requires grounded ('do not guess') evidence but no page has "
                    "been visited yet; visiting a search result to substantiate the answer."
                ),
                DetailKey.GOAL.value: f"Visit a source page to substantiate: {mandate[:100]}",
                "link_idea": "URL from search results or mandate",
                "link_count": 1,
                DetailKey.REQUIRES_DATA.value: {
                    "type": URLS_FROM_SEARCH.name,
                    "source_node_id": search_node_id,
                    # Same as the enforcement hook above: no URL is authored here, so the
                    # `url` slot is left for the resolved-value channel to fill at dispatch.
                    "slot": DetailKey.URL.value,
                },
            },
        )
        logger.info(
            f"[STEP {step_index}] ENFORCE: Injected grounding-evidence visit node "
            f"{visit_node.node_id} (search source {search_node_id})"
        )
        _record_enforce(telemetry, step_index, visit_node.node_id,
                        "visit (grounding evidence)",
                        "mandate requires 'do not guess' evidence but 0 visits exist")


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
    GroundingEvidenceEnforcementHook(),
)


def default_post_expansion_hooks() -> List[PostExpansionHook]:
    """The in-tree default: web-research mandate enforcement."""
    return list(WEB_POST_EXPANSION_HOOKS)


#: Default ceiling on how many candidate-visit pairs one remediation pass may mint. Bounds the
#: node count a pathological enumeration could otherwise create.
_COVERAGE_INJECTION_LIMIT = 8


def _search_node_for(graph: "IdeaDag", parent_id: str, candidate: str) -> Optional[str]:
    """Find a COMPLETED search whose query mentions ``candidate``.

    Reusing one avoids paying for a second search for a candidate the run already searched --
    which is the common case by the time coverage remediation fires, since the widened budget
    tends to be spent on searches.

    :returns: The search node id, or ``None``.
    """
    needle = candidate.lower()
    parent = graph.get_node(parent_id)
    if parent is None:
        return None
    for node in graph.iter_depth_first():
        if NodeDetailsExtractor.get_action(node.details) != IdeaActionType.SEARCH.value:
            continue
        if not node.details.get(DetailKey.ACTION_RESULT.value):
            continue
        query = str(node.details.get(DetailKey.QUERY.value) or "").lower()
        if needle and needle in query:
            return node.node_id
    return None


def _already_targeted(graph: "IdeaDag", candidate: str) -> bool:
    """Whether some visit node in the graph is already aimed at ``candidate``.

    Remediation can fire more than once in a run (budget extension, then grounding replan), and
    without this the second pass mints a duplicate set.
    """
    needle = candidate.lower()
    for node in graph.iter_depth_first():
        if NodeDetailsExtractor.get_action(node.details) != IdeaActionType.VISIT.value:
            continue
        if needle in str(node.title or "").lower():
            return True
    return False


def inject_coverage_visits(
    graph: "IdeaDag",
    node_id: str,
    step_index: int,
    mandate: str,
    logger: logging.Logger,
    telemetry: Optional[Any] = None,
    max_injections: int = _COVERAGE_INJECTION_LIMIT,
) -> int:
    """Mint a VISIT for every enumerated candidate the coverage gate reports missing.

    The gate counts only successful VISITS, but nothing it could trigger created one: the
    visit-injecting hooks above each early-return unless the mandate carries a ``must visit``
    phrase or navigation targets, which an ordinary "for each of the following, find X" mandate
    does not. So the gate detected the gap and the engine answered it with more searching --
    46 searches and 1 visit on the n=24 A/B, 55 searches and 2 visits once the structural caps
    were lifted.

    Deterministic on purpose: this is the same reasoning as the gate itself, which ignores what
    the model *says* about its own progress. A weak model asked to "go visit the ones you
    missed" tends to search again instead.

    :param graph: Graph being remediated.
    :param node_id: Parent to attach the new work to (normally the root).
    :param step_index: Current step, for logging.
    :param mandate: The run's mandate, parsed for its enumeration.
    :param logger: Engine logger.
    :param telemetry: Optional telemetry session for the enforcement record.
    :param max_injections: Ceiling on candidates handled in one pass.
    :returns: How many visits were injected.
    """
    parent = graph.get_node(node_id)
    if parent is None or not mandate:
        return 0

    from agent.app.idea_policies.candidate_coverage import evaluate_candidate_coverage

    try:
        cov = evaluate_candidate_coverage(graph, mandate)
    except Exception as exc:  # noqa: BLE001 - remediation must never crash a run
        logger.warning(f"[STEP {step_index}] COVERAGE INJECT: gate failed: {exc}")
        return 0
    if cov.satisfied or not cov.missing:
        return 0

    injected = 0
    for candidate in cov.missing:
        if injected >= max_injections:
            break
        if _already_targeted(graph, candidate):
            continue

        source_id = _search_node_for(graph, node_id, candidate)
        if source_id is None:
            search_node = graph.add_child(
                parent_id=node_id,
                title=f"Search {candidate}",
                details={
                    DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                    DetailKey.QUERY.value: f"{candidate} {mandate[:80]}",
                    DetailKey.IS_LEAF.value: True,
                    DetailKey.JUSTIFICATION.value: (
                        f"Coverage gate: '{candidate}' was never researched."
                    ),
                    DetailKey.GOAL.value: f"Find a source page about {candidate}",
                },
            )
            source_id = search_node.node_id
            _record_enforce(telemetry, step_index, source_id,
                            "search", f"coverage: {candidate} unsearched")

        visit_node = graph.add_child(
            parent_id=node_id,
            title=f"Visit a page about {candidate}",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                DetailKey.IS_LEAF.value: True,
                DetailKey.JUSTIFICATION.value: (
                    f"Coverage gate: no visited page is about '{candidate}', so the answer "
                    "for it would be ungrounded."
                ),
                DetailKey.GOAL.value: f"Visit a page about {candidate}",
                "link_idea": f"URL for {candidate} from search results",
                "link_count": 1,
                DetailKey.REQUIRES_DATA.value: {
                    "type": URLS_FROM_SEARCH.name,
                    "source_node_id": source_id,
                },
            },
        )
        _record_enforce(telemetry, step_index, visit_node.node_id,
                        "visit", f"coverage: {candidate} unvisited")
        injected += 1

    if injected:
        logger.warning(
            f"[STEP {step_index}] COVERAGE INJECT: {injected} visit(s) for uncovered "
            f"candidate(s): {', '.join(cov.missing[:max_injections])}"
        )
    return injected


#: Marker written on the empty search that was remediated, so one search is remediated once.
_EMPTY_SEARCH_MARKER = "_empty_search_followup"
#: Run-scoped counter, kept on the root. The trigger here is per completed search, so a
#: per-call cap would bound nothing: a branch searching into a dead pool would earn fresh
#: budget on every attempt. The budget is spent for the whole run instead.
_EMPTY_SEARCH_BUDGET_KEY = "_empty_search_followups"
#: Back-reference from a corrective search to the empty one that caused it (telemetry/debug).
_CORRECTIVE_FOR_SEARCH = "_corrective_for_search"
#: How many leading tokens of the original query survive broadening. A query that returned
#: nothing is usually over-qualified ("X first ascent year official record"), so the cheapest
#: deterministic broadening is to keep its head and drop the qualifiers.
_BROADENED_QUERY_TOKENS = 4


def _search_result_urls(node) -> List[str]:
    """Every http(s) URL a completed SEARCH node's result offers, in result order.

    Deliberately the same key set and validity rule as
    ``VisitLeafAction._extract_urls_from_parent_search_results``: that consumer defines what a
    search actually handed the run, and a second opinion here would call a search "empty" that
    the visit path can use (or the reverse).
    """
    details = node.details or {}
    result = details.get(DetailKey.ACTION_RESULT.value)
    if not isinstance(result, dict):
        return []
    action = result.get(ActionResultKey.ACTION.value) or result.get("action")
    if action != IdeaActionType.SEARCH.value:
        return []
    results = result.get(ActionResultKey.RESULTS.value) or result.get("results") or []
    if not isinstance(results, list):
        return []
    urls: List[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        candidate = (
            item.get("url") or item.get("link") or item.get("href")
            or item.get("source") or item.get("page_url")
        )
        if not candidate:
            continue
        candidate = str(candidate).strip()
        if candidate.startswith(("http://", "https://")) and candidate not in urls:
            urls.append(candidate)
    return urls


def _claimed_urls(graph: "IdeaDag") -> set:
    """Normalized URLs that some visit node already opened or is already aimed at."""
    claimed = set()
    for node in graph.iter_depth_first():
        details = node.details or {}
        result = details.get(DetailKey.ACTION_RESULT.value)
        if isinstance(result, dict) and result.get(ActionResultKey.ACTION.value) == IdeaActionType.VISIT.value:
            url = result.get(ActionResultKey.URL.value) or ""
            if url:
                claimed.add(str(url).strip().rstrip("/"))
        if NodeDetailsExtractor.get_action(details) == IdeaActionType.VISIT.value:
            for key in (DetailKey.URL.value, "optional_url"):
                url = details.get(key) or ""
                if url:
                    claimed.add(str(url).strip().rstrip("/"))
    return claimed


def search_yielded_no_visit(graph: "IdeaDag", node) -> bool:
    """Whether a COMPLETED search left the run with no page worth opening.

    True for both shapes of the same gap: the results list is empty (or unparseable), and the
    results list is non-empty but every URL in it is already visited or already claimed by
    another visit node.
    """
    urls = _search_result_urls(node)
    if not urls:
        return True
    claimed = _claimed_urls(graph)
    return all(u.rstrip("/") in claimed for u in urls)


def _broadened_query(
    query: str,
    unresolved_entities: Optional[List[str]] = None,
    existing_queries: Optional[List[str]] = None,
) -> Optional[str]:
    """A deterministic alternate query for a search that came back with nothing.

    Two sources, in order: the ledger's unresolved entity that this query was chasing (searching
    the bare entity name drops whatever qualifier starved the result set), then the query's own
    leading tokens. Returns ``None`` when neither yields something genuinely different from the
    original or from a query the run already ran -- re-issuing an identical search is how a
    remediation loop starts.
    """
    normalized = " ".join(str(query or "").split())
    if not normalized:
        return None
    lowered = normalized.lower()

    candidates: List[str] = []
    for entity in unresolved_entities or ():
        name = " ".join(str(entity or "").split())
        if name and name.lower() in lowered:
            candidates.append(name)
    head = " ".join(re.sub(r"[\"'“”]", " ", normalized).split()[:_BROADENED_QUERY_TOKENS])
    candidates.append(head)

    seen = {" ".join(str(q or "").split()).lower() for q in (existing_queries or ())}
    for candidate in candidates:
        cand = " ".join(candidate.split())
        if not cand or cand.lower() == lowered or cand.lower() in seen:
            continue
        return cand
    return None


def inject_empty_search_followup(
    graph: "IdeaDag",
    node_id: str,
    step_index: int,
    logger: logging.Logger,
    telemetry: Optional[Any] = None,
    unresolved_entities: Optional[List[str]] = None,
    max_injections: int = _COVERAGE_INJECTION_LIMIT,
) -> int:
    """Remediate ONE completed search that yielded no page worth visiting.

    Same shape of remediation as :func:`inject_coverage_visits` -- deterministic gap detection
    followed by a targeted search/visit pair -- against the other half of the gap. That function
    fires on "this enumerated candidate has no visit ANYWHERE" and injects off the root; this one
    fires on "THIS search handed the run nothing to open" and injects as a sibling of the dead
    search, so the pooled sibling URL resolution the visit path already does can see the new
    results.

    The visit is wired with ``REQUIRES_DATA`` onto the corrective search, so the pair expresses
    exactly the contract the flag is named for: a search must yield a visit.

    Idempotent per search node (marker in its details) and bounded per RUN (counter on the root),
    so repeated dead searches on one branch cannot spawn unbounded remediation.

    :param graph: Graph being remediated.
    :param node_id: The completed search node to inspect.
    :param step_index: Current step, for logging.
    :param logger: Engine logger.
    :param telemetry: Optional telemetry session for the enforcement record.
    :param unresolved_entities: Task-ledger entities still without evidence, used to pick the
        alternate query. Optional: the mechanism works from the dead search alone.
    :param max_injections: Run-scoped ceiling on corrective pairs.
    :returns: 1 if a corrective pair was injected, else 0.
    """
    node = graph.get_node(node_id)
    if node is None:
        return 0
    details = node.details or {}
    if NodeDetailsExtractor.get_action(details) != IdeaActionType.SEARCH.value:
        return 0
    if not details.get(DetailKey.ACTION_RESULT.value):
        return 0  # never ran; nothing to call empty
    if details.get(_EMPTY_SEARCH_MARKER):
        return 0
    if node.children:
        return 0  # a re-expansion already gave this search a real follow-up
    if not search_yielded_no_visit(graph, node):
        return 0

    root = graph.get_node(graph.root_id())
    used = int((root.details.get(_EMPTY_SEARCH_BUDGET_KEY) or 0) if root is not None else 0)
    if used >= max_injections:
        logger.info(
            f"[STEP {step_index}] EMPTY SEARCH: budget spent ({used}/{max_injections}); "
            f"no follow-up for {node_id[:8]}"
        )
        return 0

    existing = [
        str(n.details.get(DetailKey.QUERY.value) or "")
        for n in graph.iter_depth_first()
        if NodeDetailsExtractor.get_action(n.details) == IdeaActionType.SEARCH.value
    ]
    query = str(details.get(DetailKey.QUERY.value) or node.title or "")
    broadened = _broadened_query(query, unresolved_entities, existing)
    if not broadened:
        return 0

    parent_id = node.parent_id or graph.root_id()
    search_node = graph.add_child(
        parent_id=parent_id,
        title=f"Search {broadened}",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.QUERY.value: broadened,
            DetailKey.IS_LEAF.value: True,
            DetailKey.JUSTIFICATION.value: (
                f"The search '{query[:80]}' returned no page worth visiting; retrying with a "
                "broader query."
            ),
            DetailKey.GOAL.value: f"Find a source page for: {broadened}",
            _CORRECTIVE_FOR_SEARCH: node_id,
        },
    )
    visit_node = graph.add_child(
        parent_id=parent_id,
        title=f"Visit a page about {broadened}",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.JUSTIFICATION.value: (
                "A search must yield a visited page; the previous search yielded none."
            ),
            DetailKey.GOAL.value: f"Visit a page about {broadened}",
            "link_idea": f"URL for {broadened} from search results",
            "link_count": 1,
            DetailKey.REQUIRES_DATA.value: {
                "type": URLS_FROM_SEARCH.name,
                "source_node_id": search_node.node_id,
            },
        },
    )

    node.details[_EMPTY_SEARCH_MARKER] = search_node.node_id
    if root is not None:
        root.details[_EMPTY_SEARCH_BUDGET_KEY] = used + 1
    logger.warning(
        f"[STEP {step_index}] EMPTY SEARCH: '{query[:60]}' yielded no visitable URL; injected "
        f"search {search_node.node_id[:8]} ('{broadened}') + dependent visit "
        f"{visit_node.node_id[:8]} ({used + 1}/{max_injections})"
    )
    _record_enforce(telemetry, step_index, search_node.node_id,
                    f"search {broadened[:60]}", "empty search: no visitable URL")
    _record_enforce(telemetry, step_index, visit_node.node_id,
                    "visit", "empty search: a search must yield a visit")
    return 1
