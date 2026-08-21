"""
Grounding evaluation.

When a mandate requires substantiated evidence (navigate by following links, or "base the
answer on pages you open / do not guess"), the agent should not be allowed to finalize from
parametric memory. ``evaluate_grounding`` inspects the graph's actual successful visits and
reports whether the mandate's substantiation requirements are met. The engine uses this for
a SOFT gate: inject the missing follow-through and re-plan up to a cap, then finalize-but-
flag if still ungrounded (never hard-block).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Set

from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.mandate_requirements import (
    MandateRequirements,
    parse_mandate_requirements,
)

_logger = logging.getLogger(__name__)


def _norm(url: str) -> str:
    s = str(url or "").strip().lower()
    for pre in ("https://", "http://"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    return s.split("#", 1)[0].rstrip("/")


@dataclass
class GroundingResult:
    """Outcome of a grounding check."""

    grounded: bool
    missing: List[str] = field(default_factory=list)
    reason: str = ""
    distinct_visits: int = 0
    followed_links: int = 0


def _successful_visit_urls(graph) -> Set[str]:
    """Normalized URLs of every successfully visited page in the graph."""
    urls: Set[str] = set()
    for n in graph.iter_depth_first():
        ar = (getattr(n, "details", {}) or {}).get(DetailKey.ACTION_RESULT.value) or {}
        if not (isinstance(ar, dict) and ar.get("action") == IdeaActionType.VISIT.value and ar.get("success")):
            continue
        for u in (ar.get("urls_visited") or ([ar.get("url")] if ar.get("url") else [])):
            if u:
                urls.add(_norm(u))
    return urls


def _page_identity_verified_visit_urls(graph) -> Set[str]:
    """Like ``_successful_visit_urls``, but a visit only counts when its page identity
    corroborates the visiting leaf's own subject tokens (F37, opt-in).

    Reuses ``waypoint.page_identity_ok`` — the same h1/title/url-only guard
    ``build_waypoint`` already uses to reject a wrong-page fetch — imported inline to match
    the existing cross-policy-module convention (see ``merge.py``'s ``_datum_verified``,
    which imports ``contract_satisfaction.evaluate_step_contract`` the same way) rather than
    a module-level import, avoiding a load-order dependency between the two policy modules.

    A leaf with ZERO extractable subject tokens (a boilerplate goal that names no subject —
    the common case, not the exception) is excluded from the relevance check rather than
    failed: ``page_identity_ok`` itself fails CLOSED on zero tokens (the right call for
    ``build_waypoint``, which is about to emit a specific fact and would rather stay silent
    than emit one from an uncorroborated page), but grounding is a coarser "did the agent do
    real substantiating work" gate — failing closed here would newly reject the majority of
    already-passing runs whose leaves simply don't carry subject-token text, which is a
    noise problem, not evidence the page was wrong. So a token-less leaf's visit still counts
    toward the requirement (degrade safely: treat "can't tell" as "don't penalize"), while a
    leaf that DOES carry subject tokens but visited a page that fails to corroborate them is
    excluded, which is the actual off-topic-visit case this flag exists to catch.
    """
    from agent.app.idea_policies.contract_satisfaction import derive_step_contract
    from agent.app.idea_policies.waypoint import page_identity_ok

    urls: Set[str] = set()
    for n in graph.iter_depth_first():
        ar = (getattr(n, "details", {}) or {}).get(DetailKey.ACTION_RESULT.value) or {}
        if not (isinstance(ar, dict) and ar.get("action") == IdeaActionType.VISIT.value and ar.get("success")):
            continue
        try:
            contract = derive_step_contract(n)
        except Exception:  # noqa: BLE001 — the gate must never crash finalize
            contract = None
        if contract is not None and contract.subject_tokens:
            try:
                relevant = page_identity_ok(contract, ar)
            except Exception:  # noqa: BLE001 — the gate must never crash finalize
                relevant = True
            if not relevant:
                continue
        for u in (ar.get("urls_visited") or ([ar.get("url")] if ar.get("url") else [])):
            if u:
                urls.add(_norm(u))
    return urls


def graph_planned_retrieval(graph) -> bool:
    """True when the PLAN contains at least one search/visit node, whatever its status.

    Retrieval intent in the plan is the strongest available "this is a research task"
    signal that does not depend on the mandate using one of the magic phrases above.
    """
    retrieval = {IdeaActionType.SEARCH.value, IdeaActionType.VISIT.value}
    for n in graph.iter_depth_first():
        if (getattr(n, "details", {}) or {}).get(DetailKey.ACTION.value) in retrieval:
            return True
    return False


def requires_grounded_answer(mandate: str, graph=None) -> bool:
    """Must this task's answer come from pages the agent actually opened?

    True for an explicit substantiation mandate (navigate / "do not guess"), for one that
    names a URL or asks the agent to search/visit, and — when a graph is supplied — for any
    run whose own plan contains a search/visit node. False for work that legitimately needs
    no retrieval (summarize this text, transform this input), so the finalize grounding gate
    can never refuse a non-research answer.
    """
    try:
        req = parse_mandate_requirements(mandate)
    except Exception as e:  # noqa: BLE001 — the gate must never crash finalize
        # Fail open (grounding not required) so a parser bug can never refuse a whole run,
        # but say so: silently returning False here made a real parse bug in this SAFETY
        # gate invisible in production. Behavior unchanged — this is visibility only.
        _logger.warning(
            f"[GROUNDING] requires_grounded_answer parse failed, failing open "
            f"(grounding NOT required): {type(e).__name__}: {e} | mandate={str(mandate)[:200]!r}"
        )
        return False
    if req.needs_substantiation or req.must_visit or req.must_search or req.named_urls:
        return True
    return graph is not None and graph_planned_retrieval(graph)


def evaluate_grounding(
    graph, requirements: MandateRequirements, *, require_page_identity: bool = False,
) -> GroundingResult:
    """Decide whether the mandate's substantiation requirements are satisfied.

    - Navigation mandates: require that the agent actually followed a link — i.e. it
      visited a page that was NOT one of the explicitly-named start URLs, or it visited
      at least two distinct pages (real traversal, not a single start-page read).
    - General grounding mandates ("do not guess"): require at least one successful visit.

    ``require_page_identity`` (F37, opt-in, default OFF): when set, a visit only counts
    toward either requirement above if it ALSO passes the page-identity relevance guard
    (see ``_page_identity_verified_visit_urls``) — without it, this function is pure set
    arithmetic over visited URLs with no relevance/content check of any kind, so two
    completely off-topic pages trivially satisfy either requirement. Off keeps today's
    exact behavior byte-identical.
    """
    visited = (
        _page_identity_verified_visit_urls(graph)
        if require_page_identity
        else _successful_visit_urls(graph)
    )
    named = {_norm(u) for u in (requirements.named_urls or [])}
    followed = visited - named
    missing: List[str] = []

    if requirements.navigation:
        if not (len(followed) >= 1 or len(visited) >= 2):
            missing.append("followed-link page (only the start page was opened)")

    if requirements.grounding and not requirements.navigation:
        if len(visited) == 0:
            missing.append("at least one visited source page")

    grounded = not missing
    reason = (
        "grounded: real page evidence present"
        if grounded
        else "ungrounded: " + "; ".join(missing)
    )
    return GroundingResult(
        grounded=grounded,
        missing=missing,
        reason=reason,
        distinct_visits=len(visited),
        followed_links=len(followed),
    )
