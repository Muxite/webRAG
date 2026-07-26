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

from dataclasses import dataclass, field
from typing import List, Set

from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.mandate_requirements import (
    MandateRequirements,
    parse_mandate_requirements,
)


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
    except Exception:  # noqa: BLE001 — the gate must never crash finalize
        return False
    if req.needs_substantiation or req.must_visit or req.must_search or req.named_urls:
        return True
    return graph is not None and graph_planned_retrieval(graph)


def evaluate_grounding(graph, requirements: MandateRequirements) -> GroundingResult:
    """Decide whether the mandate's substantiation requirements are satisfied.

    - Navigation mandates: require that the agent actually followed a link — i.e. it
      visited a page that was NOT one of the explicitly-named start URLs, or it visited
      at least two distinct pages (real traversal, not a single start-page read).
    - General grounding mandates ("do not guess"): require at least one successful visit.
    """
    visited = _successful_visit_urls(graph)
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
