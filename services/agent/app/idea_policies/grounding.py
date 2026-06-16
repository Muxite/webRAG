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
from agent.app.idea_policies.mandate_requirements import MandateRequirements


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
