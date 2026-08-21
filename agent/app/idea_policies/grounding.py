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
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Set

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


# --------------------------------------------------------------------------------------
# Numeric-token provenance (B1)
#
# ``evaluate_grounding`` above counts PAGES; it never looks at what an answer actually
# claims. The 2026-08-21/22 strong-agent trace put the missing half plainly: a careful
# researcher refuses to elect an answer from a search snippet and only trusts a figure it
# can point to in the raw text of a page it opened. The mirror-image weak-model failure was
# observed live in the same session -- a real merge completion wrote "all three routes
# confirm 575 meters", a number present in NO fetched page anywhere in the run, with
# ``goal_achieved: true`` and zero cited URLs.
#
# ``goal_evidence_provenance`` (merge.py) already answers "did fetched content address the
# goal", but its downgrade is gated OFF because the wording of a correct paraphrase need not
# appear anywhere. The check below sidesteps that entirely by asking about the answer's
# NUMBERS rather than its wording: "1,310 metres", "1310 m" and "1 310 metres" are the same
# fact under one normalization, and a correct answer essentially never spells its keystone
# figure as a number that appears nowhere in what the run fetched.
# --------------------------------------------------------------------------------------

#: Units that mark a number as a MEASUREMENT rather than a year, a rank, a count or a
#: computed ratio. Deliberately short: the unit's only job is deciding which of an answer's
#: numbers are checkable claims (verification itself compares values, never units -- see
#: :func:`verify_numeric_claims`), so widening it buys nothing and risks sweeping in derived
#: quantities that legitimately appear on no page.
_UNIT_PATTERN = (
    r"kilometres|kilometers|kilometre|kilometer|metres|meters|metre|meter"
    r"|miles|mile|feet|foot|km|ft|mi|m"
)

#: Characters real pages use as thousands separators: comma, plain space, NBSP, narrow NBSP,
#: thin space, and the Swiss/typographic apostrophes.
_THOUSANDS_SEPARATORS = "[,\u00a0\u202f\u2009 '\u2019]"

#: Either a group-separated number (``1,310`` / ``1 310`` / ``1'310``) or a plain one. The
#: separated form is tried first so ``1,310`` is never read as a bare ``1``. The lookbehind
#: keeps the tail of a longer number or a dotted version string from matching on its own.
_NUMBER_PATTERN = (
    rf"\d{{1,3}}(?:{_THOUSANDS_SEPARATORS}\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?"
)

_CLAIM_RE = re.compile(
    rf"(?<![\w.])({_NUMBER_PATTERN})\s*({_UNIT_PATTERN})\b", re.IGNORECASE
)
_NUMBER_RE = re.compile(rf"(?<![\w.])(?:{_NUMBER_PATTERN})")
_SEPARATOR_RE = re.compile(_THOUSANDS_SEPARATORS)


@dataclass(frozen=True)
class NumericClaim:
    """One number-with-unit an answer asserts, plus its normalized value."""

    raw: str
    value: str
    unit: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.raw


@dataclass
class NumericProvenanceResult:
    """Which of an answer's numeric claims were found in fetched page text, and which not.

    Deliberately inspectable rather than a bare bool: the caller logs WHICH figure failed,
    which is the whole diagnostic value of the check.
    """

    verified: List[NumericClaim] = field(default_factory=list)
    unverified: List[NumericClaim] = field(default_factory=list)

    @property
    def checked(self) -> bool:
        """Did the answer assert any measurement at all? False => this check is a no-op."""
        return bool(self.verified or self.unverified)

    @property
    def unsupported(self) -> bool:
        """The actionable shape: measurements were asserted and NOT ONE of them was found.

        Stricter than "some claim failed" on purpose. A page stating a figure in other units,
        a rounded restatement, and a value the model computed from two fetched numbers all
        produce a legitimately-unverifiable claim; an answer where EVERY figure is absent
        from everything the run fetched is the hallucination shape instead.
        """
        return bool(self.unverified) and not self.verified

    def unverified_values(self) -> List[str]:
        return [c.raw for c in self.unverified]


def _normalize_number(raw: str) -> str:
    """``"1,310"`` / ``"1 310"`` / ``"1310"`` / ``"1310.0"`` all collapse to ``"1310"``."""
    cleaned = _SEPARATOR_RE.sub("", str(raw or ""))
    try:
        value = float(cleaned)
    except ValueError:
        return ""
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def extract_numeric_claims(text: Any) -> List[NumericClaim]:
    """Every number-with-unit token in ``text``, normalized and de-duplicated.

    A number with no unit (a year, a rank, a bare count, a ratio) is not a claim this check
    can adjudicate and is skipped -- which is also what keeps whole task families (computed
    ratios, ordinal answers) out of scope rather than falsely flagged.
    """
    claims: List[NumericClaim] = []
    seen = set()
    for match in _CLAIM_RE.finditer(str(text or "")):
        value = _normalize_number(match.group(1))
        if not value:
            continue
        unit = match.group(2).lower()
        if (value, unit) in seen:
            continue
        seen.add((value, unit))
        claims.append(NumericClaim(raw=match.group(0).strip(), value=value, unit=unit))
    return claims


def _evidence_values(texts: Sequence[Any]) -> Set[str]:
    """Normalized values of every number appearing anywhere in ``texts``, unit or not.

    Values are compared unit-blind on purpose: a page states the span in a table cell, an
    infobox row or a sentence, and demanding that the unit spelling match too would fail a
    correct answer for the page's typography. The unit requirement lives on the CLAIM side,
    where it selects what is worth checking.
    """
    values: Set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in _NUMBER_RE.finditer(str(text)):
            value = _normalize_number(match.group(0))
            if value:
                values.add(value)
    return values


def verify_numeric_claims(text: Any, evidence_texts: Sequence[Any]) -> NumericProvenanceResult:
    """Split ``text``'s numeric claims into those present in ``evidence_texts`` and those not."""
    claims = extract_numeric_claims(text)
    if not claims:
        return NumericProvenanceResult()
    values = _evidence_values(evidence_texts)
    result = NumericProvenanceResult()
    for claim in claims:
        (result.verified if claim.value in values else result.unverified).append(claim)
    return result


def visited_page_texts(graph) -> List[str]:
    """Raw text of every successfully visited page in the graph.

    ``content_full`` is the whole cleaned page as ``VisitLeafAction`` stored it; ``content``
    is the truncated window the prompts see, kept as a fallback for results that carry only
    it. Search results are excluded by construction -- a SEARCH node's action is not
    ``visit``, so a snippet naming a figure the run never fetched contributes nothing here,
    which is exactly the distinction the strong-agent trace insisted on.
    """
    from agent.app.idea_policies.action_constants import ActionResultKey

    texts: List[str] = []
    for n in graph.iter_depth_first():
        ar = (getattr(n, "details", {}) or {}).get(DetailKey.ACTION_RESULT.value) or {}
        if not (
            isinstance(ar, dict)
            and ar.get("action") == IdeaActionType.VISIT.value
            and ar.get("success")
        ):
            continue
        text = (
            ar.get(ActionResultKey.CONTENT_FULL.value)
            or ar.get(ActionResultKey.CONTENT.value)
            or ar.get(ActionResultKey.CONTENT_WITH_LINKS.value)
        )
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def answer_numeric_provenance(
    graph, answer_text: Any, merged_results: Optional[Sequence[Any]] = None
) -> NumericProvenanceResult:
    """Does every measurement this answer asserts appear in something the run FETCHED?

    Evidence is the union of two sources, both snippet-free:

    - the raw text of every successful visit in the whole graph (not only this node's
      children): a figure copied faithfully from a page an ancestor or a sibling branch
      opened is properly sourced, and blaming a merge for where in the graph the fetch
      happened would be a pure false positive;
    - the direct (non-search-hit) text of the merged child payloads, when supplied -- so a
      tool/extract payload that produced the figure counts too. ``merge._direct_text``
      already drops the SEARCH ``results`` list and the echoed request keys, which is the
      same snippet-vs-fetched distinction ``goal_evidence_provenance`` draws; it is reused
      rather than re-derived. Merge syntheses are excluded: a merge hands up prose, and
      crediting it would let a nested merge's own invented figure certify the parent's (the
      nested merge ran this same check over ITS children, where the pages actually are).
    """
    evidence: List[Any] = list(visited_page_texts(graph))
    if merged_results:
        from agent.app.idea_policies.merge import _direct_text

        for item in merged_results:
            if not isinstance(item, dict) or item.get("is_merge"):
                continue
            result = item.get("result")
            if result:
                evidence.append(_direct_text(result))
    return verify_numeric_claims(answer_text, evidence)
