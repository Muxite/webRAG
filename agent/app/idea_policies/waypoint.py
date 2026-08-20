"""
Deterministic (LLM-free) extraction of the VALUE a just-completed VISIT leaf's fetched page
carries — for threading into a downstream hop that needs it. Today the engine passes hop N+1
only visit *metadata* ("Visited https://... page='X'. 21066 chars, 340 links"), never the datum
(a name, a place, a number) hop N discovered; this module is the deterministic extractor that
recovers it, sitting alongside — and reusing the building blocks of — the contract-satisfaction
check in ``contract_satisfaction.py``.

A prior audit (``scripts/replay_waypoints.py``, recall pass) replayed 108 stored chain-task
results and found two DISTINCT, LLM-free extraction paths, not one:

* **cue-window** (quantity / date / numeric code facts) — a number within a bounded window of
  the datum's own contract wording (``_QUANTITY_CUES``, widened here with gaps the audit
  surfaced, e.g. "span") recovers the correct value at 100% recall-of-ceiling. Built on
  ``contract_satisfaction._find_number_near`` (refactored out of ``_number_near`` so
  ``evaluate_step_contract``/F33/the contract log stay byte-identical).
* **anchor-text** (entity / place) — NOT recoverable via a body-text regex (h1_text/title is
  3.6% useful; a body-text "lead paragraph" scan is 13.7%). Recoverable via
  ``action_result.link_contexts`` (a page's own outbound hyperlinks — Wikipedia infoboxes
  hyperlink Designer/Engineer/Born-in fields) cross-referenced against a role-cue window in the
  page's own body text, at 80-100% recall depending on hop type.

A SEPARATE precision pass (``scripts/replay_waypoints.py --precision``) then measured what
either extractor does when pointed at a page that is NOT the waypoint's correct source — the
real risk at the production write site, where a leaf's own contract text fires against whatever
page it actually fetched, right or wrong. The result was unambiguous: **a wrong-page fetch does
not make either extractor go silent, it makes it confidently WRONG.**

    cue-window : 87/87  wrong-page trials emitted a value; 87/87  (100%) of those were wrong.
    anchor-text: 27/27  wrong-page trials emitted a value; 25/27  ( 93%) of those were wrong.

(A concrete instance: the cue-window path, run blind against the Brooklyn Bridge's OWN page
while searching for the Cincinnati bridge's 1,057 ft span, confidently returns a different,
equally wrong number — the "span" cue fires on the wrong page just as readily as the right one.)

So neither path may emit a value without a **page-identity guard**: the fetched page's own
identity (h1_text / page_title / url — never its body, which routinely mentions the right
subject even on the WRONG page) must corroborate at least one of the hop's expected-subject
tokens (``contract_satisfaction._subject_tokens``) before either extractor runs at all. When the
contract text carries no subject tokens whatsoever (the common case for a boilerplate leaf goal,
e.g. a compiled-plan leaf whose instruction deliberately withholds the target's name), the guard
FAILS CLOSED — silence, not a value — rather than reusing ``_subject_present``'s "zero tokens is
trivially satisfied" convention, which would defeat the guard on exactly the leaves most at risk
of a wrong-page fetch.

``build_waypoint`` never raises for a shape it does not understand; the worst outcome is
``method="none"`` (no value, but still a diagnosable record of why: no evidence, no cue,
identity guard failed).

``scripts/replay_waypoints.py --validate-product`` re-runs both questions against this actual
module (not a prototype): with the guard, the wrong-page EMISSION rate collapses from ~100% down
to ~14% of wrong-page trials (and most of what still gets through does so because a wrong page
shares a subject token with the right one — e.g. a chain's OWN "creator" bio page corroborating
a "keystone" leaf's widened identity tokens). On the CORRECT page, only ~46% of leaves in the
audit corpus emit a value at all (the guard is deliberately conservative), and of those, ~17.5%
are the ground-truth value — the residual gap is a SEPARATE, disclosed limitation: a keyword
window has no way to prefer "the 1856 declared height" over the SAME page's current infobox
height when both sit near the word "height", or to pick the semantically-right anchor among
several plausible nearby hyperlinks. The identity guard answers "is this the right PAGE"; it
does not answer "is this the right VALUE on that page" — a caller should treat ``value`` as a
hint to corroborate, not a fact to inject unchecked, consistent with the flag defaulting OFF.

A follow-up breakdown of that residual gap (still via ``--validate-product``, now reporting
strict-correct / near-miss / genuinely-wrong and a candidate-count diagnosis) found: near-misses
are ~0% (the ground-truth regexes are already lenient on formatting, so "wrong" really is
wrong, not a strictness artifact), and of the genuinely-wrong cases 94% had 2+ plausible
candidates in the SAME window the extractor used (ambiguity — a real reading-comprehension
failure) vs. 6% where only one candidate existed and it was simply wrong (a targeting bug).
Of the ambiguity/targeting sample, every ``anchor_text`` emission against a leaf whose own
contract text DID carry a datum cue (a quantity/date ask) was wrong — an anchor is never a
legitimate number, so that fallback is now gated off in exactly that case (see the comment at
the call site below); it does not reach leaves whose own goal text gives no signal at all that
a number was being sought (a genuinely underspecified leaf, not a fixable extraction bug).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.candidate_coverage import _fuzzy_contains as fuzzy_contains
from agent.app.idea_policies.contract_satisfaction import (
    StepContract,
    _find_number_near,
    _subject_tokens,
    _visit_evidence,
    derive_step_contract,
)

# Mirrors contract_satisfaction._NUMBER_WINDOW: how far either side of a cue word counts as
# "near" it. Kept as a local literal rather than importing the private name a second time —
# the two windows are conceptually the same knob, but this module owns its own extraction pass.
_WINDOW = 200

# Role/creation words that plausibly sit next to the hyperlinked VALUE of an entity or place
# waypoint (a designer, an engineer, a birthplace, a namesake work) — the generic vocabulary a
# production extractor falls back on for a hop type ``_QUANTITY_CUES`` was never meant to cover.
# Not curated per-waypoint the way a leaf's own contract text is; broad on purpose, since the
# page-identity guard (not this list) is what keeps a mismatch from turning into a wrong answer.
_ENTITY_ROLE_CUES: Tuple[str, ...] = (
    "designer", "designed by", "chief engineer", "engineer", "engineered by",
    "architect", "architected", "founder", "founded by", "discovered by",
    "invented by", "author", "written by", "poet", "born", "birthplace",
    "born in", "creator", "created by", "built by", "constructed by",
    "developed by", "completed", "record-breaking", "launch vehicle",
    "rocket", "aqueduct", "steamship", "location", "located",
)


@dataclass
class Waypoint:
    """The datum a completed VISIT leaf's page carried, for a downstream hop to consume."""

    hop_goal: str
    value: Optional[str]
    value_kind: Optional[str]
    evidence: str
    source_url: str
    source_title: str
    node_id: str
    step: int
    method: str  # "cue_window" | "anchor_text" | "none"
    # The URL the extracted value was hyperlinked to, when the value came from a hyperlink
    # anchor (``method="anchor_text"``). This is the one payload that crosses a node boundary
    # with perfect fidelity -- the tool returned it, no reading comprehension involved -- so it
    # is what the resolved-value channel threads into a downstream hop's ``url`` slot. Always
    # None for ``cue_window`` (a number is not a link) and for ``method="none"``.
    value_url: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _TextNode:
    """Minimal node-like shim so ``derive_step_contract`` (written for live graph nodes exposing
    ``.details``/``.title``) can be re-run against a bare fallback-goal string."""

    __slots__ = ("details", "title")

    def __init__(self, text: str):
        self.details = {DetailKey.GOAL.value: text}
        self.title = text


def _widen_subject_tokens_with_parent_goal(node: Any, contract: StepContract) -> StepContract:
    """Widen a leaf's SUBJECT TOKENS (never its datum cues) with its PARENT's goal text when
    the leaf's own text names no subject at all — the generic-template gap the audit flagged:
    many leaves carry boilerplate goal text ("Visit a source page to substantiate: ...") that
    identifies no page. ``DetailKey.PARENT_GOAL`` is already denormalized onto every expanded
    child, so this needs no graph lookup.

    Deliberately does NOT widen ``datums`` the same way: a leaf's PARENT_GOAL is often the
    ENTIRE multi-hop mandate (a chain task's parent step routinely describes every hop, not
    just this one), so pulling a datum cue from it risks borrowing a DIFFERENT hop's cue word
    ("span"/"length" from hop 3's ask) onto a leaf that is really an entity/place hop with no
    number to find at all — confirmed against the corpus: this exact widening, tried first,
    made a "creator" (entity) leaf whose own text asks for no number instead go hunting for one
    and return an unrelated page coordinate. A wrong subject token is comparatively low-risk (a
    shared proper noun is the SAME risk ``_subject_present`` already accepts elsewhere); a wrong
    datum cue is not, so only the subject side widens.
    """
    if contract.subject_tokens:
        return contract
    details = getattr(node, "details", {}) or {}
    parent_goal = details.get(DetailKey.PARENT_GOAL.value)
    if not isinstance(parent_goal, str) or not parent_goal.strip():
        return contract
    parent_tokens = _subject_tokens(parent_goal)
    if not parent_tokens:
        return contract
    return StepContract(
        text=contract.text, source=contract.source,
        datums=contract.datums, subject_tokens=parent_tokens,
    )


def _page_identity(result: Dict[str, Any]) -> str:
    """h1/title/url only — deliberately NOT body text. The precision measurement this guard
    exists because of shows a wrong page's BODY routinely mentions the right subject too (the
    Brooklyn Bridge's own page is full of "Roebling"/"Cincinnati"); only the page's own identity
    is a genuine signal that it IS the expected source, not just that it TALKS ABOUT it.
    """
    parts: List[str] = []
    for key in ("page_title", "h1_text", "title", "url", "source_url"):
        val = result.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    return " ".join(parts)


def _page_identity_ok(contract: StepContract, result: Dict[str, Any]) -> bool:
    """The page-identity guard: at least one of the hop's expected-subject tokens must appear
    in the fetched page's OWN identity (not its body). Deliberately stricter than
    ``contract_satisfaction._subject_present`` (which treats zero tokens as trivially
    satisfied) — here, zero tokens means the contract carried no identity signal at all, so the
    guard fails CLOSED instead of vacuously passing on exactly the leaves most at risk (a
    boilerplate leaf goal that names no subject can't corroborate ANY page, including the wrong
    one — see the module docstring's precision numbers).
    """
    tokens = contract.subject_tokens
    if not tokens:
        return False
    identity = _page_identity(result)
    if not identity:
        return False
    return any(fuzzy_contains(tok, identity) for tok in tokens)


def _extract_cue_window(contract: StepContract, evidence_low: str) -> Tuple[Optional[str], Optional[str]]:
    """(value, value_kind) via the cue-window path, or (None, None). ``value_kind`` is the
    canonical datum name (``_QUANTITY_CUES``'s own vocabulary — "height", "span", "year", ...).
    """
    for canonical, variants in contract.datums:
        match = _find_number_near(evidence_low, variants)
        if match:
            return match.group(0), canonical
    return None, None


def _extract_anchor_text(
    evidence_low: str, link_contexts: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str]]:
    """The first known hyperlink anchor text (``action_result.link_contexts``) that lands
    within ``_WINDOW`` chars of a role-cue word's first occurrence in the page's own body —
    cross-referencing body-text proximity against confirmed hyperlink structure, which is what
    makes this stronger than a bare body-text regex (the audit's h1/title and lead-paragraph
    methods, both close to useless alone).

    Picks the CLOSEST anchor occurrence to the cue, not the longest anywhere in the window: a
    corpus-driven fix (the first version here picked the longest anchor text that appeared
    anywhere in the window, which routinely grabbed an unrelated but longer nearby link -- an
    engine's name, a relative's name, a footnote title -- over the actual adjacent value).
    Infobox-style label/value pairs put the value immediately next to its label, so proximity
    is the stronger signal; this is still a heuristic, not a guarantee, on top of the guard.

    :returns: ``(anchor_text, url)`` -- the anchor's own target URL rides along, since that is
        the structurally-reliable half of the pair (the tool returned it) and is what the
        resolved-value channel hands to a downstream hop. ``(None, None)`` when nothing lands.
    """
    if not link_contexts:
        return None, None
    anchors = [
        (v.strip(), k)
        for k, v in link_contexts.items()
        if isinstance(v, str) and isinstance(k, str) and len(v.strip()) >= 3
    ]
    if not anchors:
        return None, None
    for cue in _ENTITY_ROLE_CUES:
        match = re.search(re.escape(cue), evidence_low)
        if not match:
            continue
        start = max(0, match.start() - _WINDOW)
        end = min(len(evidence_low), match.end() + _WINDOW)
        window_text = evidence_low[start:end]
        cue_center = (match.start() + match.end()) / 2
        best_value: Optional[str] = None
        best_url: Optional[str] = None
        best_distance: Optional[float] = None
        for anchor_text, url in anchors:
            idx = window_text.find(anchor_text.lower())
            if idx < 0:
                continue
            anchor_center = start + idx + len(anchor_text) / 2
            distance = abs(anchor_center - cue_center)
            if best_distance is None or distance < best_distance:
                best_distance, best_value, best_url = distance, anchor_text, url
        if best_value is not None:
            return best_value, best_url
    return None, None


def _snippet(evidence: str, evidence_low: str, value: str) -> str:
    idx = evidence_low.find(value.lower())
    if idx < 0:
        return evidence[:160].strip()
    return evidence[max(0, idx - 80): idx + len(value) + 80].strip()


def build_waypoint(
    node: Any, result: Dict[str, Any], action: Optional[str], node_id: str, step: int,
) -> Optional["Waypoint"]:
    """Deterministic value-extraction for a just-completed leaf.

    Returns ``None`` for anything but a successful VISIT (SEARCH/THINK/SAVE/MERGE/... retrieve
    no page-grounded evidence to extract from — the same reasoning ``evaluate_step_contract``
    already applies to a search leaf's snippets: a substantiation mandate cannot be satisfied,
    let alone sourced, by text the model itself authored or engine-supplied search snippets).

    Every VISIT gets a ``Waypoint`` record, even when nothing was extracted (``method="none"``)
    — so a failure (no evidence, no cue, identity guard rejected the page) is diagnosable
    offline, the same corpus ``scripts/replay_waypoints.py`` already replays. Never raises;
    the caller (``idea_engine.py``) additionally wraps this in a try/except as defense in depth,
    but every branch here is a plain dict/string lookup.
    """
    if action != IdeaActionType.VISIT.value or not isinstance(result, dict):
        return None

    own_contract = derive_step_contract(node)
    hop_goal = own_contract.text
    source_url = str(result.get("url") or result.get("source_url") or "")
    source_title = str(
        result.get("page_title") or result.get("h1_text") or result.get("title") or ""
    )

    def _none() -> Waypoint:
        return Waypoint(
            hop_goal=hop_goal, value=None, value_kind=None, evidence="",
            source_url=source_url, source_title=source_title,
            node_id=node_id, step=step, method="none",
        )

    evidence = _visit_evidence(result)
    if not evidence.strip():
        return _none()

    contract = _widen_subject_tokens_with_parent_goal(node, own_contract)
    if not _page_identity_ok(contract, result):
        return _none()

    evidence_low = evidence.lower()

    value, value_kind = _extract_cue_window(own_contract, evidence_low)
    method = "cue_window" if value is not None else None
    value_url: Optional[str] = None  # a cue-window value is a number, never a link

    # anchor-text is an entity/place mechanism: a hyperlink anchor can never legitimately be
    # the answer to a quantity/date ask, so it only runs when the leaf's own contract shows NO
    # sign of asking for a measurable datum at all (`own_contract.datums` empty). Corpus-driven:
    # every anchor-text emission observed against a leaf whose own contract DID carry a datum
    # cue (`_QUANTITY_CUES` matched something in its own goal/expect text) was wrong by
    # construction -- a coordinate, a citation title, a nav breadcrumb, never a real number.
    # `cue_window` itself is tried first regardless (a leaf can carry a datum cue yet still miss
    # the actual number in evidence; that failure stays silent rather than falling through).
    if value is None and not own_contract.datums:
        link_contexts = result.get("link_contexts")
        anchor_value, anchor_url = _extract_anchor_text(
            evidence_low, link_contexts if isinstance(link_contexts, dict) else {}
        )
        if anchor_value is not None:
            value, value_kind, method = anchor_value, "anchor", "anchor_text"
            value_url = anchor_url

    if value is None or method is None:
        return _none()

    return Waypoint(
        hop_goal=hop_goal, value=value, value_kind=value_kind,
        evidence=_snippet(evidence, evidence_low, value),
        source_url=source_url, source_title=source_title,
        node_id=node_id, step=step, method=method, value_url=value_url,
    )
