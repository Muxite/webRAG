"""
Contract satisfaction for a completed leaf (the re-expansion control signal).

The confidence->action loop originally re-expanded a leaf whose *step-confidence judge*
score fell below a threshold. The barrage census showed that number is anti-calibrated as
a quality estimator: it sees only "the mandate + this leaf's page content" and asks "does
this look on-track", so a coherent WRONG page scores high and a correct-but-partial page
scores low (conf>=0.6 runs scored 0.475 vs 0.687 for conf<0.6 on nano/good_adaptive). It
"worked" only because low scores happened to fire the independently-useful re-expansion.

This module supplies the replacement control signal: a DETERMINISTIC, LLM-free check of
whether the step delivered what its contract required. Three rules, in order:

* **readiness**: the step produced no usable payload at all (a visit with no content, a
  search with no results). Reuses the data_contracts predicate for search
  (URLS_FROM_SEARCH.is_ready); a visit's own default contract is urls_from_visit
  (LINKS), which answers "what does this node provide downstream", not "did this
  step deliver its payload", so the visit readiness test is content-presence here.

* **datum**: the contract text asks for a measurable datum (a depth, a year, a count)
  and the retrieved evidence does not contain a number next to that datum's own wording.

* **subject**: the contract names distinctive subject tokens and the retrieved evidence
  mentions too few of them (the classic wrong-page grounding failure).

Fails toward SILENCE, never toward a false accusation: when no checkable requirement can
be derived from the leaf's contract text the result is applicable=False and the caller
is expected to fall back to whatever it did before. Only a positive absence (missing
payload, missing datum, or missing subject) reports satisfied=False.

The "contract text" is the leaf's explicit ``expect`` detail when the expect-contract
lever wrote one, else the leaf's own goal/title. The mandate is consulted only through
``mandate_requirements`` (a substantiation mandate cannot be satisfied by a search leaf's
snippets. It needs an opened page).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.candidate_coverage import _fuzzy_contains as fuzzy_contains
from agent.app.idea_policies.data_contracts import URLS_FROM_SEARCH
from agent.app.idea_policies.mandate_requirements import parse_mandate_requirements


# How much retrieved text to scan. Bounded so the check stays cheap on a large page.
_MAX_EVIDENCE_CHARS = 60000
# A datum counts as PRESENT when a number appears within this many characters of the
# datum's own wording ("maximum depth ... 511 m").
_NUMBER_WINDOW = 200
# Proper-noun subject tokens are ranked longest-first; only the top-N are demanded, so a
# long sub-goal cannot pile up requirements a legitimately-correct page would fail.
_MAX_SUBJECT_TOKENS = 3

# Words that carry no identity: plan/tool boilerplate, generic nouns, and the imperative
# verbs an LLM opens a sub-goal title with. Stripping these leaves the entity tokens.
_NOISE_TOKENS = frozenset({
    "the", "and", "for", "from", "with", "that", "this", "its", "into", "about",
    "page", "pages", "wikipedia", "wiki", "article", "url", "urls", "link", "links",
    "site", "website", "web", "search", "result", "results", "source", "sources",
    "information", "data", "value", "values", "exact", "official", "content",
    "section", "infobox", "table", "text", "field", "entry",
    "find", "determine", "identify", "report", "state", "give", "open", "read",
    "visit", "confirm", "verify", "lookup", "look", "extract", "record", "note",
    "obtain", "retrieve", "gather", "collect", "resolve", "check", "locate",
    "then", "next", "first", "step", "task", "subtask", "goal", "answer", "question",
    "using", "use", "only", "must", "should", "need", "needed", "required", "require",
    "provide", "return", "list", "name", "named", "cite", "citation", "according",
    # quantifiers and units qualify a datum, they never identify the subject
    "maximum", "minimum", "max", "min", "average", "mean", "median", "total",
    "approximate", "specific", "single", "figure", "measurement", "metric",
    "metres", "meters", "metre", "meter", "feet", "foot", "kilometres", "kilometers",
    "kilometre", "kilometer", "inches", "yards", "acres", "hectares", "years",
})


# (cue in the contract text, canonical datum, wording to look for in the evidence).
# An empty variants tuple means "any number in the evidence satisfies this datum" (used
# for cues whose datum has no stable page wording: year, count, price).
_QUANTITY_CUES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("how deep", "depth", ("depth", "deep")),
    ("deepest", "depth", ("depth", "deep")),
    ("depth", "depth", ("depth", "deep")),
    ("how tall", "height", ("height", "tall", "elevation")),
    ("how high", "height", ("height", "high", "elevation")),
    ("tallest", "height", ("height", "tall", "elevation")),
    ("height", "height", ("height", "tall", "elevation")),
    ("elevation", "elevation", ("elevation", "altitude", "above sea level")),
    ("altitude", "elevation", ("elevation", "altitude", "above sea level")),
    ("how long", "length", ("length", "long")),
    ("longest", "length", ("length", "long")),
    ("length", "length", ("length", "long")),
    ("main span", "span", ("span", "main span")),
    ("span", "span", ("span", "main span")),
    ("how wide", "width", ("width", "wide")),
    ("width", "width", ("width", "wide")),
    ("surface area", "area", ("area", "km2", "km²", "sq mi", "square")),
    ("area", "area", ("area", "km2", "km²", "sq mi", "square")),
    ("population", "population", ("population", "inhabitants", "residents")),
    ("how far", "distance", ("distance", "km", "miles", "mi")),
    ("distance", "distance", ("distance", "km", "miles", "mi")),
    ("temperature", "temperature", ("temperature", "°c", "°f", "celsius", "fahrenheit")),
    ("volume", "volume", ("volume", "capacity")),
    ("capacity", "volume", ("volume", "capacity")),
    ("weight", "weight", ("weight", "mass", "kg", "tonnes", "tons")),
    ("mass", "weight", ("weight", "mass", "kg", "tonnes", "tons")),
    ("percentage", "percentage", ("percent", "%")),
    ("percent", "percentage", ("percent", "%")),
    # Numeric datums with no stable page wording -> any number satisfies.
    ("how many", "count", ()),
    ("number of", "count", ()),
    ("how much", "amount", ()),
    ("what year", "year", ()),
    ("which year", "year", ()),
    ("in what year", "year", ()),
    ("founding year", "year", ()),
    ("founded", "year", ()),
    ("established", "year", ()),
    ("built", "year", ()),
    ("born", "year", ()),
    ("died", "year", ()),
    ("when was", "year", ()),
    ("when did", "year", ()),
    ("year", "year", ()),
    ("date", "date", ()),
    ("price", "price", ()),
    ("cost", "price", ()),
)

# Cue words are not identity tokens (a "depth" sub-goal is not about the word "depth").
_QUANTITY_WORDS = frozenset(
    word
    for cue, _canon, variants in _QUANTITY_CUES
    for word in (cue.split() + [v for v in variants if v.isalpha()])
)

_CASED_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’-]{2,}")
_DIGIT_RE = re.compile(r"\d")


@dataclass
class StepContract:
    """What a completed leaf was supposed to deliver."""

    text: str = ""
    source: str = "goal"            # "expect" (explicit contract) or "goal"
    datums: List[Tuple[str, Tuple[str, ...]]] = field(default_factory=list)
    subject_tokens: List[str] = field(default_factory=list)

    @property
    def checkable(self) -> bool:
        """True when at least one requirement can actually be tested against evidence."""
        return bool(self.datums or self.subject_tokens)


@dataclass
class ContractSatisfaction:
    """Verdict for one completed leaf."""

    applicable: bool
    satisfied: bool
    missing: List[str] = field(default_factory=list)
    reason: str = ""
    #: True only when the contract text asked for a measurable DATUM and the evidence
    #: actually carried a number beside that datum's wording. A satisfied verdict without
    #: it rests on subject tokens alone: it proves "the right PAGE was opened", never "the
    #: ask was answered". Callers that use a satisfied contract to VETO another signal need
    #: to tell those two apart (see F35 in DAG_FORMATION_REVIEW.md).
    datum_verified: bool = False


def _contract_text(node: Any) -> Tuple[str, str]:
    """The leaf's contract text and where it came from (``expect`` beats the goal)."""
    details = getattr(node, "details", {}) or {}
    expect = details.get(DetailKey.EXPECT.value)
    if isinstance(expect, str) and expect.strip():
        return expect.strip(), "expect"
    for key in (DetailKey.GOAL.value, DetailKey.ORIGINAL_GOAL.value):
        val = details.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip(), "goal"
    return str(getattr(node, "title", "") or "").strip(), "goal"


def _required_datums(text: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """Canonical measurable datums the contract text asks for, de-duplicated."""
    low = text.lower()
    out: List[Tuple[str, Tuple[str, ...]]] = []
    seen = set()
    for cue, canonical, variants in _QUANTITY_CUES:
        if cue not in low or canonical in seen:
            continue
        seen.add(canonical)
        out.append((canonical, variants))
    return out


def _subject_tokens(text: str) -> List[str]:
    """The entity tokens that identify what the leaf was supposed to look at.

    Only CAPITALIZED words qualify (minus the plan/unit noise above): a proper noun is the
    part of a sub-goal that says *which* page, while a lowercase verb/adjective ("empties",
    "current") is phrasing the page is free to spell differently. Ranked longest-first and
    capped, because long tokens carry identity ("quesnel") and short head noun ("lake") is
    shared with every wrong page. Empty when the text names no proper noun (subject
    requirement then simply does not apply).
    """
    tokens: List[str] = []
    seen = set()
    for raw in _CASED_TOKEN_RE.findall(text):
        if not raw[:1].isupper():
            continue
        tok = raw.lower()
        if tok in _NOISE_TOKENS or tok in _QUANTITY_WORDS or tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
    tokens.sort(key=len, reverse=True)
    return tokens[:_MAX_SUBJECT_TOKENS]


def derive_step_contract(node: Any) -> StepContract:
    """Parse a completed leaf's contract (explicit ``expect``, else its goal/title)."""
    text, source = _contract_text(node)
    return StepContract(
        text=text,
        source=source,
        datums=_required_datums(text),
        subject_tokens=_subject_tokens(text),
    )


def _visit_body(result: Dict[str, Any]) -> str:
    """The page text a visit brought back (empty means the step delivered no payload)."""
    body = result.get("content") or result.get("content_full") or ""
    return body[:_MAX_EVIDENCE_CHARS] if isinstance(body, str) else ""


def _visit_evidence(result: Dict[str, Any]) -> str:
    """Page identity (title/url) plus its body (what the leaf actually read)."""
    parts: List[str] = []
    for key in ("page_title", "h1_text", "title", "url", "source_url"):
        val = result.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    body = _visit_body(result)
    if body:
        parts.append(body)
    return "\n".join(parts)


def _search_evidence(result: Dict[str, Any]) -> str:
    parts: List[str] = []
    for item in (result.get("results") or [])[:10]:
        if isinstance(item, dict):
            for key in ("title", "url", "snippet", "description"):
                val = item.get(key)
                if isinstance(val, str) and val:
                    parts.append(val)
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)[:_MAX_EVIDENCE_CHARS]


#: The full number token (not just a single digit) a match against _NUMBER_RE captures.
#: Presence-equivalent to _DIGIT_RE (both start matching at the same first-digit position,
#: since _NUMBER_RE begins with \d), so swapping which one _find_number_near searches with
#: changes nothing about whether a match is found. Only what a found match's .group()
#: contains differs: the whole number instead of one digit character. That equivalence keeps
#: _number_near (and evaluate_step_contract, F33, the contract log) byte-identical across
#: this refactor (see waypoint_test.py's regression-lock test).
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _find_number_near(evidence_low: str, variants: Sequence[str]) -> Optional["re.Match[str]"]:
    """The number nearest a datum's own wording, within _NUMBER_WINDOW chars either side.

    "Nearest" means globally nearest-by-character-distance to whichever cue OCCURRENCE it sits
    beside, evaluated across every occurrence of every variant (not the leftmost number in the
    FIRST occurrence's window). An earlier version did the latter despite this docstring claiming
    "nearest" (corpus-driven fix 2026-08-16): re.search on a window returns the leftmost match
    in read order. This happens to be whatever unrelated number (date, unrelated stat) sits
    earliest in the 200-char slice, even when the TRUE value sits only a few characters past
    the cue (e.g. ship's "length ... 692 ft" lost to "27 august 1862" because the date read
    left-to-right first). Mirrors waypoint._extract_anchor_text, which picks the anchor nearest
    the cue rather than first one found in-window.

    An empty variants means the datum has no stable page wording (year, count), so the
    first number anywhere in the evidence counts (no cue occurrence to measure distance from).
    Returns the re.Match itself (a value extractor wants the digits, not just whether one
    exists). _number_near below is the historical bool-only view all existing callers keep using
    unchanged; picking a DIFFERENT (nearer) match among several candidates never changes whether
    a match exists, so its presence-only semantics are unaffected by this.
    """
    if not variants:
        return _NUMBER_RE.search(evidence_low)
    best_match: Optional["re.Match[str]"] = None
    best_distance: Optional[float] = None
    for variant in variants:
        for cue_match in re.finditer(re.escape(variant), evidence_low):
            win_start = max(0, cue_match.start() - _NUMBER_WINDOW)
            win_end = min(len(evidence_low), cue_match.end() + _NUMBER_WINDOW)
            cue_center = (cue_match.start() + cue_match.end()) / 2
            for num_match in _NUMBER_RE.finditer(evidence_low, win_start, win_end):
                num_center = (num_match.start() + num_match.end()) / 2
                distance = abs(num_center - cue_center)
                if best_distance is None or distance < best_distance:
                    best_distance, best_match = distance, num_match
    return best_match


def _number_near(evidence_low: str, variants: Sequence[str]) -> bool:
    """A number within _NUMBER_WINDOW chars of the datum's wording (see _find_number_near)."""
    return _find_number_near(evidence_low, variants) is not None


def _subject_present(evidence: str, tokens: Sequence[str]) -> bool:
    """Every demanded identity token appears in the retrieved evidence.

    ALL rather than a majority, because a majority lets the shared head noun carry the
    match ("Lake Baikal" would satisfy a "Quesnel Lake" contract on the strength of the
    word lake). The error this direction produces is one extra (bounded, empirically
    productive) re-expansion; the opposite error would falsely CREDIT a wrong page and
    silence the trigger, which is the failure F33 exists to remove. Matching reuses
    candidate_coverage's established "these two strings mean the same thing" bar, so
    singular/plural and minor spelling variance still match.
    """
    if not tokens:
        return True
    return all(fuzzy_contains(tok, evidence) for tok in tokens)


def evaluate_step_contract(node: Any, mandate: str = "") -> ContractSatisfaction:
    """Did this completed leaf deliver what its contract required?

    applicable=False means "no verdict" (no checkable requirement could be derived, either
    an action that retrieves nothing or a contract text with neither measurable datum nor
    identity token). Callers treat that as "fall back to the previous signal".
    """
    details = getattr(node, "details", {}) or {}
    result = details.get(DetailKey.ACTION_RESULT.value)
    if not isinstance(result, dict):
        return ContractSatisfaction(applicable=False, satisfied=True)

    action = details.get(DetailKey.ACTION.value)
    if action == IdeaActionType.VISIT.value:
        if not _visit_body(result).strip():
            return ContractSatisfaction(
                applicable=True, satisfied=False,
                missing=["page content"],
                reason="visit returned no page content",
            )
        evidence = _visit_evidence(result)
    elif action == IdeaActionType.SEARCH.value:
        if not URLS_FROM_SEARCH.is_ready(result, node):
            return ContractSatisfaction(
                applicable=True, satisfied=False,
                missing=["search results"],
                reason="search returned no results",
            )
        evidence = _search_evidence(result)
    else:
        # THINK/SAVE/MERGE and custom actions retrieve no external evidence; a datum
        # "found" in model-authored text is exactly the parametric answer this signal
        # exists to distrust, so we stay silent instead of crediting it.
        return ContractSatisfaction(applicable=False, satisfied=True)

    contract = derive_step_contract(node)
    if not contract.checkable:
        return ContractSatisfaction(applicable=False, satisfied=True)

    # A search leaf's contract is "produce URLs to open", not "carry the datum": engine
    # snippets naming the subject are exactly the short-circuit a substantiation mandate
    # forbids. Readiness above is the whole test for search.
    if action == IdeaActionType.SEARCH.value:
        requirements = parse_mandate_requirements(mandate) if mandate else None
        if requirements is not None and requirements.needs_substantiation:
            return ContractSatisfaction(applicable=False, satisfied=True)
        return ContractSatisfaction(
            applicable=True, satisfied=True, reason="search produced results to open"
        )

    evidence_low = evidence.lower()
    missing: List[str] = []

    if contract.datums and not any(
        _number_near(evidence_low, variants) for _canon, variants in contract.datums
    ):
        missing.append(
            "required datum: " + "/".join(canon for canon, _v in contract.datums)
        )

    if not _subject_present(evidence, contract.subject_tokens):
        missing.append("subject: " + "/".join(contract.subject_tokens))

    if missing:
        return ContractSatisfaction(
            applicable=True,
            satisfied=False,
            missing=missing,
            reason="the retrieved evidence is missing " + "; ".join(missing),
        )
    return ContractSatisfaction(
        applicable=True, satisfied=True,
        reason="the retrieved evidence carries the contract's required content",
        datum_verified=bool(contract.datums),
    )
