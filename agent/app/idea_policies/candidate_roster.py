"""Candidate-roster contract: one page-attributable verdict per enumerated candidate.

A real strong agent solving test 122 (four large radio telescopes, exactly one is the largest
filled-aperture single dish CURRENTLY in operation) visited every candidate it already
suspected was wrong and came back with a QUOTED, page-attributable reason for each
elimination before it named a survivor. The engine's weak-model mirror image was diagnosed on
2026-08-21: task 122's real 0.50 -> 0.80 improvement came from a config cap that FORCED all-four
coverage (``branch_exploration`` 0.25 -> 1.0), never from the engine deciding to check everyone
-- and a candidate the engine did "check" is recorded only as a visit, with no field anywhere
asserting WHY it was eliminated. "Visited and silently skipped" and "visited and disqualified
for a stated reason" are the same thing to every downstream mechanism.

This module builds the missing record. For each candidate the mandate enumerates it produces
``{candidate, status, disqualifier_quote, source_url}``, where the quote is a statement the run
itself made about that candidate whose content is corroborated by the raw text of the page
opened for it (for the survivor, the same shape carries the CONFIRMATION rather than a
disqualifier -- the two are not mechanically distinguishable and this module does not try).

Design notes
------------
* **Nothing is asked of the model.** No new schema field, no new prompt instruction, no new LLM
  call. Statements are mined from the merge's own synthesis plus its children's direct text, and
  corroborated against fetched page text -- the same "assertion vs. what the run actually
  fetched" comparison ``grounding.verify_numeric_claims`` makes for numbers. This follows the
  trace's B9 lesson (require presence, never trust a small model's content) and is why the
  mechanism is not dark below 14b the way an ``expect``-keyed one would be.

* **Enforcement is borrowed, not built.** A roster gap becomes a ``missing_requirements`` entry,
  which the merge's existing internal-consistency guard already turns into a not-achieved
  verdict. That guard is the largest single accuracy lever found in this chain of work, so a
  new mechanism riding it is strictly preferable to a second, parallel enforcement path.

* **Detection is unconditional, acting on it is gated** (``MergeConfig.candidate_roster_enabled``),
  matching every other check in ``MergeLeafAction.execute``: the audit is stamped on the graph
  root and the merge node whatever the flag says, so its live false-positive rate is measurable
  from report captures before it decides anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.candidate_coverage import _fuzzy_contains, _norm
from agent.app.idea_policies.grounding import _CLAIM_RE, _normalize_number
from agent.app.idea_policies.mandate_requirements import parse_mandate_requirements

#: Root-side registry key, in family with ``alternative_branch.RACE_GROUPS``: graph-wide
#: bookkeeping lives on the node that owns the whole graph's scope, which for a roster derived
#: from the mandate is the root.
CANDIDATE_ROSTER = "candidate_roster"

#: Merge-node marker naming the candidates whose disposition is not page-attributable.
ROSTER_UNDISPOSED_MARKER = "candidate_roster_undisposed"
#: Merge-node marker for the B7 magnitude tripwire.
ROSTER_MAGNITUDE_MARKER = "candidate_roster_magnitude_tripwire"

# --- entry statuses ---------------------------------------------------------------------

#: A statement about the candidate exists AND its content is corroborated by the page opened
#: for it. The only status that satisfies the contract.
DISPOSED = "disposed"
#: A statement exists but nothing the run fetched for this candidate corroborates it -- the
#: parametric-memory shape ("RATAN-600 is a ring reflector") stated without ever reading it.
UNSOURCED = "unsourced"
#: The run never said anything about this candidate at all.
UNRESOLVED = "unresolved"
#: Corroborated, but the mandate's superlative is time-indexed (B8) and the statement describes
#: a dateable EVENT without giving its date -- what stale parametric memory produces ("Arecibo
#: collapsed", recalled rather than read). An ongoing-state disposition is exempt; see
#: :func:`_year_required`.
UNDATED = "undated"

_SATISFIED = frozenset({DISPOSED})

# Sentence split: terminal punctuation followed by space, or a line break. Bullet/table rows in
# a child's direct text are one statement each, which is what the line-break arm is for.
_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+|[\n\r]+|\s+\|\s+")

#: 1500-2099. Narrow on purpose: a 4-digit number outside this window is far more often an
#: elevation or a capacity than a date.
_YEAR_RE = re.compile(r"\b(?:1[5-9]\d\d|20\d\d)\b")

#: An ONGOING-STATE disposition: the candidate is simply still doing what it does. Nothing
#: happened to it on a date, so there is no year to cite (see :func:`_year_required`).
_STATE_CUE = (
    r"(?:operational|operating|operates|in\s+operation|in\s+service|in\s+use|active|online|"
    r"functioning|functional|still\s+(?:running|standing|working|used)|"
    r"remains?\s+in\s+(?:operation|service|use)|continues?\s+to\s+operate)"
)
_ONGOING_STATE_RE = re.compile(r"\b" + _STATE_CUE + r"\b")

#: The same state word turned into a PAST or NEGATED claim -- "no longer operational", "was
#: active until", "formerly in service". Those are terminal events wearing a state word, and a
#: date is exactly what they owe.
_NEGATED_STATE_RE = re.compile(
    r"\b(?:not|never|no\s+longer|formerly|previously|once|until|ceased\s+being|was|were)\s+"
    r"(?:\w+\s+){0,2}" + _STATE_CUE + r"\b"
)

#: A STRUCTURAL/CATEGORICAL disposition: a fact about what KIND of thing the candidate is. Task
#: 122's mandate supplies two of these itself ("one is a ring reflector (not a filled aperture),
#: and one is a smaller fully steerable dish") -- geometry facts that carry no date and never
#: will, so a year requirement can only ever be a false positive on them. Cues are taken from
#: what real disqualifiers in this repo's tasks actually say: an explicit category negation
#: ("not a filled aperture", "rather than a continuous dish") and the named alternative
#: geometries themselves.
_STRUCTURAL_RE = re.compile(
    r"\b(?:"
    r"(?:not|rather\s+than)\s+(?:a|an)\s+(?:\w+[-\s]){0,3}"
    r"(?:aperture|dish|reflector|telescope|antenna|array)|"
    r"ring\s+(?:reflector|antenna|dish)|"
    r"(?:fully\s+)?steerable|"
    r"filled[-\s]aperture|"
    r"phased\s+array|interferometer"
    r")\b"
)

#: A TERMINAL EVENT: something dateable happened to the candidate and knocked it out. This is
#: the B8 target -- "Arecibo is out because it collapsed" recalled from parametric memory with
#: no date is indistinguishable from a stale guess, so the year stays load-bearing here.
_TERMINAL_EVENT_RE = re.compile(
    r"\b(?:collapse|collapsed|collapsing|closed|closure|shut\s*down|shutdown|"
    r"cease[sd]?|decommission(?:ed|ing)?|destroyed|demolished|dismantled|"
    r"retired|superseded|surpassed|replaced|defunct|abandoned|damaged\s+beyond|"
    r"out\s+of\s+(?:service|operation|commission)|no\s+longer)\b"
)

#: Cues that a statement ELECTS the candidate it names rather than eliminating it. Used only to
#: pick out the survivor for the B7 tripwire; a candidate's own status never depends on it.
_SURVIVOR_CUES = (
    "survivor", "surviving", "survives", "the answer", "elected", "winner",
    "is the largest", "is the biggest", "is the only", "only one that",
    "meets both", "satisfies", "qualifies", "therefore the",
)

#: Words carrying no evidential weight, dropped before corroborating a statement against page
#: text. Short by design: the corroboration bar is deliberately lenient (see
#: :func:`_corroborated`), so trimming stopwords is about not letting "the of and" carry a
#: statement, rather than about precision.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "no", "not", "of", "on", "one", "or", "than", "that",
    "the", "this", "to", "was", "were", "which", "with", "so", "any", "all", "there",
})

#: Metres per unit, for the B7 magnitude comparison only. Keys are the spellings
#: ``grounding._CLAIM_RE`` can capture; anything absent here makes a magnitude uncomparable,
#: and an uncomparable magnitude never fires the tripwire.
_METRES_PER_UNIT = {
    "m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometre": 1000.0, "kilometres": 1000.0,
    "kilometer": 1000.0, "kilometers": 1000.0,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
}

#: Fraction of a statement's content tokens that must appear in the candidate's page text for
#: the statement to count as page-attributable. Lenient on purpose: the check exists to catch a
#: verdict invented wholesale, and a page is a large haystack, so a strict bar would only
#: punish paraphrase -- the same reason the numeric-provenance check fires solely when EVERY
#: figure is missing.
_CORROBORATION_RATIO = 0.5


@dataclass
class RosterEntry:
    """One candidate's recorded disposition."""

    candidate: str
    status: str = UNRESOLVED
    disqualifier_quote: str = ""
    source_url: str = ""
    survivor: bool = False
    #: Magnitude in metres, when one could be read for this candidate (B7 only).
    magnitude_m: Optional[float] = None

    @property
    def satisfied(self) -> bool:
        return self.status in _SATISFIED

    def as_dict(self) -> Dict[str, Any]:
        """The stamped shape -- the contract's four fields plus the B7 bookkeeping."""
        return {
            "candidate": self.candidate,
            "status": self.status,
            "disqualifier_quote": self.disqualifier_quote,
            "source_url": self.source_url,
            "survivor": self.survivor,
            "magnitude_m": self.magnitude_m,
        }


@dataclass
class RosterAudit:
    """Outcome of a candidate-roster audit."""

    active: bool = False
    time_indexed: bool = False
    entries: List[RosterEntry] = field(default_factory=list)
    #: Candidates whose magnitude beats the elected survivor's WITHOUT a corroborated
    #: disqualifier (B7).
    magnitude_tripwire: List[str] = field(default_factory=list)

    @property
    def undisposed(self) -> List[RosterEntry]:
        return [e for e in self.entries if not e.satisfied]

    @property
    def complete(self) -> bool:
        return self.active and not self.undisposed

    def as_registry(self) -> List[Dict[str, Any]]:
        return [e.as_dict() for e in self.entries]

    def missing_requirements(self) -> List[str]:
        """One ``missing_requirements`` line per gap, sharpest first.

        The B7 tripwire is reported separately from the plain gap it implies: an undisqualified
        candidate that is also LARGER than the elected survivor is the specific failure the
        strong-agent trace caught (RATAN-600 is physically bigger than FAST and is still
        correctly eliminated -- but only because a geometry predicate was read off its page,
        which pure argmax cannot do in either direction).
        """
        lines: List[str] = []
        for name in self.magnitude_tripwire:
            lines.append(
                f"candidate '{name}' is LARGER than the elected survivor and no disqualifying "
                "reason attributable to its own page was recorded"
            )
        tripped = set(self.magnitude_tripwire)
        for entry in self.undisposed:
            if entry.candidate in tripped:
                continue
            lines.append(_gap_line(entry))
        return lines


def _gap_line(entry: RosterEntry) -> str:
    if entry.status == UNDATED:
        return (
            f"the status recorded for candidate '{entry.candidate}' carries no date, and the "
            "mandate asks about the present state"
        )
    if entry.status == UNSOURCED:
        return (
            f"the reason recorded for candidate '{entry.candidate}' is not supported by any "
            "page opened for it"
        )
    return f"no disposition was recorded for candidate '{entry.candidate}'"


# --- evidence gathering -----------------------------------------------------------------


@dataclass
class _Page:
    """One successfully visited page. ``tokens`` is computed once: the corroboration check runs
    per candidate per statement, and re-tokenizing a full Wikipedia page each time would make a
    default-arm detection pass measurably expensive."""

    identity: str
    url: str
    text: str
    _tokens: Optional[Set[str]] = None

    @property
    def tokens(self) -> Set[str]:
        if self._tokens is None:
            self._tokens = set(_norm(self.text).split())
        return self._tokens


def _visited_pages(graph) -> List[_Page]:
    """One :class:`_Page` per successfully visited page.

    ``identity`` is title/h1/url only -- never body text. Matching a candidate against a page's
    BODY would attribute the FAST page to Arecibo, since the survivor's page names every
    competitor it beat; the page opened FOR a candidate is the one whose title says so.
    """
    from agent.app.idea_policies.action_constants import ActionResultKey

    pages: List[_Page] = []
    for node in graph.iter_depth_first():
        ar = (getattr(node, "details", {}) or {}).get(DetailKey.ACTION_RESULT.value) or {}
        if not (
            isinstance(ar, dict)
            and ar.get("action") == IdeaActionType.VISIT.value
            and ar.get("success")
        ):
            continue
        url = ""
        for key in ("url", "source_url", "final_url"):
            value = ar.get(key)
            if isinstance(value, str) and value:
                url = value
                break
        identity = " | ".join(
            str(ar.get(key))
            for key in ("page_title", "h1_text", "title", "url", "source_url")
            if isinstance(ar.get(key), str) and ar.get(key)
        )
        text = (
            ar.get(ActionResultKey.CONTENT_FULL.value)
            or ar.get(ActionResultKey.CONTENT.value)
            or ar.get(ActionResultKey.CONTENT_WITH_LINKS.value)
        )
        pages.append(_Page(identity=identity, url=url,
                           text=text if isinstance(text, str) else ""))
    return pages


def _statement_sources(claim_text: Any, merged_results: Optional[Sequence[Any]]) -> List[str]:
    """Everything the run ASSERTS: this merge's synthesis plus its children's own text.

    Two kinds of child are excluded, both to keep the corroboration check from becoming
    circular or self-certifying:

    - a nested MERGE, for the same reason ``grounding.answer_numeric_provenance`` excludes one:
      it hands up prose, so crediting it would let an invented verdict certify its parent's;
    - a VISIT, whose payload is the fetched page itself and carries no verdict of its own (see
      ``VisitLeafAction``'s result fields). Counting a page dump as a statement would make every
      opened page corroborate itself, which is precisely the "visited and silently skipped"
      case the roster exists to tell apart from a real disqualification.
    """
    from agent.app.idea_policies.merge import _direct_text

    texts: List[str] = []
    if isinstance(claim_text, str) and claim_text.strip():
        texts.append(claim_text)
    for item in merged_results or []:
        if not isinstance(item, dict) or item.get("is_merge"):
            continue
        result = item.get("result")
        if isinstance(result, dict) and result.get("action") == IdeaActionType.VISIT.value:
            continue
        if result:
            direct = _direct_text(result)
            if direct:
                texts.append(direct)
    return texts


def _sentences(texts: Sequence[str]) -> List[str]:
    out: List[str] = []
    for text in texts:
        for part in _SENTENCE_RE.split(str(text or "")):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _content_tokens(text: str, drop: Set[str]) -> List[str]:
    return [
        tok for tok in _norm(text).split()
        if tok and tok not in _STOPWORDS and tok not in drop
    ]


def _corroborated(statement: str, candidate: str, page: _Page) -> bool:
    """Is what the run SAID about this candidate present in the page it opened for it?"""
    if not page.text:
        return False
    drop = set(_norm(candidate).split())
    tokens = _content_tokens(statement, drop)
    if not tokens:
        return False
    hits = sum(1 for tok in tokens if tok in page.tokens)
    return hits >= 1 and (hits / len(tokens)) >= _CORROBORATION_RATIO


#: Clause boundaries WITHIN one statement. ``_SENTENCE_RE`` has already cut at terminal
#: punctuation, so what is left to separate are the coordinated clauses of a single sentence --
#: which is how a model actually reports a four-candidate roster ("Arecibo has collapsed,
#: RATAN-600 is operational, and FAST is operational").
_CLAUSE_RE = re.compile(
    r",\s*|\s+(?:and|but|while|whereas|however|although|though)\s+", re.IGNORECASE
)

_MINE, _OTHER, _UNOWNED = 1, -1, 0


def _candidate_clause(statement: str, aliases: Sequence[str],
                      other_aliases: Sequence[str]) -> str:
    """The portion of ``statement`` that speaks about THIS candidate.

    The 2026-08-22 re-validation found a cross-candidate cue bleed: models routinely put the
    whole roster in ONE sentence, and :func:`_year_required` read all of it, so Arecibo's
    "collapsed" forced a year onto the other three candidates' dateless ongoing-state clauses.
    The scoping is in the spirit of ``alternative_branch``'s entity-anchored number windows
    (commit ``4bc25c8f``) -- keep only the text the named entity owns -- but clause-based rather
    than a character window, because a four-candidate sentence fits inside any useful window.

    Clauses naming no candidate are CONTINUATIONS and attach to the nearest named clause,
    preceding first ("RATAN-600 uses a ring reflector, not a filled aperture" is one
    disposition). When no OTHER candidate is named at all there is nothing to separate, so the
    statement is returned untouched -- single-candidate statements behave exactly as before.

    Ownership uses the same ``_fuzzy_contains`` name match the mention list is built from, so a
    clause that shortens a candidate's name past recognition ("Arecibo" for "Arecibo
    Telescope") owns nothing and the scoping fails OPEN, back to the whole statement -- the
    same name that would not have earned a mention of its own either.
    """
    pieces = _CLAUSE_RE.split(statement)
    owners = []
    for piece in pieces:
        if any(_fuzzy_contains(alias, piece) for alias in aliases):
            owners.append(_MINE)
        elif any(_fuzzy_contains(alias, piece) for alias in other_aliases):
            owners.append(_OTHER)
        else:
            owners.append(_UNOWNED)
    if _OTHER not in owners or _MINE not in owners:
        return statement
    for i, owner in enumerate(owners):
        if owner != _UNOWNED:
            continue
        back = next((o for o in reversed(owners[:i]) if o != _UNOWNED), None)
        fwd = next((o for o in owners[i + 1:] if o != _UNOWNED), None)
        owners[i] = back if back is not None else fwd
    scoped = ", ".join(
        piece for piece, owner in zip(pieces, owners) if owner == _MINE and piece.strip()
    )
    # A COORDINATED SUBJECT ("The RATAN-600 and FAST telescopes are operational", a real live
    # wording) leaves this candidate holding a bare name with the shared predicate parked in the
    # sibling's clause. Scoping to a name says nothing, so fall open to the whole statement.
    drop = {tok for alias in aliases for tok in _norm(alias).split()}
    if not _content_tokens(scoped, drop):
        return statement
    return scoped


def _year_required(statement: str) -> bool:
    """Does this disposition owe a year token (B8)?

    A time-indexed mandate ("the largest CURRENTLY IN OPERATION") makes a candidate's status a
    fact about a point in time, and the original B8 finding is that a weak model recalls that
    status from parametric memory: "Arecibo is the largest" with no date, blind to the December
    2020 collapse. A year is the cheapest deterministic evidence that a dated event was actually
    read rather than remembered.

    That reasoning only covers dispositions that ARE events. Live validation on 72 real ollama
    merge completions (2026-08-21) found the flat rule downgraded 2 of 3 CORRECT answers: models
    write "RATAN-600: Operational" and "Green Bank Telescope is fully steerable and operational"
    -- page-corroborated, and with no year available to cite, because a telescope that is simply
    still running has no terminal date. For those, page corroboration (checked separately, in
    :func:`_corroborated`) is the whole of the available evidence.

    So the year is required unless the statement is unambiguously ONGOING-STATE shaped: a state
    cue with neither a terminal-event verb nor a past/negated reading of the state word itself
    ("no longer operational" is a collapse, not a status). Anything else -- including a bare
    descriptive claim with no cue either way -- keeps the requirement, which is what preserves
    the original target: the fix narrows the rule to statements that positively assert an
    ongoing state, it does not invert the default.

    A second exemption, from the 2026-08-22 re-validation: a STRUCTURAL/CATEGORICAL disposition
    ("RATAN-600 uses a ring reflector, not a filled-aperture dish") is a fact about what kind of
    thing the candidate is. It has no date, it will never acquire one, and on task 122 it is the
    disqualifier the MANDATE ITSELF hands the run -- so demanding a year for it killed a correct
    answer outright. Like the ongoing-state exemption this is a THIRD category, not a loosening:
    a terminal-event or negated-state reading still wins, so "Arecibo collapsed" owes its date
    whatever else the clause says about geometry.
    """
    low = statement.lower()
    if _TERMINAL_EVENT_RE.search(low) or _NEGATED_STATE_RE.search(low):
        return True
    return not (_ONGOING_STATE_RE.search(low) or _STRUCTURAL_RE.search(low))


def _magnitude(text: str) -> Optional[float]:
    """First number-with-unit in ``text``, in metres, or ``None`` when uncomparable."""
    for match in _CLAIM_RE.finditer(str(text or "")):
        value = _normalize_number(match.group(1))
        factor = _METRES_PER_UNIT.get(match.group(2).lower())
        if value and factor:
            try:
                return float(value) * factor
            except ValueError:
                continue
    return None


#: A parenthetical on the candidate's own mandate line, which is where its ALIAS lives.
_PARENTHETICAL_RE = re.compile(r"\(([^)]{3,60})\)")

#: Aliases are matched against page titles, so the same "name-like" bar as a roster entry.
_MAX_ALIAS_WORDS = 6


def _aliases(mandate: str, candidate: str) -> List[str]:
    """``candidate`` plus the alias(es) its mandate line supplies in parentheses.

    Task 122 enumerates "FAST (Five-hundred-meter Aperture Spherical Telescope)", and
    ``candidate_coverage._extract_name`` keeps only the part before the parenthesis -- so the
    roster name is ``FAST`` while every page the run opened for it is titled
    *Five-hundred-meter Aperture Spherical Telescope*. Replaying the 15 captured task-122
    reports showed this is not hypothetical: the two runs that DID open all four candidate
    pages would both have been scored as never having opened the survivor's.

    A parenthetical only counts as an alias if it reads like a NAME -- at least one
    capitalized token -- which is what separates "(Five-hundred-meter Aperture Spherical
    Telescope)" from the descriptive "(a 305 m spherical dish)" on the line above it.
    """
    out = [candidate]
    for line in str(mandate or "").splitlines():
        if not _fuzzy_contains(candidate, line):
            continue
        for match in _PARENTHETICAL_RE.finditer(line):
            alias = match.group(1).strip()
            words = alias.split()
            if not 1 <= len(words) <= _MAX_ALIAS_WORDS:
                continue
            if not any(word[:1].isupper() for word in words):
                continue
            if _norm(alias) not in {_norm(a) for a in out}:
                out.append(alias)
    return out


def _mandate_magnitude(mandate: str, candidate: str) -> Optional[float]:
    """The magnitude the MANDATE itself attributes to a candidate, if any.

    Task 122's roster line for Arecibo reads "(a 305 m spherical dish)". A figure the mandate
    supplied is not something the run can have invented, so it is a sound fallback when the run
    never restated one -- and it is what lets the tripwire fire against a candidate the run
    mentioned only in passing.
    """
    for line in str(mandate or "").splitlines():
        if _fuzzy_contains(candidate, line):
            found = _magnitude(line)
            if found is not None:
                return found
    return None


# --- the audit --------------------------------------------------------------------------


def audit_candidate_roster(
    graph,
    mandate: str,
    claim_text: Any = "",
    merged_results: Optional[Sequence[Any]] = None,
) -> RosterAudit:
    """Does every candidate the mandate enumerates have a page-attributable verdict?

    Inert (``active=False``) unless the mandate both enumerates name-like candidates AND asks
    for individual disposition -- see ``MandateRequirements.requires_candidate_roster``.
    """
    req = parse_mandate_requirements(str(mandate or ""))
    if not req.requires_candidate_roster:
        return RosterAudit()

    audit = RosterAudit(active=True, time_indexed=req.time_indexed)
    pages = _visited_pages(graph)
    statements = _sentences(_statement_sources(claim_text, merged_results))

    for name in req.roster_candidates:
        audit.entries.append(
            _entry_for(name, pages, statements, req.time_indexed, str(mandate or ""),
                       req.roster_candidates)
        )
    _apply_magnitude_tripwire(audit)
    return audit


def _entry_for(
    candidate: str,
    pages: Sequence[_Page],
    statements: Sequence[str],
    time_indexed: bool,
    mandate: str,
    roster: Sequence[str] = (),
) -> RosterEntry:
    entry = RosterEntry(candidate=candidate)
    aliases = _aliases(mandate, candidate)
    others = [
        alias
        for name in roster if _norm(name) != _norm(candidate)
        for alias in _aliases(mandate, name)
    ]
    matched = [
        page for page in pages
        if any(_fuzzy_contains(alias, page.identity) for alias in aliases)
    ]
    if matched:
        entry.source_url = matched[0].url
    mentions = [
        s for s in statements if any(_fuzzy_contains(alias, s) for alias in aliases)
    ]
    if not mentions:
        entry.magnitude_m = _mandate_magnitude(mandate, candidate)
        return entry

    entry.survivor = any(
        cue in mention.lower() for mention in mentions for cue in _SURVIVOR_CUES
    )
    # Best available statement, in the order the contract cares about: corroborated AND dated
    # beats corroborated beats merely stated. The status records how far it got, so a run that
    # opened the page but paraphrased past recognition is reported differently from one that
    # never opened it.
    best = mentions[0]
    status = UNSOURCED
    for mention in mentions:
        page = next(
            (page for page in matched if _corroborated(mention, candidate, page)),
            None,
        )
        if page is None:
            continue
        # Both halves of the date test read the candidate's OWN clause: a sibling's terminal
        # event must not impose a year here, and a sibling's date must not satisfy one either.
        clause = _candidate_clause(mention, aliases, others)
        dated = bool(_YEAR_RE.search(clause)) or not _year_required(clause)
        if not time_indexed or dated:
            best, status = mention, DISPOSED
            if page.url:
                entry.source_url = page.url
            break
        if status == UNSOURCED:
            best, status = mention, UNDATED
            if page.url:
                entry.source_url = page.url
    entry.disqualifier_quote = best
    entry.status = status
    entry.magnitude_m = _magnitude(best)
    if entry.magnitude_m is None:
        entry.magnitude_m = _mandate_magnitude(mandate, candidate)
    return entry


def _apply_magnitude_tripwire(audit: RosterAudit) -> None:
    """B7: every candidate BIGGER than the elected survivor must be quotably disqualified.

    The usual weak-model bug is "take the biggest number"; inverted, the same comparison is a
    tripwire. It fires only when a survivor was actually elected and both magnitudes are
    comparable, so it is silent on the many mandates whose criterion is not a magnitude at all.
    """
    survivors = [e for e in audit.entries if e.survivor]
    if len(survivors) != 1:
        return
    survivor = survivors[0]
    if survivor.magnitude_m is None:
        return
    audit.magnitude_tripwire = [
        e.candidate
        for e in audit.entries
        if not e.satisfied
        and not e.survivor
        and e.magnitude_m is not None
        and e.magnitude_m > survivor.magnitude_m
    ]
