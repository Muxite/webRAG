"""
Candidate-coverage gate for "branch-eliminate then chain forward" task shapes.

Some mandates enumerate K similarly-named candidates and a single distinguishing
criterion, and require the agent to check ALL K before electing a survivor (see
``test_095_tier5_branch_eliminate_chain``: four Rivers Avon, exactly one empties
into the English Channel). Weak executor models tend to short-circuit — they pick
the most familiar candidate and finalize early. This module provides a DETERMINISTIC,
code-level coverage check the engine can use as a soft replan gate: if the mandate
names candidates and the graph has not touched all of them, force another pass.

Design notes
------------
* ``extract_named_candidates`` must fail OPEN: it returns ``[]`` for any mandate that
  is not an enumerated candidate list (plain prose, numbered INSTRUCTION steps, a
  single item, etc.), so the gate imposes nothing on un-enumerated shapes. It does NOT
  narrow to branch-eliminate: any mandate that enumerates >= 2 names qualifies, which
  includes breadth / fan-out rosters (test_052's six novels) — see
  ``evaluate_candidate_coverage``.

* Distinguishing a candidate list ("1. River Avon, Bristol — ...") from an
  instruction list ("1. Identify the POET ...", "2. Open that page ...") is the key
  design risk. The reliable structural signal in the real corpus: instruction steps
  begin with an imperative VERB, candidate entries begin with a proper NOUN / name.
  We reject the whole list if ANY item's first token is a known imperative verb, and
  we extract only the short NAME portion (before an em-dash / parenthetical / colon).

* Fuzzy matching reuses no code from ``got_operations.py`` on purpose: that module's
  dedup similarity is EMBEDDING-based (a Chroma distance run through
  ``plan_library.retrieval.similarity_from_distance``),
  which is unavailable offline and semantically wrong here (it dedups thought nodes,
  not proper-name spellings). We use stdlib ``difflib`` ratio + substring containment.
  We reuse the *value* of ``got_dedup_similarity_threshold`` (0.85) as the fuzzy cutoff
  because it is the project's established "these two strings mean the same thing" bar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List

from agent.app.idea_policies.base import DetailKey, IdeaActionType


# Imperative verbs that begin an INSTRUCTION step (not a candidate name). If any
# enumerated item starts with one of these, the whole list is treated as instructions
# and NO candidates are extracted (fail-open). Kept broad on purpose.
_INSTRUCTION_VERBS = frozenset(
    {
        "identify", "open", "read", "search", "find", "report", "determine",
        "visit", "go", "check", "look", "locate", "use", "confirm", "verify",
        "compare", "list", "compute", "calculate", "select", "choose", "extract",
        "navigate", "follow", "note", "record", "retrieve", "gather", "collect",
        "count", "measure", "review", "examine", "inspect", "obtain", "fetch",
        "return", "provide", "give", "state", "pick", "cross", "then", "next",
        "first", "finally", "start", "begin", "answer", "resolve", "trace",
        "match", "map", "add", "sum", "subtract", "multiply", "divide", "rank",
        "sort", "filter", "scan", "query", "call", "fill", "produce", "output",
        "write", "name",
    }
)

# A numbered list item at line start: "  1. <body>" / "1) <body>".
_NUMBERED_LINE = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$", re.MULTILINE)

# Delimiters that separate a short NAME from its trailing description.
_NAME_DELIMS = re.compile(r"\s+[—–]\s+|\s+-\s+|\s*[(:]")


def _first_token(text: str) -> str:
    m = re.match(r"[A-Za-z']+", text.strip())
    return m.group(0).lower() if m else ""


def _extract_name(body: str) -> str:
    """Return the NAME portion of a candidate item (before an em-dash / parenthetical
    / colon), stripped of surrounding quotes and punctuation."""
    name = _NAME_DELIMS.split(body, maxsplit=1)[0]
    return name.strip().strip("'\"").strip()


def strip_enumerated_items(mandate: str) -> str:
    """``mandate`` with every numbered list line blanked out.

    Lets a caller ask "does the SURROUNDING PROSE say X" without an enumerated item's own
    wording answering for it (e.g. ``shape_classifier``'s breadth check: task 034's items
    each carry their own "how many ..." question, which is a per-item ask, not an
    aggregation over the roster).
    """
    return _NUMBERED_LINE.sub(" ", mandate or "")


def extract_named_candidates(mandate: str) -> List[str]:
    """Extract an enumerated candidate list's NAMES from ``mandate``.

    Returns the name portion (before the em-dash/parenthetical/description) of each
    item in the longest consecutive ``1, 2, ... N`` numbered run with ``N >= 2``.

    Fails OPEN (returns ``[]``) when:
    * there is no consecutive numbered run of length >= 2, or
    * ANY item begins with an imperative verb (it's an INSTRUCTION list, not
      candidates — e.g. test_051 / test_065's "1. Identify ... 2. Open ..."), or
    * fewer than 2 non-empty names survive extraction.
    """
    if not mandate:
        return []

    # Collect (index, body) for every numbered line, then take the longest run that
    # starts at 1 and increments by 1 (so a stray "1." in later prose can't extend it).
    items = [(int(m.group(1)), m.group(2)) for m in _NUMBERED_LINE.finditer(mandate)]
    if not items:
        return []

    run: List[str] = []
    expected = 1
    for idx, body in items:
        if idx == expected:
            run.append(body)
            expected += 1
        elif idx == 1:
            # A new list started; restart the run from here.
            run = [body]
            expected = 2
        else:
            # Non-consecutive number: the run 1..N ended.
            if len(run) >= 2:
                break
            run = []
            expected = 1
    if len(run) < 2:
        return []

    # Instruction-list guard: any imperative-verb-led item disqualifies the whole list.
    for body in run:
        if _first_token(body) in _INSTRUCTION_VERBS:
            return []

    names = [n for n in (_extract_name(body) for body in run) if n]
    if len(names) < 2:
        return []
    return names


# ---------------------------------------------------------------------------
# Fuzzy coverage evaluation
# ---------------------------------------------------------------------------

# Reuse the project's established "same string" bar (got_dedup_similarity_threshold).
_MATCH_THRESHOLD = 0.85


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _fuzzy_contains(candidate: str, haystack: str) -> bool:
    """True if ``candidate`` (a short proper name) is present in ``haystack``.

    Substring containment first (exact, cheap), then a token-window ``difflib`` ratio
    to tolerate minor spelling/word-order variance in a node title or result text.
    """
    cand = _norm(candidate)
    hay = _norm(haystack)
    if not cand or not hay:
        return False
    if cand in hay:
        return True
    cand_tokens = cand.split()
    hay_tokens = hay.split()
    n = len(cand_tokens)
    if n == 0 or n > len(hay_tokens):
        return False
    for i in range(len(hay_tokens) - n + 1):
        window = " ".join(hay_tokens[i : i + n])
        if SequenceMatcher(None, cand, window).ratio() >= _MATCH_THRESHOLD:
            return True
    return False


#: A visited page's own body, past this many chars, no longer counts as its "lede" for the
#: body-fallback match below — a comparison table or "See also" section deep in a LONG page
#: mentioning a DIFFERENT candidate by name must not count as evidence that page is ABOUT that
#: candidate. Roughly one infobox + opening paragraph's worth of text.
_BODY_LEDE_CHARS = 1000


@dataclass
class Haystack:
    """One visited page's searchable text, split so coverage matching can tell "this page IS
    about the candidate" (``identity`` — title/h1/url) from "the candidate's name merely
    appears somewhere in the page" (``body``). See :func:`evaluate_candidate_coverage_from_haystacks`
    for why the distinction matters — matching only against a merged blob let a candidate
    resolve off an INCIDENTAL mention on a DIFFERENT visited page (e.g. a "List of Seven
    Summits" cross-reference table on an unrelated mountain's article), live-observed as
    ``visit_count`` scoring lower than ``coverage`` on several 2026-08-23 breadth-suite cells.
    """

    identity: str
    body: str = ""


def _node_haystacks(graph) -> List[Haystack]:
    """One searchable ``Haystack`` per SUCCESSFULLY-VISITED page in the graph.

    A candidate is only credited as "resolved" when a real page was OPENED for it —
    i.e. a node carries a successful ``visit`` action_result. We deliberately ignore
    node titles (the root's title is the mandate itself, which enumerates every
    candidate name, so matching against it would trivially "resolve" all candidates
    with zero navigation) and search-action results (search returns only engine
    snippets that mention a candidate's NAME without ever reading its criterion, which
    is exactly the short-circuit this gate exists to prevent).
    """
    blobs: List[Haystack] = []
    for node in graph.iter_depth_first():
        details = getattr(node, "details", {}) or {}
        ar = details.get(DetailKey.ACTION_RESULT.value)
        if not (
            isinstance(ar, dict)
            and ar.get("action") == IdeaActionType.VISIT.value
            and ar.get("success")
        ):
            continue
        identity_parts: List[str] = []
        for key in ("page_title", "h1_text", "title", "url", "source_url"):
            val = ar.get(key)
            if isinstance(val, str) and val:
                identity_parts.append(val)
        content = ar.get("content")
        body = content if isinstance(content, str) else ""
        if identity_parts or body:
            blobs.append(Haystack(identity=" | ".join(identity_parts), body=body))
    return blobs


@dataclass
class CandidateCoverageResult:
    """Outcome of a candidate-coverage check."""

    satisfied: bool
    named: List[str] = field(default_factory=list)
    resolved: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    #: name -> "identity" | "body_lede", for whichever candidates resolved. Lets a forensic
    #: pass tell "a page ABOUT this candidate was visited" from "the name showed up in some
    #: other visited page's lede" without needing a full message-capture rerun.
    resolved_via: dict = field(default_factory=dict)


def evaluate_candidate_coverage_from_haystacks(
    haystacks: List[Haystack], mandate: str
) -> CandidateCoverageResult:
    """Report whether every enumerated candidate named in ``mandate`` resolves against at
    least one of ``haystacks`` (pages actually opened).

    The graph-independent core of :func:`evaluate_candidate_coverage`, split out so any
    executor with its own notion of "a page I actually opened" (e.g. an arm built on a
    linear message history instead of an ``IdeaDag``) can reuse the same deterministic
    coverage check without depending on the native engine's graph type.

    Matching is IDENTITY-PRIORITY: a candidate resolves if its name fuzzy-matches a visited
    page's ``identity`` (title/h1/url — "this page IS about the candidate") first; only if no
    page's identity matches does an incidental mention in the first ``_BODY_LEDE_CHARS`` of
    some page's ``body`` count (a narrow fallback for pages whose title doesn't literally
    repeat the candidate's name, e.g. a disambiguated or renamed subject) — never a mention
    anywhere deeper in a page's body, which is exactly the false-positive this split exists to
    close (live-observed: a fan-out candidate credited as "covered" via a comparison table on
    a DIFFERENT visited page, without its own page ever being opened).

    Fails OPEN: when the mandate names no enumerable candidates, ``satisfied`` is True.
    """
    named = extract_named_candidates(mandate)
    if not named:
        return CandidateCoverageResult(satisfied=True, named=[], resolved=[], missing=[])

    resolved: List[str] = []
    missing: List[str] = []
    resolved_via: dict = {}
    for name in named:
        via = None
        if any(_fuzzy_contains(name, hs.identity) for hs in haystacks):
            via = "identity"
        elif any(_fuzzy_contains(name, hs.body[:_BODY_LEDE_CHARS]) for hs in haystacks if hs.body):
            via = "body_lede"
        if via:
            resolved.append(name)
            resolved_via[name] = via
        else:
            missing.append(name)
    return CandidateCoverageResult(
        satisfied=len(missing) == 0,
        named=named,
        resolved=resolved,
        missing=missing,
        resolved_via=resolved_via,
    )


def evaluate_candidate_coverage(graph, mandate: str) -> CandidateCoverageResult:
    """Report whether every enumerated candidate named in ``mandate`` has been touched
    by some node in ``graph`` (title or result text fuzzy-matches the candidate name).

    Fails OPEN: when the mandate names no enumerable candidates, ``satisfied`` is True.

    Note on scope: no shape-classifier wiring is needed here, but this gate is NOT
    branch-eliminate-only. It engages for ANY mandate whose text enumerates >= 2 names,
    which covers ``branch_eliminate`` AND ``breadth`` (verified: test_052's six novels
    are all extracted, so a breadth run must have opened a page mentioning each novel
    before it may finalize). That is the intended reading of the gate — "every
    enumerated item was actually looked at" is exactly as meaningful for a fan-out
    roster as for a candidate list. It stays inert (``satisfied=True``) only for
    mandates with no enumerated name list at all: chains, parallel merges, prose
    mandates, and numbered INSTRUCTION lists (rejected by the imperative-verb guard).
    """
    return evaluate_candidate_coverage_from_haystacks(_node_haystacks(graph), mandate)


# ---------------------------------------------------------------------------
# OBSERVE-ONLY: cross-arm entity-conflict detection (run_policy_coverage_entity_conflict_check)
# ---------------------------------------------------------------------------
#
# `evaluate_candidate_coverage_from_haystacks` above matches a candidate against a POOLED
# haystack of every visited page -- it never asks WHICH arm's page resolved a candidate, or
# whether that arm's own goal named the same candidate. On a fan-out, a wrong page opened by
# one arm can textually mention several OTHER candidates too (a "list of ..." comparison
# page, a disambiguation page), letting every one of them register as "resolved" even though
# no arm ever opened ITS OWN correct page -- `coverage_ratio` then reads 1.0 falsely. See
# `run_policy_coverage_entity_conflict_check`'s docstring in config.py.
#
# The functions below add a SEPARATE, additive signal for this -- they never change
# `evaluate_candidate_coverage`'s own verdict (`satisfied`/`resolved`/`missing`), never touch
# the graph, and are always safe to skip: a caller that never calls them sees no difference.


@dataclass
class _NodeHaystack:
    """Same successful-VISIT scan as :class:`Haystack`, but additionally carries the OWNING
    NODE's own subject tokens -- the per-node link :func:`_node_haystacks` deliberately omits,
    needed here to ask "did the arm that opened THIS page actually intend this candidate", not
    just "does this page mention it somewhere".
    """

    node_id: str
    identity: str
    body: str
    own_subject_tokens: List[str]


def _node_haystacks_for_conflicts(graph) -> List["_NodeHaystack"]:
    from agent.app.idea_policies.contract_satisfaction import derive_step_contract

    blobs: List[_NodeHaystack] = []
    for node in graph.iter_depth_first():
        details = getattr(node, "details", {}) or {}
        ar = details.get(DetailKey.ACTION_RESULT.value)
        if not (
            isinstance(ar, dict)
            and ar.get("action") == IdeaActionType.VISIT.value
            and ar.get("success")
        ):
            continue
        identity_parts: List[str] = []
        for key in ("page_title", "h1_text", "title", "url", "source_url"):
            val = ar.get(key)
            if isinstance(val, str) and val:
                identity_parts.append(val)
        content = ar.get("content")
        body = content if isinstance(content, str) else ""
        if not (identity_parts or body):
            continue
        try:
            own_subject_tokens = derive_step_contract(node).subject_tokens
        except Exception:
            own_subject_tokens = []
        blobs.append(
            _NodeHaystack(
                node_id=getattr(node, "node_id", ""),
                identity=" | ".join(identity_parts),
                body=body,
                own_subject_tokens=own_subject_tokens,
            )
        )
    return blobs


def _candidate_named_by_subject_tokens(name: str, own_subject_tokens: List[str]) -> bool:
    """True if every substantial word of ``name`` matches one of ``own_subject_tokens``.

    ``own_subject_tokens`` (``contract_satisfaction._subject_tokens``) is sorted LONGEST-FIRST,
    not in the order the words appeared in the arm's own goal text -- so joining them into one
    string and doing an ORDERED n-gram match (as :func:`_fuzzy_contains` does, correctly, for
    page text) produces false results purely from word reordering ("Erie Canal" vs the token
    list ``["canal", "erie"]`` scores a 0.5 ratio, well under the match threshold, despite being
    the exact same two words). A set-membership check per candidate word is order-independent
    and the correct comparison against this specific, already-tokenized, already-lowercased
    field.
    """
    cand_tokens = [t for t in _norm(name).split() if len(t) >= 3]
    if not cand_tokens or not own_subject_tokens:
        return False
    subject_set = {t.lower() for t in own_subject_tokens}
    for tok in cand_tokens:
        if tok in subject_set:
            continue
        if any(SequenceMatcher(None, tok, s).ratio() >= _MATCH_THRESHOLD for s in subject_set):
            continue
        return False
    return True


def detect_candidate_coverage_entity_conflicts(graph, mandate: str) -> List[dict]:
    """OBSERVE-ONLY. For each candidate that :func:`evaluate_candidate_coverage_from_haystacks`
    would report RESOLVED, checks whether the page(s) that resolve it were opened by an arm
    whose OWN goal (:func:`contract_satisfaction.derive_step_contract`'s ``subject_tokens``)
    actually named that candidate. Flags a conflict only when NONE of the resolving pages were
    intended for this candidate -- the strongest signal available, short of a live rerun, that
    the candidate's "resolved" status came from a pooled cross-arm text collision (one wrong
    page whose body -- or even, per the sibling-link fallback bug this mechanism follows up on,
    whose own title -- happens to name a candidate its own arm never intended) rather than a
    real per-arm page-open.

    Deliberately does NOT treat ``resolved_via == "identity"`` as automatically safe: a page's
    OWN identity matching the candidate is exactly what the (now-fixed) sibling-link fallback
    bug could produce for the WRONG arm ("Visit the Suez Canal page" grounding on
    ``/wiki/Erie_Canal``, whose title literally says "Erie Canal") -- only whether the arm's
    own stated goal named this candidate is evidence of genuine intent.

    Never mutates the graph, never raises (a malformed ``action_result`` degrades to "no
    conflict" for that node rather than propagating), and never changes
    ``evaluate_candidate_coverage``'s own verdict -- this is a second, independent read over
    the same graph, not a modification of the first.

    Returns ``[]`` when the mandate names no candidates, or when no conflict is found.
    """
    named = extract_named_candidates(mandate)
    if not named:
        return []

    haystacks = _node_haystacks_for_conflicts(graph)
    conflicts: List[dict] = []
    for name in named:
        resolving: List[tuple] = []  # (haystack, "identity"|"body_lede")
        for nh in haystacks:
            if _fuzzy_contains(name, nh.identity):
                resolving.append((nh, "identity"))
            elif nh.body and _fuzzy_contains(name, nh.body[:_BODY_LEDE_CHARS]):
                resolving.append((nh, "body_lede"))
        if not resolving:
            continue  # not resolved at all -- candidate_coverage's own "missing", not a conflict

        if any(_candidate_named_by_subject_tokens(name, nh.own_subject_tokens) for nh, _via in resolving):
            continue  # at least one resolving page was genuinely intended for this candidate

        # Every page that resolved this candidate was unintended by its own arm -- a candidate
        # with even one genuinely-intended resolution is not a conflict, regardless of any
        # other page it happens to also be mentioned on.
        for nh, via in resolving:
            conflicts.append(
                {
                    "candidate": name,
                    "resolved_via": via,
                    "resolving_node_id": nh.node_id,
                    "resolving_page_identity": nh.identity,
                }
            )
    return conflicts
