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
  single item, etc.), so the gate imposes nothing on non-branch-eliminate shapes.

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


def _node_haystacks(graph) -> List[str]:
    """One searchable text blob per SUCCESSFULLY-VISITED page in the graph.

    A candidate is only credited as "resolved" when a real page was OPENED for it —
    i.e. a node carries a successful ``visit`` action_result. We deliberately ignore
    node titles (the root's title is the mandate itself, which enumerates every
    candidate name, so matching against it would trivially "resolve" all candidates
    with zero navigation) and search-action results (search returns only engine
    snippets that mention a candidate's NAME without ever reading its criterion, which
    is exactly the short-circuit this gate exists to prevent). The disambiguating fact
    (e.g. a river's mouth in the infobox) lives on the page body, so we match against
    the visited page's title, url, and content only.
    """
    blobs: List[str] = []
    for node in graph.iter_depth_first():
        details = getattr(node, "details", {}) or {}
        ar = details.get(DetailKey.ACTION_RESULT.value)
        if not (
            isinstance(ar, dict)
            and ar.get("action") == IdeaActionType.VISIT.value
            and ar.get("success")
        ):
            continue
        parts: List[str] = []
        for key in ("page_title", "h1_text", "title", "url", "source_url", "content"):
            val = ar.get(key)
            if isinstance(val, str) and val:
                parts.append(val)
        if parts:
            blobs.append(" | ".join(parts))
    return blobs


@dataclass
class CandidateCoverageResult:
    """Outcome of a candidate-coverage check."""

    satisfied: bool
    named: List[str] = field(default_factory=list)
    resolved: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


def evaluate_candidate_coverage(graph, mandate: str) -> CandidateCoverageResult:
    """Report whether every enumerated candidate named in ``mandate`` has been touched
    by some node in ``graph`` (title or result text fuzzy-matches the candidate name).

    Fails OPEN: when the mandate names no enumerable candidates, ``satisfied`` is True.

    Note: no shape-classifier wiring is needed here. Because ``extract_named_candidates``
    only returns a non-empty list for enumerated candidate mandates (which is exactly
    the ``branch_eliminate`` shape in ``shape_classifier.classify_shape``), this gate
    naturally engages ONLY for branch-eliminate-shaped mandates and stays inert
    (``satisfied=True``) for chain / parallel_merge / plain fan-out tasks.
    """
    named = extract_named_candidates(mandate)
    if not named:
        return CandidateCoverageResult(satisfied=True, named=[], resolved=[], missing=[])

    haystacks = _node_haystacks(graph)
    resolved: List[str] = []
    missing: List[str] = []
    for name in named:
        if any(_fuzzy_contains(name, hay) for hay in haystacks):
            resolved.append(name)
        else:
            missing.append(name)
    return CandidateCoverageResult(
        satisfied=len(missing) == 0,
        named=named,
        resolved=resolved,
        missing=missing,
    )
