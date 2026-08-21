"""
Deterministic (non-LLM) task-shape classifier over a mandate's text.

This is a deliberately small first draft. It recognises four reasoning shapes that
have appeared in the tier-4/5 corpus and returns ``None`` (fails OPEN) for anything
else, so it can never impose the wrong discipline on an unrecognised task:

* ``"branch_eliminate"``: the mandate enumerates K similarly-named candidates and a
  single distinguishing criterion, and asks the agent to eliminate down to one
  survivor (e.g. test_095: four Rivers Avon, exactly one empties into the English
  Channel). This is the shape with the most supporting infrastructure: the candidate
  list is detected with candidate_coverage.extract_named_candidates and there is a
  real rule file at reasoning_rules/branch_eliminate.md.

* ``"chain"``: a single dependency chain where each step's answer is needed to find
  the next page, with NO candidate-list structure (e.g. test_051, test_065).
  Keyword heuristic; a rule file exists at reasoning_rules/chain.md.

* ``"parallel_merge"``: two independent chains whose results are combined by an
  arithmetic operation (e.g. test_055, test_061: absolute difference of two
  founding years / birth years). Keyword heuristic; a rule file exists at
  reasoning_rules/parallel_merge.md.

* ``"breadth"``: N independent sub-goals over an enumerated roster of NAMES, aggregated
  at the end by an argmax/argmin/count/sum over the whole set (e.g. test_052: six novels
  -> six author birth years -> earliest). Detected from the roster
  (``mandate_requirements.parse_mandate_requirements``) plus an aggregation ask found in
  the surrounding PROSE; a per-item ask inside a list line does not count.

branch_eliminate and breadth reuse real supporting infrastructure beyond keyword matching
(the candidate-coverage extractor / the roster parser); chain and parallel_merge remain
best-effort keyword detectors, intentionally less rigorous, and their keyword coverage is
unmeasured beyond the two/three examples each was written against (see
docs/handoffs/SHAPE_ADAPTATION_OPEN_QUESTIONS.md Q1 for the methodology to re-run before
trusting this beyond that). The first three have a matching reasoning-rule file; breadth
deliberately does NOT — it is classification only, added because the corpus's canonical
fan-out tasks (052 and 26 others) previously classified as ``None``, leaving every
breadth-shaped mechanism without a trigger point. Nothing downstream keys off the new
label yet: every consumer either compares against a specific other label
(``classify_answer_shape``, ``chain_closure``) or looks the label up in a table that has
no ``breadth`` entry (``expansion._auto_reasoning_rules``, the plan-library archetype
rerank), so introducing it is behaviour-neutral by construction.

Order matters: parallel_merge is checked before chain because a two-chain task literally
contains the word "chain", and breadth is checked LAST so it can only claim mandates that
would otherwise be unclassified.
"""

from __future__ import annotations

from typing import Optional

from agent.app.idea_policies.candidate_coverage import (
    extract_named_candidates,
    strip_enumerated_items,
)
from agent.app.idea_policies.mandate_requirements import parse_mandate_requirements


# Disambiguation-style language that, combined with an enumerated candidate list,
# signals a branch-eliminate shape (as opposed to a plain breadth/fan-out list that
# also happens to be numbered, e.g. test_052's six novels, test_059's five players).
_DISAMBIG_PHRASES = (
    "exactly one of",
    "each stage's target is unknown",
    "do not simply guess",
    "identify the specific",
    "the specific one",
    "eliminate to one",
    "survivor",
)

# A two-chain fan-in: both chains present AND a combining arithmetic operation.
_PARALLEL_CHAIN_PHRASES = (
    "two independent chains",
)
_PARALLEL_COMBINE_PHRASES = (
    "absolute difference",
    "compute the final answer",
)

# Single-path dependency chain markers (no candidate-list structure). The first four
# are the original set, tuned against the two canonical examples (test_051, test_065);
# the rest widen coverage to the "sequential steps" dialect seen in test_023-style
# tasks. Deliberately excludes generic hop language ("then", "next", "above",
# "previous") on its own (those over-match parallel_merge/branch_eliminate tasks that
# also describe multi-step research informally); every phrase here names the
# step-to-step DEPENDENCY explicitly, not just sequencing.
_CHAIN_PHRASES = (
    "dependency chain",
    "research chain",
    "each step's answer is required to find the next",
    "each step can only be answered by reading the previous",
    "requires sequential steps",
    "this requires sequential",
    "each step depends on the previous",
    "using the answer from the previous step",
)


# Aggregation asks that the answer-shape markers below do not already cover: a plain
# "aggregate across all six ... earliest" carries no superlative from
# ``_ANSWER_SUPERLATIVES`` and no count/sum marker, yet is the canonical breadth ask.
# Measured over the 165 task modules in ``agent/app/idea_tests``: the breadth rule below
# claims 27 mandates, all of them fan-out-and-aggregate tasks, and none of them a mandate
# that already classifies as one of the other three shapes.
_BREADTH_AGGREGATE_PHRASES = (
    "aggregate",
    "across all",
    "earliest",
    "latest",
    "total",
    "sum of",
)


def _is_parallel_merge(text: str) -> bool:
    has_two_chains = any(p in text for p in _PARALLEL_CHAIN_PHRASES) or (
        "chain a" in text and "chain b" in text
    )
    has_combine = any(p in text for p in _PARALLEL_COMBINE_PHRASES)
    return has_two_chains and has_combine


def _is_branch_eliminate(mandate: str, text: str) -> bool:
    if len(extract_named_candidates(mandate)) < 2:
        return False
    return any(p in text for p in _DISAMBIG_PHRASES)


def _is_chain(mandate: str, text: str) -> bool:
    if not any(p in text for p in _CHAIN_PHRASES):
        return False
    # A real single-path chain has no enumerated candidate list.
    return extract_named_candidates(mandate) == []


def _is_breadth(mandate: str) -> bool:
    """True for an enumerated roster of names with an aggregation ask over the whole set.

    Two independent signals, both required:

    * the roster: ``parse_mandate_requirements`` already extracts name-like enumerated
      candidates and already detects the individual-disposition cues that mark a
      branch-eliminate ("exactly one", "eliminate", "survivor"). A roster WITHOUT those
      cues is a fan-out set, not an elimination set.
    * the aggregation ask, read from the surrounding prose only
      (``strip_enumerated_items``): an argmax phrasing, a count marker, or one of
      ``_BREADTH_AGGREGATE_PHRASES``. Scanning the whole mandate instead would let a
      per-item question ("5. ... how many species ...", task 034) or a per-item fact
      ("2. ... the earliest legal code ...", task 035) stand in for an aggregation the
      mandate never asks for.

    ``_is_argmax`` / ``_ANSWER_COUNT_MARKERS`` are defined in the answer-shape section
    below; the reuse is deliberate (one definition of "this asks for an extremum / a
    cardinality") and resolves at call time.
    """
    req = parse_mandate_requirements(mandate)
    if len(req.roster_candidates) < 2 or req.individual_disposition:
        return False
    prose = strip_enumerated_items(mandate).lower()
    return (
        _is_argmax(prose)
        or _has_any(prose, _ANSWER_COUNT_MARKERS)
        or _has_any(prose, _BREADTH_AGGREGATE_PHRASES)
    )


def classify_shape(mandate: str) -> Optional[str]:
    """Classify a mandate into one of the four known reasoning shapes, or ``None``.

    Fails OPEN: returns ``None`` for empty input and for any mandate that does not
    clearly match one of the recognised shapes, so callers can treat ``None`` as
    "apply no shape-specific discipline".
    """
    if not mandate:
        return None
    text = mandate.lower()
    # parallel_merge first: a two-chain task literally contains the word "chain".
    if _is_parallel_merge(text):
        return "parallel_merge"
    if _is_branch_eliminate(mandate, text):
        return "branch_eliminate"
    if _is_chain(mandate, text):
        return "chain"
    # breadth last: it may only claim mandates the three older shapes did not.
    if _is_breadth(mandate):
        return "breadth"
    return None


# ---------------------------------------------------------------------------
# Answer-shape classifier (finalize reconcile-chain gate)
# ---------------------------------------------------------------------------
#
# classify_answer_shape decides whether a mandate asks for a specific,
# extractable/derivable answer (computation, count, argmax/argmin, disambiguation
# survivor, or single factual value). This identifies tasks where post-synthesis
# recompute/verify/variation passes in idea_finalize can catch a "right page, wrong
# value" slip. Returns None (fail-open, the caller SKIPS the passes) for open-ended
# or narrative work where re-deriving a single value is meaningless and only burns tokens.
#
# It reuses classify_shape's labels where they already answer the question (a
# branch-eliminate IS a disambiguation, a parallel-merge IS a computation, a chain
# resolves to a single value), then falls back to conservative keyword heuristics.
# ``breadth`` is deliberately NOT mapped: a fan-out roster's answer may be an argmax, a
# count or a sum, so the label alone does not decide it and the keyword fallback below
# (which is what these mandates already went through) stays in charge.
# Deliberately conservative on the "run" side: an unrecognised phrasing returns
# None (skip) rather than guessing, so the passes only spend where the task is
# clearly answer-shaped.

# Disambiguation: pick the one specific item that satisfies a criterion.
_ANSWER_DISAMBIG_MARKERS = (
    "which of the",
    "exactly one of",
    "identify the specific",
    "the specific one",
    "disambiguate",
)

# Count: the answer is a cardinality.
_ANSWER_COUNT_MARKERS = (
    "how many",
    "number of",
    "count of",
    "count the",
    "count how",
)

# Computation: the answer is derived by an arithmetic operation.
_ANSWER_COMPUTATION_MARKERS = (
    "how much",
    "difference between",
    "the difference",
    "absolute difference",
    "sum of",
    "product of",
    "ratio of",
    "ratio between",
    "average of",
    "combined",
    "compute",
    "calculate",
    "multiply",
    "subtract",
    "divide",
    "add up",
    "percentage",
    "percent",
)

# Argmax/argmin: select the extremum among candidates. A superlative alone is NOT
# enough ("maximum depth" is an attribute *name*, not a selection over candidates);
# a selection word must co-occur so plain single-value attribute lookups stay
# ``single_value``.
_ANSWER_SUPERLATIVES = (
    "largest", "biggest", "highest", "deepest", "tallest", "longest", "greatest",
    "smallest", "lowest", "shortest", "oldest", "newest", "widest", "heaviest",
    "most", "least", "fewest", "nearest", "farthest", "furthest",
)
_ANSWER_SELECTION_WORDS = (
    "which", "among", "of the following", "of these", "of all", "identify", "rank",
)

# Single value: a specific factual scalar (a value, year, date, name, measurement).
_ANSWER_SINGLE_VALUE_MARKERS = (
    "what is the", "what was the", "what are the", "what year", "in what year",
    "in which year", "on what date", "report the", "give the number", "give the exact",
    "state the", "name the", "find the", "determine the", "how deep", "how tall",
    "how long", "how high", "how old", "how far", "how wide", "how heavy",
    "when did", "when was", "who is", "who was", "where is", "where was",
)


def _has_any(text: str, markers) -> bool:
    return any(m in text for m in markers)


def _is_argmax(text: str) -> bool:
    return _has_any(text, _ANSWER_SELECTION_WORDS) and _has_any(text, _ANSWER_SUPERLATIVES)


def classify_answer_shape(mandate: str) -> Optional[str]:
    """Classify a mandate into an answer shape, or None for open-ended/narrative.

    Returns one of "computation", "count", "argmax", "disambiguation",
    "single_value" when the task asks for a specific derivable answer, else None.
    Fails OPEN toward None (skip) for anything not clearly answer-shaped.
    """
    if not mandate:
        return None
    base = classify_shape(mandate)
    if base == "branch_eliminate":
        return "disambiguation"
    if base == "parallel_merge":
        return "computation"
    if base == "chain":
        return "single_value"

    text = mandate.lower()
    # argmax before disambiguation: "which of these ... deepest" is a selection-over-extremum, and
    # a disambiguation marker like "which of the" is a substring of "which of these".
    if _is_argmax(text):
        return "argmax"
    if _has_any(text, _ANSWER_DISAMBIG_MARKERS):
        return "disambiguation"
    if _has_any(text, _ANSWER_COUNT_MARKERS):
        return "count"
    if _has_any(text, _ANSWER_COMPUTATION_MARKERS):
        return "computation"
    if _has_any(text, _ANSWER_SINGLE_VALUE_MARKERS):
        return "single_value"
    return None
