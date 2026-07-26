"""
Deterministic (non-LLM) task-shape classifier over a mandate's text.

This is a deliberately small first draft. It recognises three reasoning shapes that
have appeared in the tier-4/5 corpus and returns ``None`` (fails OPEN) for anything
else, so it can never impose the wrong discipline on an unrecognised task:

* ``"branch_eliminate"`` — the mandate enumerates K similarly-named candidates and a
  single distinguishing criterion, and asks the agent to eliminate down to one
  survivor (e.g. ``test_095``: four Rivers Avon, exactly one empties into the English
  Channel). This is the ONLY shape with matching infrastructure today: the candidate
  list is detected with ``candidate_coverage.extract_named_candidates`` and there is a
  real rule file at ``reasoning_rules/branch_eliminate.md``.

* ``"chain"`` — a single dependency chain where each step's answer is needed to find
  the next page, with NO candidate-list structure (e.g. ``test_051``, ``test_065``).
  Keyword heuristic only; no matching rule file exists yet.

* ``"parallel_merge"`` — two independent chains whose results are combined by an
  arithmetic operation (e.g. ``test_055``, ``test_061``: absolute difference of two
  founding years / birth years). Keyword heuristic only; no matching rule file yet.

Only ``branch_eliminate`` reuses real supporting infrastructure; ``chain`` and
``parallel_merge`` are best-effort keyword detectors and are intentionally less
rigorous. Order matters: ``parallel_merge`` is checked before ``chain`` because a
two-chain task literally contains the word "chain".
"""

from __future__ import annotations

from typing import Optional

from agent.app.idea_policies.candidate_coverage import extract_named_candidates


# Disambiguation-style language that, combined with an enumerated candidate list,
# signals a branch-eliminate shape (as opposed to a plain breadth/fan-out list that
# also happens to be numbered — e.g. test_052's six novels, test_059's five players).
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

# Single-path dependency chain markers (no candidate-list structure).
_CHAIN_PHRASES = (
    "dependency chain",
    "research chain",
    "each step's answer is required to find the next",
    "each step can only be answered by reading the previous",
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


def classify_shape(mandate: str) -> Optional[str]:
    """Classify a mandate into one of the three known reasoning shapes, or ``None``.

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
    return None


# ---------------------------------------------------------------------------
# Answer-shape classifier (finalize reconcile-chain gate)
# ---------------------------------------------------------------------------
#
# ``classify_answer_shape`` decides whether a mandate asks for a *specific,
# extractable/derivable answer* (a computation, a count, an argmax/argmin, a
# disambiguation survivor, or a single factual value) — the tasks where the
# post-synthesis recompute/verify/variation passes in ``idea_finalize`` can catch a
# "right page, wrong value" slip. It returns ``None`` (fail-open → the caller SKIPS
# the passes) for open-ended / narrative work where re-deriving a single value is
# meaningless and only burns tokens.
#
# It reuses ``classify_shape``'s labels where they already answer the question (a
# branch-eliminate IS a disambiguation, a parallel-merge IS a computation, a chain
# resolves to a single value), then falls back to conservative keyword heuristics.
# Deliberately conservative on the "run" side: an unrecognised phrasing returns
# ``None`` (skip) rather than guessing, so the passes only spend where the task is
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
    """Classify a mandate into an *answer shape*, or ``None`` for open-ended/narrative.

    Returns one of ``"computation"``, ``"count"``, ``"argmax"``, ``"disambiguation"``,
    ``"single_value"`` when the task asks for a specific derivable answer, else ``None``.
    Fails OPEN toward ``None`` (skip) for anything not clearly answer-shaped.
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
