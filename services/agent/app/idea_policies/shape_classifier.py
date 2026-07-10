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
