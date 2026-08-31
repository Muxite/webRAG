"""Arm-symmetric ANSWER / PARTIAL / ABSTAIN verdict, derivable for ANY execution variant.

The problem, verified in code: only ``evidence_loop`` emits a verdict at all --
``execution.output.ledger_verdict``, derived by ``Ledger.verdict()``
(``testing/execution_evidence_loop.py``). ``sequential_react_extract`` writes only
``success = bool(deliverable)`` (its own docstring: "no ledger, no row minting, no verdict").
``langgraph_react`` writes ``final_deliverable`` / ``success`` / ``goal_achieved`` /
``action_summary`` / ``warning`` -- no verdict at all. Comparing a real verdict against
``success=True`` is not a comparison: on one stored run ``sequential_react_extract`` reported
``success=True`` on 48 of 48 cells.

:func:`derive_verdict` computes a verdict from ONLY what every arm actually emits: the final
answer text (``idea_test_utils.extract_final_text``) and the arm-symmetric evidence source
(``idea_test_utils.visited_evidence``, which already fixed this exact class of "arm-comparison
artifact, not a capability difference" bug for tasks 046/047 by refusing to source evidence from
``result["graph"]`` alone). It deliberately never reads ``ledger_verdict``, ``success`` or
``goal_achieved`` -- so a disagreement with ``ledger_verdict`` on an ``evidence_loop`` cell is a
measurable finding, not something papered over by deferring to the native field.

Two constraints this design works within, both verified in code:

* ``citation_echo`` telemetry is wired ONLY into ``IdeaDagEngine.finalize`` (``idea_engine.py``),
  i.e. the ``graph`` / ``sequential`` / ``naive_discretion`` arms. It is not available on
  ``evidence_loop``, ``sequential_react_extract`` or ``langgraph_react``, so it cannot be an
  arm-symmetric signal and this module does not use it.
* The arms' output contracts differ (see above), so the only fields read here --
  ``final_deliverable`` and whatever ``visited_evidence`` can recover -- are the ones every arm
  actually has.

Verdict rule (deterministic, identical for every arm)
------------------------------------------------------
1. No visited evidence at all -> :data:`VERDICT_ABSTAIN`. Nothing was gathered to ground
   anything the model might say, so the answer -- whatever it claims -- cannot be trusted.
2. Empty final text, or final text matching one of a small set of explicit non-answer phrases
   (``"could not determine"``, ``"unable to find"``, ``"insufficient evidence"``, ...) ->
   :data:`VERDICT_ABSTAIN`. This is the only case that reads the text for its own sake rather
   than for claims within it.
3. Otherwise, extract the text's CHECKABLE claims: standalone numeric tokens (>= 2 digits, to
   keep single-digit noise like a ledger table's row index from posing as a claim) and
   double-quoted spans. Each claim is checked for a literal (case-insensitive) match in the
   concatenation of every visited page's text (``idea_test_utils.evidence_text``).

   * No checkable claims at all -> :data:`VERDICT_PARTIAL`. Evidence was gathered and the model
     said something, but nothing in it is mechanically checkable -- worth reporting, not an
     abstention, and not creditable as a verified ANSWER either.
   * Every claim is grounded -> :data:`VERDICT_ANSWER`.
   * Some or none of the claims are grounded -> :data:`VERDICT_PARTIAL`. An unsupported number
     asserted against real, gathered evidence is a reportable (possibly fabricated) claim, not
     "we obtained nothing" -- that distinction is exactly what :data:`VERDICT_ABSTAIN` is reserved
     for by :meth:`Ledger.verdict`'s own definition ("no row obtained anything at all"), which
     this rule mirrors on purpose.

Known limitation, stated plainly rather than hidden: this can only ground a claim against
evidence :func:`idea_test_utils.visited_evidence` can actually recover. For ``langgraph_react``
and plain ``sequential_react``, the stored corpus in ``agent/idea_test_results`` carries NEITHER
``observability["evidence"]`` (a validation-time-only projection, never persisted --
``testing/runner.py``) NOR a populated ``result["graph"]`` NOR an ``output["pages"]`` freeze (only
``evidence_loop`` / ``sequential_react_extract`` emit that). Re-scoring those two arms' stored
cells therefore derives :data:`VERDICT_ABSTAIN` unconditionally, regardless of the true score --
a genuine structural gap in what got persisted, not a defect in this rule. See the module's test
suite and the risk-coverage tooling built on top of it for how this shows up in practice.
"""
from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Optional

from agent.app.idea_test_utils import extract_final_text, evidence_text, visited_evidence

VERDICT_ANSWER = "ANSWER"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_ABSTAIN = "ABSTAIN"

# Explicit, natural-language non-answer phrasing. Deliberately narrow: this is NOT a fuzzy
# "sounds unsure" detector, and it deliberately does not include ledger jargon like "ABSTAIN" or
# "VERDICT:" -- evidence_loop's own decorated deliverable embeds a literal "VERDICT: ABSTAIN"
# line, and matching that string here would make this module's ABSTAIN for evidence_loop trivially
# equal to the native ledger_verdict it exists to check independently, defeating the point.
_ABSTENTION_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bcannot (?:be )?determin\w*\b",
    r"\bcould not (?:find|determine|locate|obtain|verify)\b",
    r"\bunable to (?:find|determine|locate|answer|obtain)\b",
    r"\bno (?:information|data|evidence|sources?)\s+(?:was|were)?\s*(?:found|available)\b",
    r"\binsufficient (?:evidence|information|data)\b",
    r"\bnot (?:enough|sufficient) (?:information|evidence)\b",
    r"\bi don't know\b",
    r"\bi cannot answer\b",
))

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_QUOTE_RE = re.compile(r'"([^"]{4,120})"')
# A bare numeric token below this many digits (commas/sign/decimal point stripped) is treated as
# noise (row indices, list bullets, single-digit counts embedded in surrounding table formatting)
# rather than a checkable claim.
_MIN_CLAIM_DIGITS = 2


def _is_abstention_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in _ABSTENTION_PATTERNS)


def _claims(text: str) -> FrozenSet[str]:
    """Standalone numeric tokens and double-quoted spans in ``text``, normalized for matching."""
    claims = set()
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        digits_only = token.replace(",", "").replace(".", "").lstrip("-")
        if len(digits_only) >= _MIN_CLAIM_DIGITS:
            claims.add(token.replace(",", ""))
    for match in _QUOTE_RE.finditer(text):
        quoted = match.group(1).strip()
        if quoted:
            claims.add(quoted.lower())
    return frozenset(claims)


def _claim_grounded(claim: str, corpus: str) -> bool:
    return bool(claim) and claim.lower() in corpus


def derive_verdict(result: Dict[str, Any], observability: Optional[Dict[str, Any]] = None) -> str:
    """ANSWER / PARTIAL / ABSTAIN for ANY execution variant, from text + evidence alone.

    :param result: Test result payload (``{"output": ..., "graph": ...}`` or the raw
        ``execution`` block of a stored cell -- either shape works, since only
        ``extract_final_text`` and ``visited_evidence`` read it).
    :param observability: Observability payload; see :func:`idea_test_utils.visited_evidence`.
    :return: One of :data:`VERDICT_ANSWER` / :data:`VERDICT_PARTIAL` / :data:`VERDICT_ABSTAIN`.
    """
    evidence = visited_evidence(result, observability)
    if not evidence:
        return VERDICT_ABSTAIN

    text = extract_final_text(result).strip()
    if not text or _is_abstention_text(text):
        return VERDICT_ABSTAIN

    claims = _claims(text)
    if not claims:
        return VERDICT_PARTIAL

    corpus = evidence_text(result, observability)
    if all(_claim_grounded(claim, corpus) for claim in claims):
        return VERDICT_ANSWER
    return VERDICT_PARTIAL
