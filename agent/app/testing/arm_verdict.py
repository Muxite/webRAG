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
evidence :func:`idea_test_utils.visited_evidence` can actually recover. For plain
``sequential_react``, the stored corpus in ``agent/idea_test_results`` carries NEITHER
``observability["evidence"]`` (a validation-time-only projection, never persisted --
``testing/runner.py``) NOR a populated ``result["graph"]`` NOR an ``output["pages"]`` freeze, so
re-scoring its stored cells derives :data:`VERDICT_ABSTAIN` unconditionally, regardless of the
true score -- a genuine structural gap in what got persisted, not a defect in this rule.
``langgraph_react`` no longer has this gap: commit 2cdc9066 made it persist ``output["pages"]``,
and ``ledgernum22r3``'s ``langgraph_react`` cells carry it, so this rule can and does ground
claims against langgraph's visited evidence like any other arm. See the module's test suite and
the risk-coverage tooling built on top of it for how this shows up in practice.

Measured structural limitation of the literal-match rule itself (not the persistence gap above),
reproduced over all 198 scorable cells of the ``ledgernum22r3`` campaign: step 3's ALL-claims-
grounded requirement for :data:`VERDICT_ANSWER` systematically SELECTS AGAINST correct arithmetic
answers on this suite's numeric tasks. The DERIVED-arithmetic tasks are 210-221 (sum/difference,
ratio/quotient, computed-ratio argmax) and they are deliberately leak-proofed -- the keystone is
constructed so it never appears verbatim on any visited page -- so a literal-match rule can never
credit a correct derived number; it can only credit an arm that happened to produce no checkable
derived claim at all. The data confirms this: 28 of the 29 ANSWER-tier cells in the campaign come
from 222-231, the NON-derived tasks (unit-mismatch refusal, missing-operand abstention,
plausible-but-unsupported numeric), and exactly ONE comes from the whole 210-221 derived range.
ANSWER-tier cells score WORSE on
average than PARTIAL-tier cells from the same arm -- ``langgraph_react`` ANSWER mean 0.362 vs its
own PARTIAL mean 0.705; ``sequential_react_extract`` ANSWER mean 0.379 vs PARTIAL mean 0.574;
``evidence_loop`` ANSWER mean 0.562 vs PARTIAL mean 0.548. :data:`VERDICT_ANSWER` under this rule
therefore does not mean "the answer was verified correct" on the numeric suite -- it means "the
answer's claims happened to be literally quotable from a page," which anti-correlates with
producing the actually-correct derived number. A replacement rule (one that can credit a
mechanically-verified DERIVATION, not just a literal page quote -- see ``evidence_graph.py`` and
:func:`scripts.claim_metrics.derivation_fabrication_rate`) is planned separately; this paragraph
only records the finding, it does not change :func:`derive_verdict`.
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


def derive_verdict_graded(result: Dict[str, Any], observability: Optional[Dict[str, Any]] = None) -> str:
    """ANSWER / PARTIAL / ABSTAIN, delegating claim support to the arm-blind auditor.

    Unlike :func:`derive_verdict`, this credits a claim that is :data:`claim_audit.RECOMPUTABLE`
    -- a value reproduced by one whitelisted operation over operands the answer itself states --
    not only a claim located verbatim on a visited page. That is the fix for the structural
    defect this module's docstring measures: the numeric suite's derived tasks are leak-proofed,
    so a correct derived keystone can never be :data:`claim_audit.ON_PAGE`, and the old
    literal-match rule could therefore never reach :data:`VERDICT_ANSWER` for one.

    ``derive_verdict`` itself is left completely unchanged by this function's existence -- the
    old rule stays computable so every report can show old-vs-new side by side.

    Rule (every threshold read from ``scripts/ledger_kpi_spec.json``'s ``graded_verdict`` block,
    never a literal in this function -- that file is hash-frozen):

    1. :data:`VERDICT_ABSTAIN` if no pages were stored, the answer has no checkable claim, or the
       text matches one of :data:`_ABSTENTION_PATTERNS` (the same patterns
       :func:`derive_verdict` uses -- reused, not reimplemented).
    2. :data:`VERDICT_ANSWER` if the auditor's ``support_rate`` meets ``answer_min_support`` AND
       at least one claim is supported (``on_page`` or ``recomputable``).
    3. :data:`VERDICT_PARTIAL` otherwise.

    :param result: same shape :func:`derive_verdict` accepts -- ``{"output": ..., "graph": ...}``
        or a raw ``execution`` block; only ``output`` (``final_deliverable`` / ``pages``) is read,
        since :func:`claim_audit.audit` is arm-blind by construction.
    :param observability: accepted for signature parity with :func:`derive_verdict`; unused, since
        :func:`claim_audit.audit` deliberately reads only ``output.pages``, never the
        multi-source ``visited_evidence`` fallback -- see ``claim_audit.py``'s module docstring.
    :return: One of :data:`VERDICT_ANSWER` / :data:`VERDICT_PARTIAL` / :data:`VERDICT_ABSTAIN`.
    """
    # Local import: claim_audit imports `_claims` from this module at load time, so a top-level
    # import here would be circular.
    from agent.app.testing.claim_audit import ON_PAGE, RECOMPUTABLE, audit, load_spec

    text = extract_final_text(result).strip()
    if not text or _is_abstention_text(text):
        return VERDICT_ABSTAIN

    record = audit(result)
    if record["pages_stored"] == 0:
        return VERDICT_ABSTAIN
    if record["checkable_claims"] == 0:
        return VERDICT_ABSTAIN

    spec = load_spec()["graded_verdict"]
    supported = record["counts"][ON_PAGE] + record["counts"][RECOMPUTABLE]
    support_rate = record["support_rate"] or 0.0
    if support_rate >= float(spec["answer_min_support"]) and (
        supported >= 1 or not bool(spec["answer_requires_at_least_one_supported_claim"])
    ):
        return VERDICT_ANSWER
    return VERDICT_PARTIAL
