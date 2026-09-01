"""Arm-blind claim auditing: the same evidence record, reconstructed for any execution variant.

Every KPI in ``docs/LEDGER_KPI_SPEC.md`` that compares arms is computed from this module, and it
is written so that it CANNOT see which arm produced a cell. That is not stylistic. This repo has
twice published an arm comparison that measured an artifact of its own instrumentation rather
than the arms -- tasks 046/047, where graph-only adjacency pinned the non-graph arms at 0.000,
and the redundancy figure that read 0% for ``langgraph_react`` only because
``execution_langgraph.pages_from_telemetry`` deduplicates ``output.pages`` at write time. An
auditor that never learns the arm cannot reproduce that class of error.

The single adapter :func:`audit_input` is the only function here that touches a stored cell.
Everything downstream consumes :class:`AuditInput`, which carries no arm identifier and no
arm-exclusive field.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.app.testing.arm_verdict import _claims
from agent.app.idea_test_utils import normalize_url
from agent.app.testing.evidence_graph import verify_value_against_stored_page

#: A claim located verbatim on a page the arm stored.
ON_PAGE = "on_page"
#: A claim no page carries, reproduced by one whitelisted operation over `on_page` operands.
RECOMPUTABLE = "recomputable"
#: A claim that is neither.
UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class AuditInput:
    """What every arm demonstrably emits, and nothing else.

    :param final_text: the answer the arm delivered.
    :param pages: the pages the arm froze into its result, each ``{url, text, ...}``.
    :param visits: one entry per visit EVENT, in order, including repeats.
    """

    final_text: str
    pages: Tuple[Mapping[str, Any], ...]
    visits: Tuple[str, ...]


def _execution(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    """The execution block, accepting either a whole stored cell or the block itself."""
    block = cell.get("execution")
    return block if isinstance(block, Mapping) else cell


def _visit_events(execution: Mapping[str, Any]) -> List[str]:
    """Every visit URL in order, from telemetry -- never from ``output.pages``.

    ``output.pages`` is deduplicated at write time by ``langgraph_react`` and appended per visit
    by the other two arms, so it counts different things per arm and can never be a shared
    denominator. ``agent_io.visit`` records one timing per visit for every arm.
    """
    raw = execution.get("telemetry_raw")
    timings = raw.get("timings") if isinstance(raw, Mapping) else None
    urls: List[str] = []
    for timing in timings or []:
        if not isinstance(timing, Mapping) or timing.get("name") != "visit":
            continue
        payload = timing.get("payload")
        url = str((payload or {}).get("url") or "").strip() if isinstance(payload, Mapping) else ""
        if url:
            urls.append(url)
    return urls


def audit_input(cell: Mapping[str, Any]) -> AuditInput:
    """Project a stored cell onto the arm-symmetric surface.

    :param cell: a stored result JSON, or its ``execution`` block.
    :returns: the :class:`AuditInput` every downstream metric reads.
    :raises: nothing -- a cell missing a field yields an empty one, never an exception.
    """
    execution = _execution(cell)
    output = execution.get("output")
    output = output if isinstance(output, Mapping) else {}
    pages = tuple(p for p in (output.get("pages") or []) if isinstance(p, Mapping))
    return AuditInput(
        final_text=str(output.get("final_deliverable") or ""),
        pages=pages,
        visits=tuple(_visit_events(execution)),
    )


#: A number on a page, with whatever unit token sat immediately after it. The unit is kept as
#: written -- it is only ever compared for EQUALITY, never converted, per `LEDGER_PLAN` section 7.
_OPERAND_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*([A-Za-z\u00b0%$\u20ac\u00a3]+)?")

_SPEC_PATH = Path(__file__).resolve().parents[3] / "scripts" / "ledger_kpi_spec.json"

#: The six operations the numeric suite's tasks actually use. Frozen in the spec; a seventh
#: operation would widen the search and raise the coincidental-match floor, so adding one is an
#: amendment to a frozen document, not a code change.
_OPERATIONS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "abs_difference": lambda a, b: abs(a - b),
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b if b else None,
    "share_of_total": lambda a, b: a / (a + b) if (a + b) else None,
}


@lru_cache(maxsize=1)
def load_spec() -> Mapping[str, Any]:
    """The frozen KPI specification. No threshold in this module is a literal.

    `agent/tests/ledger_kpi_spec_frozen_test.py` hashes this file, so a tolerance or operation
    set cannot move without the change appearing in a diff.
    """
    return json.loads(_SPEC_PATH.read_text())


def _operands(source: AuditInput, limit: int, *,
              stated_only: bool = True) -> List[Tuple[float, str]]:
    """Numbers written on the pages the arm stored, each with its literal unit token.

    Two bounds, both in the frozen spec because both are properties of the METRIC rather than
    implementation details -- widening either raises the rate at which the pair search reproduces
    a target by luck:

    * ``limit`` caps how many candidates the search sees.
    * ``stated_only`` keeps only operands the ANSWER itself states. Measured on the tuning split,
      dropping this constraint pushes the cross-cell coincidence floor from 0.0138 to 0.2526 --
      five times the ceiling, and above the genuine recomputable rate, i.e. the class becomes
      noise. It is also the substantively right rule: a real derivation is reported with its
      inputs.
    """
    stated = {c.replace(",", "") for c in _claims(source.final_text)} if stated_only else None
    found: List[Tuple[float, str]] = []
    seen: set = set()
    for page in source.pages:
        for raw, unit in _OPERAND_RE.findall(str(page.get("text") or "")):
            digits = raw.replace(",", "")
            if stated is not None and digits not in stated:
                continue
            try:
                value = float(digits)
            except ValueError:
                continue
            key = (value, (unit or "").lower())
            if key in seen:
                continue
            seen.add(key)
            found.append((value, (unit or "").lower()))
            if len(found) >= limit:
                return found
    return found


def _recomputes(target: float, operands: Sequence[Tuple[float, str]], tolerance: float) -> bool:
    """True when one whitelisted operation over two same-unit operands reproduces ``target``.

    Operands must share a unit token exactly. Nothing here converts between dimensions -- a
    mismatched pair is simply ineligible, because the unit-mismatch tasks (222-224) exist to
    verify that a refusal happens and an auditor that converted would score them by committing
    the error under test.
    """
    for i, (a, unit_a) in enumerate(operands):
        for j, (b, unit_b) in enumerate(operands):
            if i == j or unit_a != unit_b:
                continue
            for operate in _OPERATIONS.values():
                try:
                    result = operate(a, b)
                except (ZeroDivisionError, ArithmeticError):
                    continue
                if result is None:
                    continue
                scale = max(abs(target), 1e-9)
                if abs(result - target) / scale <= tolerance:
                    return True
    return False


def classify_claims(source: AuditInput) -> Dict[str, str]:
    """Classify every checkable claim in the answer as :data:`ON_PAGE` or :data:`UNSUPPORTED`.

    Claim extraction reuses ``arm_verdict._claims`` (numeric tokens of >= 2 digits and quoted
    spans) so the auditor and the verdict rule can never disagree about what a claim IS.
    Location reuses ``evidence_graph.verify_value_against_stored_page`` rather than a substring
    test, so a truncated page yields *unverifiable* rather than a false ``absent``, and a scale
    word is fully applied instead of being dropped as an opaque suffix.

    :param source: the arm-symmetric projection of a cell.
    :returns: ``{claim: classification}``.
    :raises: nothing.
    """
    spec = load_spec()["support"]
    tolerance = float(spec["relative_tolerance"])
    operands = _operands(source, int(spec["max_operands_per_cell"]),
                         stated_only=bool(spec["operands_must_appear_in_answer"]))

    classified: Dict[str, str] = {}
    for claim in sorted(_claims(source.final_text)):
        classified[claim] = UNSUPPORTED
        for page in source.pages:
            if verify_value_against_stored_page(dict(page), claim).verified:
                classified[claim] = ON_PAGE
                break
        if classified[claim] == ON_PAGE:
            continue
        try:
            target = float(str(claim).replace(",", ""))
        except ValueError:
            continue  # a quoted span is not a quantity; it can only ever be on_page
        if _recomputes(target, operands, tolerance):
            classified[claim] = RECOMPUTABLE
    return classified


def _repeat_rate(visits: Sequence[str]) -> Optional[float]:
    """Share of visit events that re-read a URL already read in this cell, or None if it visited
    nothing. None is not zero: an arm that never visited has no redundancy to report."""
    if not visits:
        return None
    normalized = [normalize_url(u) for u in visits]
    return 1.0 - (len(set(normalized)) / len(normalized))


def audit(cell: Mapping[str, Any]) -> Dict[str, Any]:
    """The full arm-symmetric evidence record for one stored cell.

    Reads only what :func:`audit_input` projects, so the result is invariant under deletion of
    every arm-exclusive field -- the property `claim_audit_test` asserts directly.

    :param cell: a stored result JSON, or its ``execution`` block.
    :returns: the record every L2/L4/L8 number is computed from. Counts are integers; rates are
        floats, or None where the denominator is empty (absent is never zero).
    :raises: nothing.
    """
    source = audit_input(cell)
    classified = classify_claims(source)
    counts = {
        ON_PAGE: sum(1 for v in classified.values() if v == ON_PAGE),
        RECOMPUTABLE: sum(1 for v in classified.values() if v == RECOMPUTABLE),
        UNSUPPORTED: sum(1 for v in classified.values() if v == UNSUPPORTED),
    }
    checkable = len(classified)
    supported = counts[ON_PAGE] + counts[RECOMPUTABLE]
    return {
        "claims": classified,
        "counts": counts,
        "checkable_claims": checkable,
        # None, not 0.0: an answer with nothing checkable in it has not earned a perfect score,
        # and folding it in as 0 would reward exactly the claim-poor answers this phase found the
        # old verdict rule was already selecting for.
        "support_rate": (supported / checkable) if checkable else None,
        "unsupported_rate": (counts[UNSUPPORTED] / checkable) if checkable else None,
        "visits": len(source.visits),
        "distinct_visits": len({normalize_url(u) for u in source.visits}),
        "repeat_visit_rate": _repeat_rate(source.visits),
        "pages_stored": len(source.pages),
    }


def recomputable_against_foreign_pages(answer_cell: Mapping[str, Any],
                                       foreign_cell: Mapping[str, Any]) -> Dict[str, bool]:
    """Would this cell's claims be 'recomputed' from an UNRELATED cell's pages?

    The coincidence control for :data:`RECOMPUTABLE`. A search over pairs drawn from a couple of
    dozen page numbers can reproduce a target by accident, and a support metric that never
    measured that floor would report its own arithmetic luck as evidence. The frozen spec
    requires the resulting rate to be printed beside every claim-support number.

    Uses the same operands, operations and tolerance as :func:`classify_claims`, because a
    control computed under looser settings would understate the floor it exists to expose.

    :param answer_cell: the cell whose stated claims are tested.
    :param foreign_cell: a cell from a DIFFERENT task, supplying the operands.
    :returns: ``{claim: True}`` when the foreign pages reproduce it by coincidence.
    :raises: nothing.
    """
    spec = load_spec()["support"]
    tolerance = float(spec["relative_tolerance"])
    answer = audit_input(answer_cell)
    foreign = audit_input(foreign_cell)
    operands = _operands(
        AuditInput(final_text=answer.final_text, pages=foreign.pages, visits=()),
        int(spec["max_operands_per_cell"]),
        stated_only=bool(spec["operands_must_appear_in_answer"]),
    )

    coincidences: Dict[str, bool] = {}
    for claim in sorted(_claims(answer.final_text)):
        try:
            target = float(str(claim).replace(",", ""))
        except ValueError:
            continue
        coincidences[claim] = _recomputes(target, operands, tolerance)
    return coincidences
