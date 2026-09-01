"""The arm-blind claim auditor.

Every test here exists because the alternative produced a wrong number in this project's own
history; the docstrings name which one.
"""
from __future__ import annotations

import pytest

from agent.app.testing.claim_audit import audit_input


def _cell(*, final_text: str = "", pages=None, visit_urls=None, **extra) -> dict:
    """A stored-cell shape with only the fields the auditor is allowed to read, plus `extra`."""
    cell = {
        "execution": {
            "output": {"final_deliverable": final_text, "pages": list(pages or [])},
            "telemetry_raw": {
                "timings": [
                    {"name": "visit", "payload": {"url": u}} for u in (visit_urls or [])
                ],
            },
        },
    }
    cell["execution"]["output"].update(extra)
    return cell


def test_visits_come_from_telemetry_not_from_the_stored_pages():
    """`output.pages` is deduplicated at write time by one arm, so it cannot count visits.

    `execution_langgraph.pages_from_telemetry` skips a URL already in its `seen` set. Counting
    re-reads from `output.pages` therefore reports 0% redundancy for `langgraph_react` -- a
    persistence artifact that this session measured and had to retract. The visit events in
    `telemetry_raw.timings` are the arm-symmetric truth.
    """
    cell = _cell(
        pages=[{"page_id": "p1", "url": "https://example.com/a", "text": "x"}],
        visit_urls=["https://example.com/a", "https://example.com/a", "https://example.com/b"],
    )

    result = audit_input(cell)

    assert len(result.visits) == 3
    assert len(set(result.visits)) == 2


def test_a_number_printed_on_a_visited_page_is_supported_as_on_page():
    from agent.app.testing.claim_audit import ON_PAGE, classify_claims

    cell = _cell(
        final_text="The chimney is 419.7 metres tall.",
        pages=[{"page_id": "p1", "url": "u", "text": "Its chimney reaches 419.7 metres."}],
    )

    classified = classify_claims(audit_input(cell))

    assert classified["419.7"] == ON_PAGE


def test_a_number_on_no_page_and_derivable_from_nothing_is_unsupported():
    from agent.app.testing.claim_audit import UNSUPPORTED, classify_claims

    cell = _cell(
        final_text="The tower is 1234.5 metres tall.",
        pages=[{"page_id": "p1", "url": "u", "text": "The article mentions 42 metres."}],
    )

    classified = classify_claims(audit_input(cell))

    assert classified["1234.5"] == UNSUPPORTED


def test_a_derived_answer_absent_from_every_page_is_recomputable_from_on_page_operands():
    """The defect this class exists to fix, measured on `ledgernum22r3`.

    The suite is leak-proofed, so a DERIVED keystone never appears verbatim on any page. Under a
    pure on-page rule, 28 of 29 ANSWER-tier cells came from the non-arithmetic tasks 222-231, and
    `langgraph_react`'s ANSWER tier scored 0.362 against its own PARTIAL tier's 0.712 -- the tier
    was selecting "this arm produced no derived number". Crediting a value that one whitelisted
    operation reproduces from located operands is what stops the metric measuring our own
    leak-proofing.
    """
    from agent.app.testing.claim_audit import ON_PAGE, RECOMPUTABLE, classify_claims

    cell = _cell(
        final_text="Tower A is 419.7 metres and Tower B is 330.0 metres, so the difference "
                   "is 89.7 metres.",
        pages=[{"page_id": "p1", "url": "u",
                "text": "Tower A stands 419.7 metres. Tower B stands 330.0 metres."}],
    )

    classified = classify_claims(audit_input(cell))

    assert classified["89.7"] == RECOMPUTABLE
    # the operands are credited on their own terms, not as part of the derivation
    assert classified["419.7"] == ON_PAGE
    assert classified["330.0"] == ON_PAGE


def test_operands_carrying_incompatible_units_are_never_combined():
    """`LEDGER_PLAN` section 7: no unit or currency conversion, ever.

    The unit-mismatch tasks (222-224) exist to verify that a refusal happens. If the auditor
    combined across dimensions it would score those tasks by committing the very error they test
    for.
    """
    from agent.app.testing.claim_audit import UNSUPPORTED, classify_claims

    cell = _cell(
        final_text="The combined figure is 530.0.",
        pages=[{"page_id": "p1", "url": "u",
                "text": "It is 200.0 km long and costs 330.0 USD."}],
    )

    classified = classify_claims(audit_input(cell))

    assert classified["530.0"] == UNSUPPORTED


ARM_EXCLUSIVE_FIELDS = (
    "ledger", "ledger_verdict", "ledger_status_counts", "ledger_resolution_counts",
    "extractions", "evidence_graph", "derivation_gate", "derivation_validity",
    "derivation_refusals", "quote_verified_count", "quote_unverified_count",
    "quote_unchecked_count", "quote_fail_reasons", "roster_named", "roster_resolved",
    "roster_rows", "roster_truncated", "roster_complete", "roster_gate", "tool_transport",
    "rows_resolved", "rows_quote_backed", "rows_unresolved", "unverified_provenance",
)


def test_the_audit_is_identical_when_every_arm_exclusive_field_is_removed():
    """The arm-blindness guarantee, asserted rather than promised.

    This repo has twice shipped an "arm comparison" that measured its own instrumentation:
    tasks 046/047, where graph-only adjacency pinned the non-graph arms at 0.000, and the
    redundancy figure that read 0% for one arm because its pages are deduplicated at write time.
    If deleting every field only one arm emits changes the audit, the audit is measuring the arm's
    paperwork instead of its behaviour.
    """
    from agent.app.testing.claim_audit import audit

    cell = _cell(
        final_text="The difference is 89.7 metres, from 419.7 and 330.0.",
        pages=[{"page_id": "p1", "url": "u",
                "text": "Tower A stands 419.7 metres. Tower B stands 330.0 metres."}],
        visit_urls=["https://example.com/a", "https://example.com/a"],
        # everything below is arm-exclusive paperwork the auditor must not read
        ledger=[{"entity": "e", "value": "89.7"}],
        ledger_verdict="ANSWER",
        evidence_graph={"nodes": [{"kind": "source", "value": "419.7"}]},
        extractions=[{"value": "419.7", "value_verified": True}],
        tool_transport="native",
    )
    redacted = _cell(
        final_text="The difference is 89.7 metres, from 419.7 and 330.0.",
        pages=[{"page_id": "p1", "url": "u",
                "text": "Tower A stands 419.7 metres. Tower B stands 330.0 metres."}],
        visit_urls=["https://example.com/a", "https://example.com/a"],
    )

    assert audit(cell) == audit(redacted)


def test_the_auditor_never_names_an_arm_exclusive_field_in_its_own_source():
    """A field the auditor cannot name, it cannot come to depend on.

    The redaction test above only catches a leak for fields present in its fixture. This catches
    the next one someone adds.
    """
    import ast
    import inspect

    from agent.app.testing import claim_audit

    tree = ast.parse(inspect.getsource(claim_audit))
    # A cell field is reached through a STRING LITERAL (`output["ledger"]`, `.get("ledger")`).
    # A module path is a bare identifier, so importing `...evidence_graph` for its verification
    # helpers is not a leak -- reading a cell's `"evidence_graph"` key would be.
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    for field in ("ledger", "ledger_verdict", "evidence_graph", "extractions",
                  "execution_variant", "tool_transport", "tooling_profile"):
        assert field not in literals, (
            f"claim_audit reads the arm-exclusive cell field {field!r}; the audit would then "
            "measure one arm's paperwork rather than every arm's behaviour"
        )


def test_recomputability_is_not_credited_when_the_operands_come_from_another_cell():
    """The control that makes `recomputable` reportable at all.

    A pair search over two dozen numbers can hit a target by coincidence. The frozen spec
    therefore requires every L2/L4 table to print the rate at which claims are 'recomputed' from
    a DIFFERENT cell's pages -- the metric's own false-positive floor. This test asserts the
    control exists and is wired to the same classifier, not that the floor has any particular
    value; the value is measured and reported, never assumed.
    """
    from agent.app.testing.claim_audit import recomputable_against_foreign_pages

    answer = _cell(final_text="From 419.7 m and 330.0 m the difference is 89.7 m.")
    foreign = _cell(pages=[{"page_id": "p1", "url": "u", "text": "Unrelated: 5.0 and 7.0 apples."}])

    assert recomputable_against_foreign_pages(answer, foreign)["89.7"] is False


def test_a_claim_a_foreign_page_can_reproduce_is_reported_as_a_coincidence():
    from agent.app.testing.claim_audit import recomputable_against_foreign_pages

    answer = _cell(final_text="From 100.7 m and 11.0 m the difference is 89.7 m.")
    foreign = _cell(pages=[{"page_id": "p1", "url": "u", "text": "Values 100.7 m and 11.0 m."}])

    coincidences = recomputable_against_foreign_pages(answer, foreign)

    assert coincidences["89.7"] is True


def test_operands_the_answer_never_mentions_cannot_credit_a_derivation():
    """The constraint that made `recomputable` a signal instead of arithmetic luck.

    Measured on the tuning split: with operands drawn from every number on every page, the
    cross-cell coincidence floor was 0.2526 -- five times the frozen ceiling, and HIGHER than the
    genuine recomputable rate. A pair search over ~24 numbers and 6 operations fires ~3,300
    candidates at a 2% tolerance window, so it hits almost anything. Requiring the operands to be
    quantities the answer itself states drops the floor to 0.0138 while genuine detection holds
    (0.408 -> 0.351), because a real derivation is reported with its inputs.
    """
    from agent.app.testing.claim_audit import UNSUPPORTED, classify_claims

    # 89.7 == 419.7 - 330.0, but the answer cites neither operand, so the match is a coincidence
    # of the page's contents rather than evidence of a derivation.
    cell = _cell(
        final_text="The difference is 89.7 metres.",
        pages=[{"page_id": "p1", "url": "u",
                "text": "Tower A stands 419.7 metres. Tower B stands 330.0 metres."}],
    )

    assert classify_claims(audit_input(cell))["89.7"] == UNSUPPORTED
