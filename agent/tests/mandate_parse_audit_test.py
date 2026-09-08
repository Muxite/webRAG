"""Regression oracle for the CURRENT (broken) mandate-parse table on tasks 210-221.

WHY THIS TEST PINS BROKEN BEHAVIOUR
-----------------------------------
Phase 0a of ``docs/superpowers/plans/2026-09-08-ledger-dag-replan.md`` audits what the two
existing mandate parsers actually see on the real derivation suite. The finding they encode is
an *anti-correlation*: exactly the tasks whose operation is recognised have zero entity slots,
and exactly the tasks with five entity slots have no recognised operation. So ``host_derive``
has 0/192 availability on the real suite -- there is no task where both halves are present.

  * 210-217 -- lettered ``A.``/``B.`` items whose bodies begin with "Open"/"Read", so
    ``extract_named_candidates`` bails (digit-only ``_NUMBERED_LINE`` plus the
    ``_INSTRUCTION_VERBS`` veto in ``agent/app/idea_policies/candidate_coverage.py``).
    ``mandate_demanded_operation`` DOES fire -> operation in {difference, sum, quotient},
    ``n_entities == 0``.
  * 218-221 -- five bare-name candidates parse fine (``n_entities == 5``), but the argmax
    phrasing trips the deliberate ``argmax_phrasing`` veto in
    ``agent/app/answer_numbers.py:mandate_demanded_operation`` -> ``operation is None``.

THIS TEST IS EXPECTED TO GO RED.
The day the Phase-2a slot parser (``agent/app/mandate_slots.py``) lands and is wired into these
call sites, the table flips (210-217 grow 2 slots each; 218-221 gain an operation) and these
assertions fail. **That failure is the acceptance signal for Phase 2a, not a defect.** Do not
mark this test skip/xfail to keep the suite green -- update the pinned table (and this
docstring) as part of the Phase-2a change, so the flip is visible in that diff.
"""
import pytest

from scripts.mandate_parse_audit import DERIVATION_SUITE_IDS, audit_modules

TWO_OPERAND_IDS = [str(i) for i in range(210, 218)]
ARGMAX_IDS = [str(i) for i in range(218, 222)]


@pytest.fixture(scope="module")
def table():
    return {row["test_id"]: row for row in audit_modules(DERIVATION_SUITE_IDS)}


def test_suite_ids_are_210_through_221():
    assert DERIVATION_SUITE_IDS == [str(i) for i in range(210, 222)]


def test_every_module_resolves(table):
    assert sorted(table) == [str(i) for i in range(210, 222)]
    for row in table.values():
        assert row["module"].startswith("agent.app.idea_tests.test_")
        assert row["error"] is None


@pytest.mark.parametrize("test_id", TWO_OPERAND_IDS)
def test_two_operand_tasks_have_an_operation_and_no_entities(table, test_id):
    """210-217: the operation cue fires, the candidate extractor returns nothing."""
    row = table[test_id]
    assert row["operation"] in {"difference", "sum", "quotient"}, row
    assert row["reason"] == f"{row['operation']}_cue", row
    assert row["n_entities"] == 0, row
    assert row["entities"] == [], row


@pytest.mark.parametrize("test_id", ARGMAX_IDS)
def test_argmax_tasks_have_five_entities_and_no_operation(table, test_id):
    """218-221: five candidates parse, the argmax veto suppresses the operation."""
    row = table[test_id]
    assert row["operation"] is None, row
    assert row["reason"] == "argmax_phrasing", row
    assert row["n_entities"] == 5, row
    assert len(row["entities"]) == 3, row


def test_the_anti_correlation_holds_across_the_whole_suite(table):
    """No task in 210-221 has BOTH an operation and >=2 entity slots -> host_derive is 0/12."""
    both = [tid for tid, row in table.items()
            if row["operation"] is not None and row["n_entities"] >= 2]
    assert both == [], f"host_derive availability is no longer 0/12: {both}"
