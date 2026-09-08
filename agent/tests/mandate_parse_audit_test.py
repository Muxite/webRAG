"""Regression oracle for the mandate-parse table on tasks 210-221.

WHAT THIS TEST PINS
-------------------
Phase 0a of ``docs/superpowers/plans/2026-09-08-ledger-dag-replan.md`` audited what the two
pre-existing mandate parsers see on the real derivation suite. The finding they encode is an
*anti-correlation*: exactly the tasks whose operation is recognised have zero entity slots, and
exactly the tasks with five entity slots have no recognised operation, so a host-side
``host_derive`` built on that pair had 0/12 joint availability.

  * 210-217 -- lettered ``A.``/``B.`` items whose bodies begin with "Open", so
    ``extract_named_candidates`` bails (digit-only ``_NUMBERED_LINE`` plus the
    ``_INSTRUCTION_VERBS`` veto in ``agent/app/idea_policies/candidate_coverage.py``).
    ``mandate_demanded_operation`` DOES fire -> operation in {difference, sum, quotient},
    ``n_entities == 0``.
  * 218-221 -- five bare-name candidates parse fine (``n_entities == 5``), but the argmax
    phrasing trips the deliberate ``argmax_phrasing`` veto in
    ``agent/app/answer_numbers.py:mandate_demanded_operation`` -> ``operation is None``.

**This test went red at Phase 2a, and is now green again with the slot columns.** The legacy
half above is unchanged and still pinned exactly as it was -- ``extract_named_candidates`` was
deliberately NOT widened (8 consumers), so its 0 entities on 210-217 remain a fact of the
codebase. What flipped is the NEW half: ``agent/app/mandate_slots.py:parse_slots`` reads both
roster shapes, so ``n_slots`` is 2 on 210-217 and 5 on 218-221, and joint availability measured
through slots is 12/12 while the legacy joint availability stays 0/12.

Field phrases are pinned as the mandates actually word them, which is *not* "two distinct
phrases per two-operand task": 211/213/214/217 ask the SAME field ("its area, in km2") of two
different entities, and 215/216 ask two different fields of the SAME entity. What is always
true is that the two slots differ as an ``(entity, field_phrase)`` PAIR -- which is the
distinctness ``host_derive`` needs, and why Phase 2c keys operand distinctness on
``(page_id, start, end)`` rather than on the entity.
"""
import pytest

from scripts.mandate_parse_audit import DERIVATION_SUITE_IDS, audit_modules, format_table

TWO_OPERAND_IDS = [str(i) for i in range(210, 218)]
ARGMAX_IDS = [str(i) for i in range(218, 222)]
#: The two-operand tasks that ask two different fields of ONE entity (cost/capacity for the
#: Puskás Aréna, route length / journey time for the Tōkaidō Shinkansen).
SINGLE_ENTITY_IDS = ["215", "216"]


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


# ---------------------------------------------------------------------------
# The LEGACY parsers -- unchanged by Phase 2a, pinned exactly as before
# ---------------------------------------------------------------------------


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


def test_the_legacy_anti_correlation_still_holds(table):
    """No task in 210-221 has BOTH an operation and >=2 LEGACY entity slots -> 0/12, still."""
    both = [tid for tid, row in table.items()
            if row["operation"] is not None and row["n_entities"] >= 2]
    assert both == [], f"extract_named_candidates was widened after all: {both}"


# ---------------------------------------------------------------------------
# The Phase-2a slot parser -- what flipped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("test_id", TWO_OPERAND_IDS)
def test_two_operand_tasks_yield_two_slots(table, test_id):
    """210-217: two slots, and they are distinguishable as (entity, field) pairs."""
    row = table[test_id]
    assert row["n_slots"] == 2, row
    assert len(row["slot_entities"]) == 2, row
    assert all(row["slot_entities"]) and all(row["slot_fields"]), row
    # Distinguishable by entity, by field, or by both -- never by neither.
    assert len(set(row["slot_entities"])) == 2 or len(row["slot_fields"]) == 2, row


@pytest.mark.parametrize("test_id", SINGLE_ENTITY_IDS)
def test_single_entity_tasks_have_two_distinct_field_phrases(table, test_id):
    """215/216 read two DIFFERENT figures off ONE page -> one entity, two field phrases."""
    row = table[test_id]
    assert len(set(row["slot_entities"])) == 1, row
    assert len(row["slot_fields"]) == 2, row


@pytest.mark.parametrize("test_id", sorted(set(TWO_OPERAND_IDS) - set(SINGLE_ENTITY_IDS)))
def test_two_entity_tasks_have_two_distinct_entities(table, test_id):
    """210-214/217 read the same-or-different field off TWO pages. 211/213/214/217 word the
    field IDENTICALLY for both entities, so ``slot_fields`` is a single phrase there -- the
    entity is what separates the slots."""
    row = table[test_id]
    assert len(set(row["slot_entities"])) == 2, row
    assert 1 <= len(row["slot_fields"]) <= 2, row


@pytest.mark.parametrize("test_id", ARGMAX_IDS)
def test_argmax_tasks_yield_five_slots_sharing_one_field(table, test_id):
    """218-221: five bare-name slots, all carrying the one field phrase the prose states."""
    row = table[test_id]
    assert row["n_slots"] == 5, row
    assert len(row["slot_fields"]) == 1, row
    assert "infobox" in row["slot_fields"][0], row


def test_joint_availability_via_slots_is_twelve_of_twelve(table):
    """The Phase-2a acceptance number: every task in 210-221 now has a derivable shape AND
    >= 2 parsed operand slots (the 218-221 shape is ``argmax_phrasing``, which is
    ``mandate_demanded_operation``'s deliberate answer, not a parse failure)."""
    usable = [tid for tid, row in table.items()
              if (row["operation"] is not None or row["reason"] == "argmax_phrasing")
              and row["n_slots"] >= 2]
    assert sorted(usable) == [str(i) for i in range(210, 222)]


def test_the_table_reports_both_joint_availability_lines(table):
    rendered = format_table([table[tid] for tid in DERIVATION_SUITE_IDS])
    assert "joint availability (operation AND >=2 entity slots): 0/12" in rendered
    assert ("joint_availability_slots (operation or argmax_phrasing AND >=2 parsed slots): "
            "12/12") in rendered
    assert "n_slots" in rendered.splitlines()[0]
    assert "slot_fields[:2]" in rendered.splitlines()[0]
