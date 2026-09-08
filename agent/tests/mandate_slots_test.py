"""Unit tests for ``agent.app.mandate_slots`` -- the Phase-2a lexical slot parser.

Two halves: the twelve REAL derivation mandates (loaded as real package modules, the way
``scripts/mandate_parse_audit.py`` does, so module constants keep identity), and synthetic edge
cases for the shapes the real suite does not contain. Nothing here touches the network or a
model; ``parse_slots`` is pure regex over the task statement.
"""
import glob
import importlib
import os

import pytest

from agent.app.mandate_slots import Slot, parse_slots, slot_field_phrases

TWO_OPERAND_IDS = [str(i) for i in range(210, 218)]
ARGMAX_IDS = [str(i) for i in range(218, 222)]
#: 215/216 ask for two DIFFERENT fields of the SAME entity (cost/capacity, length/journey time).
SINGLE_ENTITY_IDS = {"215", "216"}
#: Report-item rosters whose items begin with an article, not an imperative verb -- the shape
#: ``_INSTRUCTION_VERBS`` alone does not catch, and the one the name-shape test must reject.
INSTRUCTION_ROSTER_IDS = ["044", "093"]


def _mandate(test_id: str) -> str:
    matches = sorted(glob.glob(os.path.join("agent", "app", "idea_tests", f"test_{test_id}_*.py")))
    assert matches, f"no module matching test_{test_id}_*.py"
    mod = importlib.import_module("agent.app.idea_tests." + os.path.basename(matches[0])[:-3])
    return mod.get_task_statement()


@pytest.fixture(scope="module")
def slots():
    return {tid: parse_slots(_mandate(tid))
            for tid in TWO_OPERAND_IDS + ARGMAX_IDS + INSTRUCTION_ROSTER_IDS}


# ---------------------------------------------------------------------------
# The real suite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("test_id", TWO_OPERAND_IDS)
def test_two_operand_mandates_parse_into_two_directed_slots(slots, test_id):
    """210-217: the lettered ``A.``/``B.`` roster yields exactly two slots, each distinct."""
    got = slots[test_id]
    assert len(got) == 2, got
    assert [s.index for s in got] == [0, 1]
    for slot in got:
        assert isinstance(slot, Slot)
        assert slot.entity and not slot.entity.lower().startswith("open")
        assert slot.field_phrase
        assert slot.url is None  # these mandates name no URLs; the agent must search
    # The pair is always distinguishable -- by entity, by field, or by both.
    assert (got[0].entity, got[0].field_phrase) != (got[1].entity, got[1].field_phrase), got


@pytest.mark.parametrize("test_id", sorted(SINGLE_ENTITY_IDS))
def test_single_entity_two_field_mandates_share_an_entity_and_differ_in_field(slots, test_id):
    """215/216 are the single-entity quotients: same page, two different figures on it."""
    got = slots[test_id]
    assert got[0].entity == got[1].entity, got
    assert got[0].field_phrase != got[1].field_phrase, got
    assert len(slot_field_phrases(got)) == 2


@pytest.mark.parametrize("test_id", sorted(set(TWO_OPERAND_IDS) - SINGLE_ENTITY_IDS))
def test_two_entity_mandates_name_two_different_entities(slots, test_id):
    """210-214/217: two entities. Their FIELD phrases may be identical (211/213/214/217 ask the
    same field of both entities), which is why slot distinctness is the (entity, field) pair."""
    got = slots[test_id]
    assert got[0].entity != got[1].entity, got


def test_the_real_field_phrases_keep_their_unit_hints(slots):
    """A field phrase is the mandate's own wording, unit hint included (211's ", in meters")."""
    assert slots["211"][0].field_phrase == "its MAXIMUM DEPTH, in meters"
    assert slots["211"][1].field_phrase == "its MAXIMUM DEPTH, in meters"
    assert slots["217"][0].field_phrase == "its surface area, in km2"


def test_entities_drop_their_qualifier_parenthetical(slots):
    """"Lake Tahoe (California/Nevada, USA)" -> "Lake Tahoe": the parenthetical is a hint to the
    reader, and would only hurt a URL-slug / page-text match downstream."""
    assert [s.entity for s in slots["217"]] == ["Lake Titicaca", "Lake Tahoe"]
    assert slots["210"][0].entity == "GRES-2 Power Station chimney"
    assert slots["215"][0].entity == "Puskás Aréna"


@pytest.mark.parametrize("test_id", ARGMAX_IDS)
def test_argmax_mandates_parse_into_five_bare_name_slots_with_one_shared_field(slots, test_id):
    """218-221: five numbered bare names; the field is stated once, in the surrounding prose."""
    got = slots[test_id]
    assert len(got) == 5, got
    assert [s.index for s in got] == [0, 1, 2, 3, 4]
    assert len({s.entity for s in got}) == 5, got
    phrases = slot_field_phrases(got)
    assert len(phrases) == 1, phrases
    assert "infobox" in phrases[0]


def test_the_shared_field_comes_from_the_prose_not_from_a_roster_item(slots):
    """The 218 field names BOTH per-entity figures and no river."""
    phrase = slots["218"][0].field_phrase
    assert "LENGTH" in phrase and "DRAINAGE BASIN" in phrase
    assert "Mekong" not in phrase and "Amazon" not in phrase
    assert [s.entity for s in slots["218"]] == [
        "Mekong", "Yangtze", "Nile", "Mississippi", "Amazon"]


@pytest.mark.parametrize("test_id", INSTRUCTION_ROSTER_IDS)
def test_report_item_rosters_produce_no_slots(slots, test_id):
    """044/093 enumerate what to REPORT ("1. The exact name of that vulnerable C function."),
    not entities. Every item begins with an article, so the imperative-verb veto alone misses
    them; the name-shape test is what keeps them out."""
    assert slots[test_id] == []


# ---------------------------------------------------------------------------
# Synthetic edge cases
# ---------------------------------------------------------------------------

_LETTERED = (
    "You need TWO values:\n"
    "  A. Open the Wikipedia page for Mount Everest and read its elevation, in meters.\n"
    "  B. Open the Wikipedia page for K2 and read its elevation, in meters.\n"
    "\nThen COMPUTE the difference.\n"
)
_NUMBERED_DIRECTED = (
    "You need TWO values:\n"
    "  1. Open the Wikipedia page for Mount Everest and read its elevation, in meters.\n"
    "  2) Open the Wikipedia page for K2 and read its elevation, in meters.\n"
)
_PARENTHESISED = (
    "You need TWO values:\n"
    "  (a) Open the Wikipedia page for Mount Everest and read its elevation, in meters.\n"
    "  (b) Open the Wikipedia page for K2 and read its elevation, in meters.\n"
)


@pytest.mark.parametrize("mandate", [_LETTERED, _NUMBERED_DIRECTED, _PARENTHESISED],
                         ids=["lettered", "numbered", "parenthesised"])
def test_all_three_marker_styles_parse_the_same_two_slots(mandate):
    got = parse_slots(mandate)
    assert [(s.entity, s.field_phrase) for s in got] == [
        ("Mount Everest", "its elevation, in meters"),
        ("K2", "its elevation, in meters"),
    ]


def test_a_url_written_into_an_item_is_captured():
    mandate = (
        "  A. Open the Wikipedia page for Mount Everest "
        "(https://en.wikipedia.org/wiki/Mount_Everest) and read its elevation, in meters.\n"
        "  B. Open https://en.wikipedia.org/wiki/K2 and read its elevation, in meters.\n"
    )
    got = parse_slots(mandate)
    assert [s.url for s in got] == [
        "https://en.wikipedia.org/wiki/Mount_Everest", "https://en.wikipedia.org/wiki/K2"]
    assert [s.entity for s in got] == ["Mount Everest", "K2"]


def test_bare_name_items_without_an_open_body_take_the_shared_prose_field():
    mandate = (
        "For EACH of the following peaks, open its page and read its prominence, in metres:\n"
        "  1. Mount Everest\n  2. K2\n  3. Denali\n"
        "\nThen determine which has the highest prominence.\n"
    )
    got = parse_slots(mandate)
    assert [s.entity for s in got] == ["Mount Everest", "K2", "Denali"]
    assert slot_field_phrases(got) == ["its prominence, in metres"]


def test_bare_name_items_fall_back_to_derive_field_label_without_a_read_cue():
    """No "and read <phrase>" anywhere -> the field is ``derive_field_label``'s prose label,
    which is never empty, so the slots still carry a usable (if blunt) field."""
    mandate = "Which of these is tallest?\n  1. Mount Everest\n  2. K2\n"
    got = parse_slots(mandate)
    assert len(got) == 2
    assert slot_field_phrases(got) == ["Which of these is tallest?"]


def test_a_mixed_roster_fails_open():
    """One directed item and one bare name is not one shape, so nothing is parsed."""
    mandate = (
        "  A. Open the Wikipedia page for Mount Everest and read its elevation, in meters.\n"
        "  B. K2\n"
    )
    assert parse_slots(mandate) == []


def test_an_instruction_verb_roster_fails_open():
    mandate = "  1. Identify the poet.\n  2. Open that poet's page.\n  3. Report the year.\n"
    assert parse_slots(mandate) == []


def test_a_single_item_is_not_a_roster():
    assert parse_slots("  A. Open the Wikipedia page for K2 and read its elevation.\n") == []


def test_a_non_consecutive_run_fails_open():
    mandate = ("  A. Open the Wikipedia page for Mount Everest and read its elevation.\n"
               "  D. Open the Wikipedia page for K2 and read its elevation.\n")
    assert parse_slots(mandate) == []


@pytest.mark.parametrize("mandate", ["", "   \n\n", "Just some prose with no roster at all.",
                                     None, 42, ["1. Mekong", "2. Nile"]],
                         ids=["empty", "blank", "prose", "none", "int", "list"])
def test_unrecognised_input_returns_an_empty_list_and_never_raises(mandate):
    assert parse_slots(mandate) == []


def test_slot_field_phrases_is_distinct_in_order():
    slots = [Slot("A", "height", None, 0), Slot("B", "width", None, 1),
             Slot("C", "height", None, 2), Slot("D", "", None, 3)]
    assert slot_field_phrases(slots) == ["height", "width"]
    assert slot_field_phrases([]) == []
