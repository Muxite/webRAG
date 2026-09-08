"""``LedgerToolkit.host_derive`` — the mandate-shaped derivation the HOST computes for itself.

Phase 2c of ``docs/superpowers/plans/2026-09-08-ledger-dag-replan.md``. Everything the model can
see is untouched: no prompt line, no tool, no observation string. The host reads the mandate, picks
the operands off pages the run already fetched, and recomputes the demanded arithmetic in Python.

The mandates under test are the REAL ones (``agent/app/idea_tests/test_2{10..21}_*.py``), loaded
the way ``scripts/mandate_parse_audit.py`` loads them, because the whole point of the replan's
Phase 0a finding was that the previous plan's hand-authored fixture mandates hid a 0/192
availability. Only the PAGES are synthetic, and they are written in the flattened-infobox line
shape ``quantity_index`` actually reads (label line, bare value line, unit line).
"""
from __future__ import annotations

import glob
import importlib
import os

import pytest

from agent.app.ledger_tools import HOST_DERIVE_TAG, LedgerToolkit
from agent.app.operand_attribution import document_order_ranker

_TESTS_PKG = "agent.app.idea_tests"


def statement(test_id: str) -> str:
    """The real mandate of task ``test_id``, imported as a package module."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    matches = sorted(glob.glob(os.path.join(root, "agent", "app", "idea_tests",
                                            f"test_{test_id}_*.py")))
    assert matches, f"no task module for {test_id}"
    module = importlib.import_module(f"{_TESTS_PKG}." + os.path.basename(matches[0])[:-3])
    return module.get_task_statement()


# --------------------------------------------------------------------------------------------
# Synthetic pages, in the flattened-Wikipedia-infobox shape `quantity_index._scan_infobox` reads.
# A unitless infobox value ("94" under "Floor count") is indexed with unit "count", which is why
# a floor count is a legitimate operand rather than a "trivial bare int".
# --------------------------------------------------------------------------------------------

GRES2_URL = "https://en.wikipedia.org/wiki/Ekibastuz_GRES-2_Power_Station"
GRES2_PAGE = """GRES-2 Power Station chimney
The chimney of the Ekibastuz GRES-2 Power Station is the tallest chimney in the world.
Height
419.7
m
Completed
1987
"""

INCO_URL = "https://en.wikipedia.org/wiki/Inco_Superstack"
INCO_PAGE = """Inco Superstack
The Inco Superstack is a smokestack in Sudbury, Ontario, Canada.
Height
380.0
m
Completed
1972
"""
#: The same page with its one height written in FEET -- the unit-mismatch fixture.
INCO_PAGE_FEET = INCO_PAGE.replace("380.0\nm", "1,247\nft")

BAIKAL_URL = "https://en.wikipedia.org/wiki/Lake_Baikal"
BAIKAL_PAGE = """Lake Baikal
Lake Baikal is a rift lake in Siberia, Russia.
Surface area
31,722
km
2
Max. depth
1,642
m
Average depth
744.4
m
"""

TANGANYIKA_URL = "https://en.wikipedia.org/wiki/Lake_Tanganyika"
TANGANYIKA_PAGE = """Lake Tanganyika
Lake Tanganyika is an African Great Lake.
Surface area
32,600
km
2
Max. depth
1,470
m
Average depth
570
m
"""

PUSKAS_URL = "https://en.wikipedia.org/wiki/Pusk%C3%A1s_Ar%C3%A9na"
#: The construction cost is written out in full rather than as "575 million": a scale WORD is
#: carried in `QuantityRef.unit` and `evidence_graph.numeric_value` deliberately does not read a
#: scale out of its `unit=` argument, so a "575 million" fixture would assert the magnitude loss
#: of a neighbouring module rather than anything about `host_derive`.
PUSKAS_PAGE = """Puskás Aréna
Puskás Aréna is a football stadium in Budapest, Hungary.
Construction cost
575,000,000
Capacity
67,215
Opened
2019
"""

SHINKANSEN_URL = "https://en.wikipedia.org/wiki/T%C5%8Dkaid%C5%8D_Shinkansen"
SHINKANSEN_PAGE = """Tōkaidō Shinkansen
The Tōkaidō Shinkansen is a Japanese high-speed rail line between Tokyo and Shin-Osaka.
Line length
515.4
km
Journey time
2.35
h
Opened
1964
"""

#: (entity, url slug, length km, basin area km2) for task 218. Mekong wins on length/basin
#: although it is neither the longest river nor the one with the largest basin.
RIVERS = [
    ("Mekong", "Mekong", "4,909", "795,000"),
    ("Yangtze", "Yangtze", "6,300", "1,800,000"),
    ("Nile", "Nile", "6,650", "3,254,555"),
    ("Mississippi", "Mississippi_River", "3,766", "2,980,000"),
    ("Amazon", "Amazon_River", "6,400", "7,000,000"),
]

#: (entity, url slug, architectural height, floor count) for task 221. One World Trade Center
#: wins on height/floors although Burj Khalifa is far taller.
BUILDINGS = [
    ("One World Trade Center", "One_World_Trade_Center", "541.3\nm", "94"),
    ("Burj Khalifa", "Burj_Khalifa", "828\nm", "163"),
    ("Taipei 101", "Taipei_101", "508\nm", "101"),
    ("Shanghai Tower", "Shanghai_Tower", "632\nm", "128"),
    ("Willis Tower", "Willis_Tower", "442.1\nm", "108"),
]


def _river_page(name: str, length: str, basin: str) -> str:
    return (f"{name}\n{name} is a major river.\nLength\n{length}\nkm\n"
            f"Basin size\n{basin}\nkm\n2\n")


def _building_page(name: str, height: str, floors: str) -> str:
    return (f"{name}\n{name} is a supertall skyscraper.\nArchitectural height\n{height}\n"
            f"Floor count\n{floors}\nCompleted\n2010\n")


def _wiki(slug: str) -> str:
    return f"https://en.wikipedia.org/wiki/{slug}"


@pytest.fixture
def kit():
    return LedgerToolkit()


def _rivers_kit(kit):
    for name, slug, length, basin in RIVERS:
        kit.register_page(_wiki(slug), _river_page(name, length, basin))
    return kit


def _buildings_kit(kit, heights=None):
    for index, (name, slug, height, floors) in enumerate(BUILDINGS):
        if heights is not None:
            height = heights[index]
        kit.register_page(_wiki(slug), _building_page(name, height, floors))
    return kit


def _selected(result):
    """The rows of ``result["slots"]`` that actually chose an operand."""
    return [row for row in result["slots"] if row["reason"] == "selected"]


def _spans(result):
    return [(row["page_id"], row["entry"]["start"], row["entry"]["end"])
            for row in _selected(result)]


def _node(kit, node_id):
    return next(n for n in kit.artifact()["nodes"] if n["id"] == node_id)


# --------------------------------------------------------------------------------------------
# Two-operand mandates (210-217)
# --------------------------------------------------------------------------------------------


def test_absolute_difference_is_computed_from_the_two_slots_the_mandate_names(kit):
    """Task 210: two entities, two pages, one field each -> abs(419.7 - 380.0) m."""
    kit.register_page(GRES2_URL, GRES2_PAGE)
    kit.register_page(INCO_URL, INCO_PAGE)

    result = kit.host_derive(statement("210"))

    assert result["reason"] == "computed"
    assert result["operation"] == "difference"
    assert result["absolute"] is True
    assert result["value"] == pytest.approx(39.7)
    assert result["unit"] == "m"
    assert result["mode"] is None
    assert result["ranker"] == "hand_rule"
    assert [row["entity"] for row in _selected(result)] == ["GRES-2 Power Station chimney",
                                                            "Inco Superstack"]
    assert [row["entry"]["value"] for row in _selected(result)] == ["419.7", "380.0"]


def test_every_node_the_mechanism_mints_carries_its_own_provenance_tag(kit):
    """The tag is what lets a consumer separate host-minted rows from model-driven `derive` rows."""
    kit.register_page(GRES2_URL, GRES2_PAGE)
    kit.register_page(INCO_URL, INCO_PAGE)

    result = kit.host_derive(statement("210"))

    derived = _node(kit, result["node_id"])
    assert derived["minted_by"] == HOST_DERIVE_TAG
    assert derived["derivation_valid"] is True
    operands = [_node(kit, node_id) for node_id in derived["input_ids"]]
    assert [node["minted_by"] for node in operands] == [HOST_DERIVE_TAG, HOST_DERIVE_TAG]
    assert all(node["quote"] for node in operands), "each operand keeps a verbatim page quote"


def test_a_shared_field_phrase_across_two_entities_still_picks_two_distinct_operands(kit):
    """Task 211 asks the SAME field ("its MAXIMUM DEPTH") of two different entities."""
    kit.register_page(BAIKAL_URL, BAIKAL_PAGE)
    kit.register_page(TANGANYIKA_URL, TANGANYIKA_PAGE)

    result = kit.host_derive(statement("211"))

    assert result["reason"] == "computed"
    assert result["operation"] == "sum"
    assert result["value"] == pytest.approx(3112.0)
    assert result["unit"] == "m"
    assert len(set(_spans(result))) == 2
    assert len({row["page_id"] for row in _selected(result)}) == 2


def test_two_fields_of_one_entity_are_two_distinct_entries_on_the_same_page(kit):
    """Task 216: route length / journey time of ONE line -- why distinctness is keyed on the SPAN
    and not on the entity, and why km / h is a legal quotient (a rate, not a conversion)."""
    kit.register_page(SHINKANSEN_URL, SHINKANSEN_PAGE)

    result = kit.host_derive(statement("216"))

    assert result["reason"] == "computed"
    assert result["operation"] == "quotient"
    assert result["value"] == pytest.approx(515.4 / 2.35, abs=1e-6)
    assert result["unit"] == "km/h"
    assert [row["entry"]["label"] for row in _selected(result)] == ["Line length", "Journey time"]
    assert len({row["page_id"] for row in _selected(result)}) == 1
    assert len(set(_spans(result))) == 2


def test_an_operand_the_quantity_index_cannot_see_refuses_instead_of_substituting(kit):
    """Task 215's construction cost is a bare euro figure: `quantity_index` indexes a unit-less
    infobox value only under a whitelisted COUNT label ("Capacity", "Floors", ...), so the cost row
    is not in the index at all. The only candidate the page offers for the cost slot is therefore
    the CAPACITY figure -- and `_HOST_DERIVE_MIN_SCORE` is exactly what stops the mechanism from
    dividing the capacity by itself and reporting a confident number. This is a real availability
    limit of the index, pinned rather than fixtured around."""
    kit.register_page(PUSKAS_URL, PUSKAS_PAGE)

    result = kit.host_derive(statement("215"))

    assert result["reason"] == "operand_not_found"
    assert [row["reason"] for row in result["slots"]] == ["below_min_score", "selected"]
    assert result["slots"][0]["score"] < 0.93
    assert result["value"] is None and result["node_id"] is None


def test_incompatible_units_refuse_and_are_recorded_rather_than_converted(kit):
    """A metres-minus-feet difference is never silently converted; the refusal is on the artifact."""
    kit.register_page(GRES2_URL, GRES2_PAGE)
    kit.register_page(INCO_URL, INCO_PAGE_FEET)

    result = kit.host_derive(statement("210"))

    assert result["reason"] == "unit_mismatch"
    assert result["value"] is None and result["node_id"] is None
    assert kit.artifact()["derivation_refusals"], "the refusal must be recorded, not just returned"
    assert kit.artifact()["derivation_refusals"][-1]["code"] == "UNIT_MISMATCH"


def test_a_page_without_the_asked_field_reports_operand_not_found(kit):
    """Refusing beats guessing: nothing on this page carries depth label evidence."""
    kit.register_page(BAIKAL_URL,
                      "Lake Baikal\nLake Baikal is a rift lake.\nSurface area\n31,722\nkm\n2\n")
    kit.register_page(TANGANYIKA_URL, TANGANYIKA_PAGE)

    result = kit.host_derive(statement("211"))

    assert result["reason"] == "operand_not_found"
    assert result["value"] is None
    assert [row["reason"] for row in result["slots"]] == ["below_min_score", "selected"]


def test_an_operation_without_a_roster_reports_fewer_than_two_slots(kit):
    kit.register_page(GRES2_URL, GRES2_PAGE)

    result = kit.host_derive("Then COMPUTE the SUM of the two values you read, in m.")

    assert result["reason"] == "fewer_than_two_slots"
    assert result["operation"] == "sum"
    assert result["slots"] == []


def test_a_chain_mandate_has_no_shape_to_derive(kit):
    kit.register_page(GRES2_URL, GRES2_PAGE)

    result = kit.host_derive(
        "Find the architect of the Inco Superstack, then report the city they were born in.")

    assert result["reason"] == "no_unambiguous_shape"
    assert result["operation"] is None and result["value"] is None


def test_a_toolkit_that_registered_no_page_can_derive_nothing(kit):
    result = kit.host_derive(statement("210"))

    assert result["reason"] == "no_pages"
    assert result["n_pages"] == 0 and result["n_entries"] == 0
    assert result["value"] is None


def test_the_mechanism_never_raises_on_junk(kit):
    for mandate in (None, "", 17, "= = =", statement("210")):
        assert kit.host_derive(mandate)["reason"] in (
            "no_unambiguous_shape", "fewer_than_two_slots", "no_pages", "error")


# --------------------------------------------------------------------------------------------
# Argmax mandates (218-221): one ratio per entity, then an extremum over the ratios
# --------------------------------------------------------------------------------------------


def test_argmax_over_a_per_entity_ratio_names_the_winning_entity(kit):
    """Task 218: length / basin area for five rivers, then the max."""
    result = _rivers_kit(kit).host_derive(statement("218"))

    assert result["reason"] == "computed"
    assert result["mode"] == "max"
    assert result["operation"] == "ratio"
    assert result["winner_entity"] == "Mekong"
    # `add_arith` normalises a derived value to 6 decimal places, so the reported figure is the
    # page arithmetic rounded, never a float tail the pages do not support.
    assert result["value"] == pytest.approx(4909 / 795000, abs=1e-6)
    # Two operand rows per entity (numerator and denominator), all ten selected.
    assert len(_selected(result)) == 10
    assert len(set(_spans(result))) == 10


def test_the_argmax_winner_is_the_extremum_nodes_own_winning_input(kit):
    result = _buildings_kit(kit).host_derive(statement("221"))

    assert result["reason"] == "computed"
    assert result["winner_entity"] == "One World Trade Center"
    assert result["value"] == pytest.approx(541.3 / 94, rel=1e-6)

    extremum = _node(kit, result["node_id"])
    assert extremum["operation"] == "max"
    assert extremum["minted_by"] == HOST_DERIVE_TAG
    assert len(extremum["input_ids"]) == 5
    ratios = [_node(kit, node_id) for node_id in extremum["input_ids"]]
    assert all(node["operation"] == "ratio" for node in ratios)
    assert all(node["minted_by"] == HOST_DERIVE_TAG for node in ratios)
    assert extremum["value"] == max(ratios, key=lambda node: float(node["value"]))["value"]


def test_entities_measured_in_different_units_refuse_rather_than_convert(kit):
    """One building's height in feet makes the five ratios incomparable -- and no conversion is
    the module's standing non-goal, so this is a refusal, not a rescale."""
    heights = [height for _, _, height, _ in BUILDINGS]
    heights[4] = "1,451\nft"

    result = _buildings_kit(kit, heights=heights).host_derive(statement("221"))

    assert result["reason"] == "unit_inconsistent_across_entities"
    assert result["value"] is None and result["winner_entity"] is None
    assert kit.artifact()["derivation_refusals"][-1]["code"] == "UNIT_MISMATCH"


def test_an_argmax_mandate_with_no_stated_formula_is_not_guessed_at(kit):
    result = _rivers_kit(kit).host_derive(
        "Which of these five rivers has the highest average discharge?\n"
        "  1. Mekong\n  2. Yangtze\n  3. Nile\n  4. Mississippi\n  5. Amazon\n")

    assert result["reason"] == "argmax_formula_unparsed"
    assert result["value"] is None and result["node_id"] is None


def test_too_few_entities_resolve_to_report_an_extremum(kit):
    """Four of the five river pages were never fetched, so there is nothing to compare."""
    name, slug, length, basin = RIVERS[0]
    kit.register_page(_wiki(slug), _river_page(name, length, basin))

    result = kit.host_derive(statement("218"))

    assert result["reason"] == "operand_not_found"
    assert len(result["slots"]) == 10
    assert len(_selected(result)) == 2


# --------------------------------------------------------------------------------------------
# The ablation arm and the reported contract
# --------------------------------------------------------------------------------------------


def test_the_document_order_control_ranker_plumbs_through(kit):
    """The negative control ranks nothing: it takes each page's FIRST entry. Its constant 0.5 is
    not on the hand rule's scale, so the replay arm passes `min_score=0.0` with it."""
    kit.register_page(BAIKAL_URL, BAIKAL_PAGE)
    kit.register_page(TANGANYIKA_URL, TANGANYIKA_PAGE)

    result = kit.host_derive(statement("211"), ranker=document_order_ranker(), min_score=0.0)

    assert result["ranker"] == "document_order"
    assert result["min_score"] == 0.0
    assert [row["entry"]["label"] for row in _selected(result)] == ["Surface area", "Surface area"]


def test_the_result_dict_always_carries_the_whole_contract(kit):
    """Other lanes code against these keys, present on every reason -- computed or refused."""
    kit.register_page(GRES2_URL, GRES2_PAGE)
    expected = {"reason", "operation", "absolute", "mode", "value", "value_text", "unit",
                "node_id", "winner_entity", "slots", "ranker", "n_pages", "n_entries",
                "min_score"}

    for mandate in (statement("210"), statement("218"), "Describe the tower."):
        result = kit.host_derive(mandate)
        assert set(result) == expected
        for row in result["slots"]:
            assert set(row) == {"index", "entity", "field_phrase", "page_id", "url", "entry",
                                "score", "reason"}
            assert row["entry"] is None or set(row["entry"]) == {"label", "value", "unit",
                                                                 "start", "end", "source"}
