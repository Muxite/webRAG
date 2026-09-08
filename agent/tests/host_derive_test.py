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

#: The REAL Inco article's lead, abridged: its infobox names the GRES-2 chimney that surpassed it,
#: so the lead text of the INCO page carries every identifying token of the OTHER slot's entity
#: ("gres", "power", "station", "chimney"). Its URL slug still names only Inco.
INCO_PAGE_NAMING_GRES2 = """Inco Superstack
The Vale-Inco Superstack at the Inco Copper Cliff smelter
Record height
Tallest in the world from 1971 to 1987
Preceded by
Mitchell Power Plant
Surpassed by
Ekibastuz GRES-2 Power Station
Type
Chimney
Height
380.0
m
"""

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

TITICACA_URL = "https://en.wikipedia.org/wiki/Lake_Titicaca"
TITICACA_PAGE = """Lake Titicaca
Lake Titicaca is a lake in the Andes on the border of Peru and Bolivia.
Surface area
8,372
km
2
Max. depth
281
m
"""

TAHOE_URL = "https://en.wikipedia.org/wiki/Lake_Tahoe"
#: Tahoe's own article writes its surface area in SQUARE MILES -- the real 217 heterogeneity.
TAHOE_PAGE = """Lake Tahoe
Lake Tahoe is a freshwater lake in the Sierra Nevada of California and Nevada.
Surface area
191
sq mi
Max. depth
501
m
"""

#: The 216 page with its journey time written in MINUTES: a km-per-minute rate is still a rate.
SHINKANSEN_PAGE_MINUTES = SHINKANSEN_PAGE.replace("2.35\nh", "141\nmin")

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


def test_a_slot_no_registered_page_names_refuses_instead_of_reading_another_entity(kit):
    """Task 210 with ONLY the Inco page fetched. `host_derive` used to fall back to ALL registered
    pages when no page named a slot's entity, so the GRES-2 slot resolved against the INCO page,
    both slots selected `Height = 380.0 m`, and the host minted a confident |380 - 380| = 0.0 m.
    21 of the replay's 50 wrong hand-rule rows are that one substitution
    (`docs/handoffs/HOST_DERIVE_REPLAY_2026-09-08.md` section 4a). No page names the entity is a
    REFUSAL."""
    kit.register_page(INCO_URL, INCO_PAGE)

    result = kit.host_derive(statement("210"))

    assert result["reason"] == "operand_not_found"
    assert result["value"] is None and result["node_id"] is None
    assert [row["reason"] for row in result["slots"]] == ["no_candidate_page", "selected"]
    assert result["slots"][0]["page_id"] is None and result["slots"][0]["score"] is None
    assert not [node for node in kit.artifact()["nodes"]
                if node["minted_by"] == HOST_DERIVE_TAG], "nothing is minted for a refused cell"


def test_a_page_whose_URL_names_ANOTHER_slots_entity_is_not_read_for_this_one(kit):
    """The residual of the same substitution, and the reason removing the all-pages fallback alone
    changed nothing on the stored cells: the real Inco article's lead NAMES the GRES-2 Power
    Station chimney that surpassed it, so the GRES-2 slot matched the Inco page on lead tokens and
    read `Height = 380.0 m` off it anyway -- 29 of the 30 remaining wrong hand-rule rows (210, 212,
    211) are that shape. A page whose URL slug names one of the mandate's OTHER entities is that
    entity's page, and lead-text mentions do not overrule the slug."""
    kit.register_page(INCO_URL, INCO_PAGE_NAMING_GRES2)

    result = kit.host_derive(statement("210"))

    assert result["reason"] == "operand_not_found"
    assert [row["reason"] for row in result["slots"]] == ["no_candidate_page", "selected"]
    assert result["value"] is None and result["node_id"] is None


def test_a_page_naming_both_entities_in_its_lead_is_still_shared_by_both_slots(kit):
    """The guard against over-reach: a comparison page whose SLUG names neither entity is claimed
    by neither, so the lead-token match still serves both slots exactly as it did before."""
    kit.register_page("https://en.wikipedia.org/wiki/List_of_tallest_chimneys",
                      "List of tallest chimneys\nThe Ekibastuz GRES-2 Power Station chimney is "
                      "the tallest; the Inco Superstack is second.\n"
                      "GRES-2\nHeight\n419.7\nm\nInco\nHeight\n380.0\nm\n")

    result = kit.host_derive(statement("210"))

    assert result["reason"] == "computed"
    assert result["value"] == pytest.approx(39.7)
    assert len({row["page_id"] for row in _selected(result)}) == 1


def test_an_argmax_entity_with_no_page_of_its_own_is_skipped_not_read_off_a_neighbour(kit):
    """The same fallback in the argmax path: four of the five rivers were never fetched, so those
    four entities simply do not compete rather than each reading Mekong's numbers."""
    name, slug, length, basin = RIVERS[0]
    kit.register_page(_wiki(slug), _river_page(name, length, basin))

    result = kit.host_derive(statement("218"))

    assert result["reason"] == "operand_not_found"
    assert len(_selected(result)) == 2
    assert [row["reason"] for row in result["slots"] if row["entity"] != "Mekong"] == [
        "no_candidate_page"] * 8


def test_two_slots_reading_the_SAME_field_must_agree_on_their_unit(kit):
    """Task 217 asks "its surface area, in km2" of BOTH lakes, and the two articles write that one
    field in different units (`8,372 km` -- the flattened infobox loses the exponent -- and
    `191 sq mi`). `_compat_quotient` allows two different units on purpose, because a rate like
    km / min is legitimate (task 216), so the narrower rule is on the FIELD PHRASE: when both
    slots read the same field, the two operands must carry the same canonical unit. 19 of the
    replay's 50 wrong hand-rule rows minted `43.83 km/sq mi` from this pair."""
    kit.register_page(TITICACA_URL, TITICACA_PAGE)
    kit.register_page(TAHOE_URL, TAHOE_PAGE)

    result = kit.host_derive(statement("217"))

    assert result["reason"] == "unit_mismatch"
    assert result["value"] is None and result["node_id"] is None
    assert [row["entry"]["unit"] for row in _selected(result)] == ["km", "sq mi"]
    assert kit.artifact()["derivation_refusals"][-1]["code"] == "UNIT_MISMATCH"


def test_a_same_field_pair_that_does_agree_on_its_unit_still_computes(kit):
    """The rule is a unit check, not a ban on same-field mandates: 211 asks ONE field of two
    entities and both pages write metres, so it stays available."""
    kit.register_page(BAIKAL_URL, BAIKAL_PAGE)
    kit.register_page(TANGANYIKA_URL, TANGANYIKA_PAGE)

    assert kit.host_derive(statement("211"))["reason"] == "computed"


def test_a_rate_over_two_DIFFERENT_fields_keeps_its_two_different_units(kit):
    """The guard against over-reach: 216 divides a length in km by a time in minutes. The two
    slots name different fields, so the same-unit rule does not apply and the rate survives."""
    kit.register_page(SHINKANSEN_URL, SHINKANSEN_PAGE_MINUTES)

    result = kit.host_derive(statement("216"))

    assert result["reason"] == "computed"
    assert result["value"] == pytest.approx(515.4 / 141, abs=1e-6)
    assert result["unit"] == "km/min"


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


# --------------------------------------------------------------------------------------------
# The two failure families the 2026-09-08 replay's newly-included rows exposed
# (`docs/handoffs/HOST_DERIVE_REPLAY_2026-09-08.md` section 15.5)
# --------------------------------------------------------------------------------------------

#: Baikal's article as the replay found it: no `Max. depth` row at all, so the only depth the page
#: offers the 211 slot is its AVERAGE one.
BAIKAL_PAGE_AVERAGE_ONLY = """Lake Baikal
Lake Baikal is a rift lake in Siberia, Russia.
Surface area
31,722
km
2
Average depth
744.4
m
"""

#: The same maximum depth written with the unabbreviated qualifier -- `Max.` and `Maximum` are the
#: same field, and the compatibility rule must not read a spelling as a disagreement.
TANGANYIKA_PAGE_SPELLED_OUT = TANGANYIKA_PAGE.replace("Max. depth", "Maximum depth")


def test_two_slots_reading_the_SAME_field_must_agree_on_their_index_LABEL(kit):
    """Task 211 asks each lake's MAXIMUM depth. The replay's Baikal page carried only an
    `Average depth` row and Tanganyika's carried a maximum one; both are metres, so every unit
    check passed and the host summed 744.4 + 1,471 = 2,215.4 m -- two well-supported operands that
    are not comparable as a pair (`HOST_DERIVE_REPLAY_2026-09-08.md` section 15.5, family A). A
    same-field mandate now requires the two LABELS to be compatible too."""
    kit.register_page(BAIKAL_URL, BAIKAL_PAGE_AVERAGE_ONLY)
    kit.register_page(TANGANYIKA_URL, TANGANYIKA_PAGE)

    result = kit.host_derive(statement("211"))

    assert result["reason"] == "operand_field_mismatch"
    assert result["value"] is None and result["node_id"] is None
    assert [row["entry"]["label"] for row in _selected(result)] == ["Average depth", "Max. depth"]
    assert kit.artifact()["derivation_refusals"][-1]["code"] == "OPERAND_FIELD_MISMATCH"
    assert not [node for node in kit.artifact()["nodes"]
                if node["minted_by"] == HOST_DERIVE_TAG], "nothing is minted for a refused cell"


def test_the_same_qualifier_spelled_two_ways_is_one_field_not_a_mismatch(kit):
    """The guard against over-reach: `Max. depth` and `Maximum depth` are the same measurement,
    and a mandate whose two articles abbreviate differently must stay available."""
    kit.register_page(BAIKAL_URL, BAIKAL_PAGE)
    kit.register_page(TANGANYIKA_URL, TANGANYIKA_PAGE_SPELLED_OUT)

    result = kit.host_derive(statement("211"))

    assert result["reason"] == "computed"
    assert result["value"] == pytest.approx(1642.0 + 1470.0)
    assert [row["entry"]["label"] for row in _selected(result)] == ["Max. depth", "Maximum depth"]


def test_a_two_operand_mandate_naming_two_DIFFERENT_fields_is_not_label_checked(kit):
    """The other half of the guard: 216 reads a line length and a journey time off ONE page, so
    the labels SHOULD differ and the rule must not fire."""
    kit.register_page(SHINKANSEN_URL, SHINKANSEN_PAGE)

    result = kit.host_derive(statement("216"))

    assert result["reason"] == "computed"
    assert [row["entry"]["label"] for row in _selected(result)] == ["Line length", "Journey time"]


def test_an_argmax_over_a_partly_resolved_roster_refuses_instead_of_crowning_a_winner(kit):
    """Task 218 names five rivers. The replay's qwen cells fetched pages for two and three of
    them, and the host crowned the Yangtze over a roster that silently excluded the real winner
    (`HOST_DERIVE_REPLAY_2026-09-08.md` section 15.5, family B). An extremum is only as sound as
    its roster: every slot entity must resolve, or the mechanism declines."""
    for name, slug, length, basin in RIVERS[1:4]:
        kit.register_page(_wiki(slug), _river_page(name, length, basin))

    result = kit.host_derive(statement("218"))

    assert result["reason"] == "incomplete_roster"
    assert result["value"] is None and result["node_id"] is None
    assert result["winner_entity"] is None
    unresolved = sorted({row["entity"] for row in result["slots"]
                         if row["reason"] != "selected"})
    assert unresolved == ["Amazon", "Mekong"]
    refusal = kit.artifact()["derivation_refusals"][-1]
    assert refusal["code"] == "INCOMPLETE_ROSTER"
    assert "Amazon" in refusal["message"] and "Mekong" in refusal["message"]
    assert not [node for node in kit.artifact()["nodes"]
                if node["minted_by"] == HOST_DERIVE_TAG], "no per-entity ratio is minted either"


def test_a_roster_too_thin_to_compare_at_all_stays_operand_not_found(kit):
    """The two refusals are layered, and the boundary is pinned: fewer than two entities resolved
    means there was no comparison to make in the first place (an availability outcome, reported as
    `operand_not_found` exactly as before), while `incomplete_roster` is reserved for a comparison
    the host COULD have computed and declined because the roster was missing entries. Both are
    refusals, so the risk accounting is identical either way; only the diagnosis differs."""
    name, slug, length, basin = RIVERS[0]
    kit.register_page(_wiki(slug), _river_page(name, length, basin))

    result = kit.host_derive(statement("218"))

    assert result["reason"] == "operand_not_found"
    assert not [node for node in kit.artifact()["nodes"]
                if node["minted_by"] == HOST_DERIVE_TAG]


def test_a_complete_roster_is_unaffected_by_the_roster_rule(kit):
    """Every one of the five rivers resolves, so 218 computes exactly as it did before."""
    result = _rivers_kit(kit).host_derive(statement("218"))

    assert result["reason"] == "computed"
    assert result["winner_entity"] == "Mekong"
