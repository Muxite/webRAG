"""
Offline unit tests for the wide-margin computed-ratio argmax task (test 064) — free.

Cover the keystone gate (Issyk-Kul = highest volume-to-surface-area ratio, the computed-ratio
argmax), in single- and multi-line layout and via the deliverables[0] primary slot; the wrong
answers that shortcut to the biggest raw quantity (Lake Victoria = largest by BOTH volume AND
surface area) gating every credit-bearing check to zero while the UN-gated coverage diagnostic is
retained; a different wrong lake (Lake Erie) also failing the keystone; the "lists all five without
picking" case failing the keystone; a rival asserted as the winner in an adjacent clause failing;
the gated ratio-value check accepting ~278 m and ~0.278 km and rejecting a value outside tolerance;
partial coverage scoring an exact fraction; the visit gate; the WIDE-MARGIN / double-decoy ground-
truth invariants; and that the compiled plan is a well-formed five-leaf pure-fan-out DAG that leaks
no figure, ratio or winner.
"""
from agent.app.idea_tests import test_064_tier5_computed_ratio_argmax_wide as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


# Full, correct answer (single line): all ten figures, all five URLs, the winner asserted, ratio ~278 m.
_FULL_SINGLE = (
    "Volume-to-surface-area ratios (mean depth) computed from each lake's infobox figures: "
    "Issyk-Kul 1,736 km3 / 6,236 km2 (https://en.wikipedia.org/wiki/Issyk-Kul); "
    "Lake Victoria 2,424 km3 / 59,947 km2 (https://en.wikipedia.org/wiki/Lake_Victoria); "
    "Lake Ladoga 837 km3 / 17,891 km2 (https://en.wikipedia.org/wiki/Lake_Ladoga); "
    "Lake Erie 480 km3 / 25,667 km2 (https://en.wikipedia.org/wiki/Lake_Erie); "
    "Lake Winnipeg 294 km3 / 24,514 km2 (https://en.wikipedia.org/wiki/Lake_Winnipeg). "
    "Although Lake Victoria is the largest by both volume and surface area, Issyk-Kul has the "
    "highest volume-to-surface-area ratio, about 278 m (1,736/6,236 = 0.278 km)."
)

# Same content, MULTI-LINE layout: the winner sits under a 'Highest ... ratio:' header on its own line.
_FULL_MULTI = (
    "Volume / surface area (mean depth), from each infobox:\n"
    "  Issyk-Kul - 1,736 km3 / 6,236 km2 - https://en.wikipedia.org/wiki/Issyk-Kul\n"
    "  Lake Victoria - 2,424 km3 / 59,947 km2 - https://en.wikipedia.org/wiki/Lake_Victoria\n"
    "  Lake Ladoga - 837 km3 / 17,891 km2 - https://en.wikipedia.org/wiki/Lake_Ladoga\n"
    "  Lake Erie - 480 km3 / 25,667 km2 - https://en.wikipedia.org/wiki/Lake_Erie\n"
    "  Lake Winnipeg - 294 km3 / 24,514 km2 - https://en.wikipedia.org/wiki/Lake_Winnipeg\n"
    "Highest volume-to-surface-area ratio:\n"
    "  Issyk-Kul (about 278 m)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer, newline-heavy layout: the keystone ("Issyk-Kul" under "Highest ... ratio:"),
    # all ten figures, the ratio value and all five citations must still register identically.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_deliverables_list_primary_slot_drives_keystone():
    # Keystone reads deliverables[0]; coverage/value/citation read the whole deliverables list.
    r = {
        "deliverables": [
            "Issyk-Kul has the highest volume-to-surface-area ratio of the five.",
            "Its ratio is about 278 m (1,736 km3 / 6,236 km2 = 0.278 km).",
            "Issyk-Kul 1,736 / 6,236; Lake Victoria 2,424 / 59,947; Lake Ladoga 837 / 17,891; "
            "Lake Erie 480 / 25,667; Lake Winnipeg 294 / 24,514.",
            "Sources: https://en.wikipedia.org/wiki/Issyk-Kul ; "
            "https://en.wikipedia.org/wiki/Lake_Victoria ; https://en.wikipedia.org/wiki/Lake_Ladoga ; "
            "https://en.wikipedia.org/wiki/Lake_Erie ; https://en.wikipedia.org/wiki/Lake_Winnipeg",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_biggest_volume_shortcut_gates_to_zero():
    # The cheap-model error: shortcut to the biggest raw VOLUME (Lake Victoria, 2,424 km3) and name
    # it. Even though all five lakes and all ten figures appear, every credit-bearing check is 0
    # except the UN-gated coverage diagnostic. (Victoria is the biggest-volume decoy.)
    wrong = (
        "Lake Victoria holds the most water of the five - 2,424 km3 over a 59,947 km2 surface - so "
        "Lake Victoria has the deepest profile and is the answer "
        "(https://en.wikipedia.org/wiki/Lake_Victoria). For completeness: Issyk-Kul 1,736 km3 / "
        "6,236 km2 (https://en.wikipedia.org/wiki/Issyk-Kul); Lake Ladoga 837 / 17,891 "
        "(https://en.wikipedia.org/wiki/Lake_Ladoga); Lake Erie 480 / 25,667 "
        "(https://en.wikipedia.org/wiki/Lake_Erie); Lake Winnipeg 294 / 24,514 "
        "(https://en.wikipedia.org/wiki/Lake_Winnipeg)."
    )
    r = _r(wrong)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 0.0   # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0       # gated on keystone
    assert t.validate_coverage(r, _OBS)["score"] == 1.0       # UN-gated: ten figures retained


def test_biggest_area_shortcut_gates_to_zero():
    # The other raw-quantity trap: pick the largest-SURFACE-AREA lake (Lake Victoria, 59,947 km2).
    # The biggest-volume and biggest-area lakes are the SAME giant (Victoria), so this is the second
    # framing of the doubly-baited decoy. Winner (Issyk-Kul) absent from the primary slot -> every
    # gated check is 0.
    r = {
        "deliverables": [
            "Lake Victoria has the largest surface area (59,947 km2), so it has the highest ratio.",
            "Victoria 59,947 km2, 2,424 km3; Issyk-Kul 6,236 / 1,736; Ladoga 17,891 / 837; "
            "Erie 25,667 / 480; Winnipeg 24,514 / 294.",
            "https://en.wikipedia.org/wiki/Lake_Victoria",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 0.0   # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0       # gated on keystone


def test_wrong_entity_gates_to_zero():
    # A different wrong lake (Lake Erie, a salient named Great Lake) named as the winner -> keystone 0.
    r = _r(
        "Lake Erie has the highest volume-to-surface-area ratio of the five "
        "(https://en.wikipedia.org/wiki/Lake_Erie)."
    )
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_ratio(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_lists_all_without_picking_fails_keystone():
    # Names all five but never asserts which ratio is highest -> keystone not satisfied.
    r = _r(
        "The five lakes are Issyk-Kul, Lake Victoria, Lake Ladoga, Lake Erie and Lake Winnipeg; "
        "each is a sizeable body of water."
    )
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0


def test_rival_asserted_winner_in_adjacent_clause_fails_keystone():
    # Hardening: a RIVAL asserted as the winner, with Issyk-Kul merely mentioned as runner-up in the
    # next clause, must NOT satisfy the keystone. The proximity window is bounded at both sentence
    # periods and clause-separating semicolons, so "the highest ratio" (Victoria's) cannot reach an
    # Issyk-Kul mention in the next clause.
    for txt in (
        "Lake Victoria has the highest volume-to-area ratio; Issyk-Kul is second.",
        "Lake Victoria has the highest volume-to-area ratio. Issyk-Kul is second.",
        "Lake Ladoga is deeper on average than Issyk-Kul.",  # 'than Issyk-Kul' -> Issyk-Kul loses
    ):
        assert t.validate_keystone_argmax(_r(txt), _OBS)["score"] == 0.0, txt
    # ...while Issyk-Kul genuinely asserted as the max still passes, even with a rival in the clause.
    for txt in (
        "The highest volume-to-surface-area ratio belongs to Issyk-Kul.",
        "Lake Victoria has more water than Issyk-Kul, but the highest ratio is Issyk-Kul.",
        "Issyk-Kul, with the deepest mean depth, has the highest ratio of the five.",
    ):
        assert t.validate_keystone_argmax(_r(txt), _OBS)["score"] == 1.0, txt


def test_ratio_value_accepts_metres_and_km():
    # The gated value check accepts the winner's ratio expressed in metres (~278) or kilometres
    # (~0.278); a wrong value (e.g. computing area/volume) is rejected.
    base = "Issyk-Kul has the highest volume-to-surface-area ratio. "
    assert t.validate_winner_ratio(_r(base + "Its mean depth is 278.4 m."), _OBS)["score"] == 1.0
    assert t.validate_winner_ratio(_r(base + "The ratio is about 280 m."), _OBS)["score"] == 1.0
    assert t.validate_winner_ratio(_r(base + "1,736/6,236 = 0.278 km."), _OBS)["score"] == 1.0
    # Inverted (area/volume = 6,236/1,736 = 3.59) is outside the band -> value not credited.
    assert t.validate_winner_ratio(_r(base + "I got 3.59 (6,236/1,736)."), _OBS)["score"] == 0.0


def test_wrong_ratio_value_outside_tolerance_zeroes_value():
    # Correct winner (keystone passes) but the ratio is reported as 147 (Lake Superior-style mid
    # value / a mis-read), outside the +/- 3% window around 278 m. The value check zeroes; coverage
    # stays.
    text = (
        "Issyk-Kul has the highest volume-to-surface-area ratio of the five. I (mistakenly) "
        "reported it as 147 m. Figures: Issyk-Kul 1,736 km3 and 6,236 km2; "
        "Lake Victoria 2,424 km3 and 59,947 km2; Lake Ladoga 837 km3 and 17,891 km2; "
        "Lake Erie 480 km3 and 25,667 km2; Lake Winnipeg 294 km3 and 24,514 km2 "
        "(https://en.wikipedia.org/wiki/Issyk-Kul)."
    )
    r = _r(text)
    assert t.validate_keystone_argmax(r, _OBS)["passed"]
    assert t.validate_winner_ratio(r, _OBS)["score"] == 0.0   # 147 not within ~270-287 m
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    # Only two lakes' figures gathered (Issyk-Kul + Victoria), keystone still asserted -> coverage 2/5.
    text = (
        "Issyk-Kul has the highest volume-to-surface-area ratio, about 278 m: 1,736 km3 over "
        "6,236 km2, far ahead of Lake Victoria's 2,424 km3 over 59,947 km2 "
        "(https://en.wikipedia.org/wiki/Issyk-Kul)."
    )
    r = _r(text)
    assert abs(t.validate_coverage(r, _OBS)["score"] - 0.4) < 1e-9   # 2 of 5 lakes
    assert t.validate_keystone_argmax(r, _OBS)["passed"]
    assert t.validate_winner_ratio(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]
    assert t.validate_visits(r, {"visit": {"count": 5}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"]


def test_wide_margin_double_decoy_invariants():
    # The ground-truth design invariants: the ratio winner is NEITHER the biggest-volume lake NOR
    # the biggest-area lake, and it clears 2nd place by a wide margin (the 059 noise fix).
    by_vol = max(t.ENTITIES, key=lambda e: e["volume"])
    by_area = max(t.ENTITIES, key=lambda e: e["area"])
    ranked = sorted(t.ENTITIES, key=lambda e: e["volume"] / e["area"], reverse=True)
    assert ranked[0] is t.WINNER                         # Issyk-Kul wins the ratio
    assert by_vol is not t.WINNER and by_vol["name"] == "Lake Victoria"   # decoy A
    assert by_area is not t.WINNER and by_area["name"] == "Lake Victoria"  # decoy B
    w = t.WINNER["volume"] / t.WINNER["area"]
    second = ranked[1]["volume"] / ranked[1]["area"]
    margin = (w - second) / second
    assert margin > 0.15, margin                          # design floor
    assert margin > 4.0, margin                           # actually ~4.95 (495%) -- wide-margin fix


def test_compiled_plan_validates_and_is_five_leaf_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 5
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True            # five independent lakes, one parallel wave
    assert struct["waves"] == [["issyk_kul", "victoria", "ladoga", "erie", "winnipeg"]]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the five GIVEN lakes and their regions may appear, but no volume figure, no
    # area figure, no ratio, and no statement of which lake won.
    for leak in ("1736", "1,736", "6236", "6,236", "2424", "2,424", "59947", "59,947",
                 "837", "17891", "17,891", "480", "25667", "25,667", "294", "24514", "24,514",
                 "278", "0.278", "46.8", "40.4"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # No pre-stated winner: the aggregation does not tie Issyk-Kul to a 'highest ratio' assertion.
    assert not t._ISSYK_WINS.search(plan["aggregation"])
