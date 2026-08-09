"""
Offline unit tests for the page-only height argmax task (test 091) — free, no LLM calls.

Cover the keystone gate (Deriner Dam = tallest of six Turkish dams, the height argmax), in
single- and multi-line layout and via the deliverables[0] primary slot; the fame-decoy wrong
answer (Atatürk Dam, the most famous name, only 4th-tallest) gating every credit-bearing check
to zero while the UN-gated coverage diagnostic is retained; the runner-up (Altınkaya) also
failing the keystone; the "lists all six without picking" case; a rival asserted as the winner
in an adjacent clause failing; the gated height-value check accepting ~249 m and rejecting an
out-of-band value; partial coverage scoring an exact fraction; the visit gate; the fame-decoy /
wide-margin ground-truth invariants; and that the compiled plan is a well-formed six-leaf pure
fan-out DAG with self-describing leaves that leaks no height figure or winner.
"""
from agent.app.idea_tests import test_091_tier5_dam_argmax_turkey as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 6}}


# Full, correct answer (single line): all six heights, all six URLs, the winner asserted at 249 m.
_FULL_SINGLE = (
    "Heights read from each dam's infobox: "
    "Deriner Dam 249 m (https://en.wikipedia.org/wiki/Deriner_Dam); "
    "Altınkaya Dam 195 m (https://en.wikipedia.org/wiki/Alt%C4%B1nkaya_Dam); "
    "Oymapınar Dam 185 m (https://en.wikipedia.org/wiki/Oymap%C4%B1nar_Dam); "
    "Hasan Uğurlu Dam 175 m (https://en.wikipedia.org/wiki/Hasan_U%C4%9Furlu_Dam); "
    "Atatürk Dam 169 m (https://en.wikipedia.org/wiki/Atat%C3%BCrk_Dam); "
    "Karakaya Dam 158 m (https://en.wikipedia.org/wiki/Karakaya_Dam). "
    "Although Atatürk Dam is the most famous, Deriner Dam is the tallest of the six at 249 m."
)

# Same content, MULTI-LINE layout: the winner sits under a 'Tallest:' header on its own line.
_FULL_MULTI = (
    "Height (m), from each infobox:\n"
    "  Deriner Dam - 249 m - https://en.wikipedia.org/wiki/Deriner_Dam\n"
    "  Altınkaya Dam - 195 m - https://en.wikipedia.org/wiki/Alt%C4%B1nkaya_Dam\n"
    "  Oymapınar Dam - 185 m - https://en.wikipedia.org/wiki/Oymap%C4%B1nar_Dam\n"
    "  Hasan Uğurlu Dam - 175 m - https://en.wikipedia.org/wiki/Hasan_U%C4%9Furlu_Dam\n"
    "  Atatürk Dam - 169 m - https://en.wikipedia.org/wiki/Atat%C3%BCrk_Dam\n"
    "  Karakaya Dam - 158 m - https://en.wikipedia.org/wiki/Karakaya_Dam\n"
    "Tallest of the six:\n"
    "  Deriner Dam (249 m)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_height(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_height(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_deliverables_list_primary_slot_drives_keystone():
    r = {
        "deliverables": [
            "Deriner Dam is the tallest of the six dams.",
            "Its height is 249 m.",
            "Deriner Dam 249 m; Altınkaya Dam 195 m; Oymapınar Dam 185 m; "
            "Hasan Uğurlu Dam 175 m; Atatürk Dam 169 m; Karakaya Dam 158 m.",
            "Sources: https://en.wikipedia.org/wiki/Deriner_Dam ; "
            "https://en.wikipedia.org/wiki/Alt%C4%B1nkaya_Dam ; "
            "https://en.wikipedia.org/wiki/Oymap%C4%B1nar_Dam ; "
            "https://en.wikipedia.org/wiki/Hasan_U%C4%9Furlu_Dam ; "
            "https://en.wikipedia.org/wiki/Atat%C3%BCrk_Dam ; "
            "https://en.wikipedia.org/wiki/Karakaya_Dam",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_height(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_fame_decoy_gates_to_zero():
    # The cheap-model error: pick the most famous name (Atatürk Dam) as the "biggest". Even though
    # all six dams and all six heights appear, every credit-bearing check is 0 except the UN-gated
    # coverage diagnostic. (Atatürk is the fame decoy — 169 m, only 4th.)
    wrong = (
        "Atatürk Dam is the most famous dam in Turkey, so Atatürk Dam is the tallest "
        "(https://en.wikipedia.org/wiki/Atat%C3%BCrk_Dam). For completeness: "
        "Deriner Dam 249 m (https://en.wikipedia.org/wiki/Deriner_Dam); "
        "Altınkaya Dam 195 m (https://en.wikipedia.org/wiki/Alt%C4%B1nkaya_Dam); "
        "Oymapınar Dam 185 m (https://en.wikipedia.org/wiki/Oymap%C4%B1nar_Dam); "
        "Hasan Uğurlu Dam 175 m (https://en.wikipedia.org/wiki/Hasan_U%C4%9Furlu_Dam); "
        "Atatürk Dam 169 m; Karakaya Dam 158 m (https://en.wikipedia.org/wiki/Karakaya_Dam)."
    )
    r = _r(wrong)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_height(r, _OBS)["score"] == 0.0   # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0        # gated on keystone
    assert t.validate_coverage(r, _OBS)["score"] == 1.0        # UN-gated: six heights retained


def test_runner_up_gates_to_zero():
    # The runner-up (Altınkaya, 195 m) named as the winner -> keystone 0, gated checks 0.
    r = _r(
        "Altınkaya Dam is the tallest of the six at 195 m "
        "(https://en.wikipedia.org/wiki/Alt%C4%B1nkaya_Dam)."
    )
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_height(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_lists_all_without_picking_fails_keystone():
    r = _r(
        "The six dams are Deriner Dam, Atatürk Dam, Altınkaya Dam, Oymapınar Dam, "
        "Hasan Uğurlu Dam and Karakaya Dam; each is a large dam in Turkey."
    )
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0


def test_rival_asserted_winner_in_adjacent_clause_fails_keystone():
    # A RIVAL asserted as the winner, with Deriner merely mentioned as runner-up in the next
    # clause, must NOT satisfy the keystone. The proximity window is bounded at periods and
    # semicolons and by rival names, and 'taller than Deriner' is excluded.
    for txt in (
        "Atatürk Dam is the tallest; Deriner Dam is second.",
        "Atatürk Dam is the tallest. Deriner Dam is second.",
        "Altınkaya Dam is taller than Deriner Dam.",   # 'than Deriner' -> Deriner loses
    ):
        assert t.validate_keystone_argmax(_r(txt), _OBS)["score"] == 0.0, txt
    # ...while Deriner genuinely asserted as the max still passes, even with a rival in the clause.
    for txt in (
        "The tallest of the six is Deriner Dam.",
        "Atatürk Dam is more famous than Deriner Dam, but the tallest is Deriner Dam.",
        "Deriner Dam, an arch dam on the Çoruh, is the tallest of the six.",
    ):
        assert t.validate_keystone_argmax(_r(txt), _OBS)["score"] == 1.0, txt


def test_height_value_accepts_249_and_rejects_out_of_band():
    base = "Deriner Dam is the tallest of the six. "
    assert t.validate_winner_height(_r(base + "Its height is 249 m."), _OBS)["score"] == 1.0
    assert t.validate_winner_height(_r(base + "Height about 250 m."), _OBS)["score"] == 1.0
    # A wrong value (e.g. reading the reservoir length 249xx or the famous 169 m) is out of band.
    assert t.validate_winner_height(_r(base + "I reported it as 169 m."), _OBS)["score"] == 0.0
    assert t.validate_winner_height(_r(base + "I reported it as 275 m."), _OBS)["score"] == 0.0


def test_wrong_height_value_outside_tolerance_zeroes_value():
    # Correct winner (keystone passes) but the height mis-reported as 200 m, outside 244-254.
    text = (
        "Deriner Dam is the tallest of the six. I (mistakenly) reported it as 200 m. Heights: "
        "Deriner Dam, Altınkaya Dam 195 m, Oymapınar Dam 185 m, Hasan Uğurlu Dam 175 m, "
        "Atatürk Dam 169 m, Karakaya Dam 158 m (https://en.wikipedia.org/wiki/Deriner_Dam)."
    )
    r = _r(text)
    assert t.validate_keystone_argmax(r, _OBS)["passed"]
    assert t.validate_winner_height(r, _OBS)["score"] == 0.0   # 200 not within 244-254
    # coverage: Deriner has no height figure in this text, so 5 of 6 credited
    assert abs(t.validate_coverage(r, _OBS)["score"] - 5 / 6) < 1e-9


def test_partial_coverage_scores_fraction():
    # Only two dams' heights gathered (Deriner + Altınkaya), keystone still asserted -> coverage 2/6.
    text = (
        "Deriner Dam is the tallest of the six at 249 m, well ahead of Altınkaya Dam's 195 m "
        "(https://en.wikipedia.org/wiki/Deriner_Dam)."
    )
    r = _r(text)
    assert abs(t.validate_coverage(r, _OBS)["score"] - 2 / 6) < 1e-9
    assert t.validate_keystone_argmax(r, _OBS)["passed"]
    assert t.validate_winner_height(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_answer_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_argmax(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_argmax(r, ungrounded_obs)["passed"] is False
    assert t.validate_winner_height(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_argmax(r, ungrounded_obs)["score"],
        t.validate_coverage(r, ungrounded_obs)["score"],
        t.validate_winner_height(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]
    assert t.validate_visits(r, {"visit": {"count": 6}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"]


def test_fame_decoy_wide_margin_invariants():
    # Ground-truth design invariants: the height winner is the tallest, is NOT the most famous
    # dam in the set, and clears 2nd place by a wide margin.
    ranked = sorted(t.ENTITIES, key=lambda e: e["height_m"], reverse=True)
    assert ranked[0] is t.WINNER and t.WINNER["name"] == "Deriner Dam"
    famous = next(e for e in t.ENTITIES if e["key"] == "ataturk")
    assert famous is not t.WINNER                       # fame decoy != winner
    assert famous["height_m"] == 169 and ranked.index(famous) == 4  # Atatürk is only 5th-ranked (0-based 4)
    margin = (t.WINNER["height_m"] - ranked[1]["height_m"]) / ranked[1]["height_m"]
    assert margin > 0.15, margin                        # design floor
    assert abs(margin - (249 - 195) / 195) < 1e-9       # ~27.7%


def test_compiled_plan_validates_and_is_six_leaf_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True             # six independent dams, one parallel wave
    assert struct["waves"] == [
        ["deriner", "altinkaya", "oymapinar", "hasan_ugurlu", "ataturk", "karakaya"]
    ]


def test_compiled_plan_leaves_self_describe_entity():
    # Each leaf's own answer text ('expect') must restate WHICH dam it is about, so aggregation
    # binding survives leaf-ID stripping (the sibling-test aggregation-binding bug fix).
    plan = t.get_compiled_plan()
    for leaf, e in zip(plan["leaves"], t.ENTITIES):
        assert e["name"] in leaf["expect"], (e["name"], leaf["expect"])
        assert e["name"] in leaf["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the six GIVEN dams and their regions may appear, but no height figure and
    # no statement of which dam won.
    for leak in ("249", "195", "185", "175", "169", "158"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # No pre-stated winner: the aggregation does not tie Deriner to a 'tallest' assertion.
    assert not t._DERINER_WINS.search(plan["aggregation"])
