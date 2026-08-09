"""Offline unit tests for test 115 (largest deserts polar trap -> Antarctic -> Vostok record low)."""
from agent.app.idea_tests import test_115_tier5_largest_desert_polar_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: the Sahara is the largest HOT desert, ~9,200,000 km² "
    "(https://en.wikipedia.org/wiki/Sahara); the Arabian Desert is a subtropical hot desert on the "
    "Arabian Peninsula (https://en.wikipedia.org/wiki/Arabian_Desert); the Gobi is a cold desert in "
    "a rain shadow (https://en.wikipedia.org/wiki/Gobi_Desert); the Antarctic Desert is the largest "
    "desert, a polar desert ~14,200,000 km² (https://en.wikipedia.org/wiki/Polar_desert) — the "
    "survivor. Stage 2: its coldest point hosts Vostok Station "
    "(https://en.wikipedia.org/wiki/Vostok_Station). Stage 3: the record low there is -89.2 °C."
)

_FULL_MULTI = (
    "STAGE 1 — deserts:\n"
    "  Sahara Desert -> largest hot desert, ~9,200,000 km²\n"
    "    https://en.wikipedia.org/wiki/Sahara\n"
    "  Arabian Desert -> subtropical hot desert (Arabian Peninsula)\n"
    "    https://en.wikipedia.org/wiki/Arabian_Desert\n"
    "  Gobi Desert -> cold desert, rain shadow\n"
    "    https://en.wikipedia.org/wiki/Gobi_Desert\n"
    "  Antarctic Desert -> largest, polar desert ~14,200,000 km²  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Polar_desert\n"
    "STAGE 2 — station:\n"
    "  Vostok Station\n"
    "    https://en.wikipedia.org/wiki/Vostok_Station\n"
    "STAGE 3 — record low:\n"
    "  -89.2\n"
    "  degrees C\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_temp(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_station(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_temp(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_station(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Sahara Desert -> largest hot desert ~9,200,000 km²; Arabian Desert -> subtropical hot "
        "desert; Gobi Desert -> cold desert rain shadow; Antarctic Desert -> polar desert "
        "~14,200,000 km². The largest desert is the Sahara; it is very hot."
    )
    r = _r(wrong)
    assert t.validate_keystone_temp(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_station(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_temp(_r("elevation 3,488 m"), _OBS)["score"] == 0.0
    assert t.validate_keystone_temp(_r("area 14,200,000 km2"), _OBS)["score"] == 0.0
    assert t.validate_keystone_temp(_r("record low -89.2 C"), _OBS)["score"] == 1.0


def test_partial_branch_exploration():
    text = ("The Sahara is the largest hot desert (~9,200,000 km²). The Antarctic Desert is a polar "
            "desert. The remaining two candidates were not checked.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_temp(r, _OBS)["score"] == 0.0
    assert t.validate_survivor_and_station(r, _OBS)["score"] == 0.0


def test_branch_exploration_requires_visits():
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_branch_exploration(r, {"visit": {"count": 4}})["score"] == 1.0


def test_survivor_and_station_needs_both():
    # Keystone present, station named but survivor desert not -> partial 0.5.
    text = "The record low at Vostok Station is -89.2 °C."
    r = _r(text)
    assert abs(t.validate_survivor_and_station(r, _OBS)["score"] - 0.5) < 1e-9


def test_visits_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_compiled_plan_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor_station"]
    assert struct["waves"][2] == ["station_record"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_sahara", "cand_arabian", "cand_antarctic", "cand_gobi"):
        assert "{" + key + "}" in by_id["survivor_station"]["instruction"]
    assert "{survivor_station}" in by_id["station_record"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("89.2", "vostok"):
        assert leak not in blob, f"plan leaks {leak!r}"
