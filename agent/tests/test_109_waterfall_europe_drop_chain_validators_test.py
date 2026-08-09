"""
Offline unit tests for the record-waterfall branch-then-chain task (test 109) — free, no LLM.

Covers the leak-resistant keystone gate (the survivor's tallest single-drop height, 575 m — distinct
from the 845 m total), the UN-gated elimination-coverage diagnostic (four waterfalls, capped by
visits), the keystone-gated survivor/drop and citation secondaries, single- and multi-line layouts,
the fame decoy (electing Angel Falls) and the total-vs-drop confusion (reporting 845 m). The compiled
plan is a genuine branch-then-chain DAG (4 -> 1 -> 1) that templates upstream, is self-describing,
and leaks nothing.
"""
from agent.app.idea_tests import test_109_tier5_waterfall_europe_drop_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: Angel Falls (https://en.wikipedia.org/wiki/Angel_Falls) is in South America; Tugela "
    "Falls (https://en.wikipedia.org/wiki/Tugela_Falls) in Africa; Yosemite Falls "
    "(https://en.wikipedia.org/wiki/Yosemite_Falls) in North America; Vinnufossen "
    "(https://en.wikipedia.org/wiki/Vinnufossen) in Norway, Europe — the survivor. Stage 3: its "
    "tallest single uninterrupted drop is 575 m (the total is 845 m)."
)

_FULL_MULTI = (
    "STAGE 1 — continent:\n"
    "  Angel Falls -> South America\n"
    "    https://en.wikipedia.org/wiki/Angel_Falls\n"
    "  Tugela Falls -> Africa\n"
    "    https://en.wikipedia.org/wiki/Tugela_Falls\n"
    "  Yosemite Falls -> North America\n"
    "    https://en.wikipedia.org/wiki/Yosemite_Falls\n"
    "  Vinnufossen -> Europe (Norway)  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Vinnufossen\n"
    "STAGE 3 — tallest single drop:\n"
    "  575\n"
    "  metres (single uninterrupted drop)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_drop(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_drop(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_drop(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_drop(r, _OBS)["score"] == 1.0


def test_fame_decoy_angel_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Angel Falls, Tugela Falls, Yosemite Falls and Vinnufossen all checked. I take the famous "
        "Angel Falls as the world's highest; its total height is 979 m."
    )
    r = _r(wrong)
    assert t.validate_keystone_drop(r, _OBS)["score"] == 0.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_drop(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_total_height_instead_of_single_drop_fails():
    # Reporting the survivor's TOTAL height (845 m) instead of the single drop (575 m) must fail.
    text = (
        "Survivor: Vinnufossen in Norway, a tiered waterfall; its total height is 845 m."
    )
    r = _r(text)
    assert t.validate_keystone_drop(r, _OBS)["passed"] is False
    assert t.validate_survivor_and_drop(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_numbers():
    assert t.validate_keystone_drop(_r("code 5750 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_drop(_r("marker 1575 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_drop(_r("total 845 m"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = "I checked only Angel Falls and Vinnufossen; not the other two waterfalls."
    r = _r(text)
    assert abs(t.validate_elimination_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_drop(r, _OBS)["score"] == 0.0


def test_elimination_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_elimination_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_elimination_coverage(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_elimination_coverage(r, {"visit": {"count": 4}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 4}})["score"] - (4 / 5)) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Vinnufossen tallest single drop: 575 m", "survivor: Vinnufossen"]}
    assert t.validate_keystone_drop(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor"]
    assert struct["waves"][2] == ["single_drop"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("falls_angel", "falls_tugela", "falls_yosemite", "falls_vinnufossen"):
        assert "{" + key + "}" in by_id["survivor"]["instruction"]
    assert "{survivor}" in by_id["single_drop"]["instruction"]
    assert "single-drop" in by_id["single_drop"]["expect"].lower() or \
           "single" in by_id["single_drop"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # 'vinnufossen' IS a GIVEN candidate name (allowed); the leak concern is the answer figures and
    # pre-labelling which candidate is the European survivor.
    for leak in ("575", "845", "norway", "europe (norway)"):
        assert leak not in blob, f"plan leaks {leak!r}"
