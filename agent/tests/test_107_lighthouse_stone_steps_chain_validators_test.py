"""
Offline unit tests for the tall-lighthouse branch-then-chain task (test 107) — free, no LLM.

Covers the leak-resistant keystone gate (the survivor's 360 stone steps / 82.5 m granite tower),
the UN-gated elimination-coverage diagnostic (four lighthouses, capped by visits), the keystone-
gated survivor/material and citation secondaries, single- and multi-line layouts, and the fame decoy
(electing the steel Jeddah Light). The compiled plan is a genuine branch-then-chain DAG (4 -> 1 -> 1)
that templates upstream, is self-describing, and leaks nothing.
"""
from agent.app.idea_tests import test_107_tier5_lighthouse_stone_steps_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: Jeddah Light (https://en.wikipedia.org/wiki/Jeddah_Light) is steel; Yokohama Marine "
    "Tower (https://en.wikipedia.org/wiki/Yokohama_Marine_Tower) is a steel tower; Lange Nelle at "
    "Ostend (https://en.wikipedia.org/wiki/Lange_Nelle) is concrete; the Phare de l'Île Vierge "
    "(https://en.wikipedia.org/wiki/%C3%8Ele_Vierge_Lighthouse) is granite — the tallest stone "
    "lighthouse, the survivor. Stage 3: inside, 360 steps of stone lead up the 82.5 m tower."
)

_FULL_MULTI = (
    "STAGE 1 — material:\n"
    "  Jeddah Light -> steel\n"
    "    https://en.wikipedia.org/wiki/Jeddah_Light\n"
    "  Yokohama Marine Tower -> steel\n"
    "    https://en.wikipedia.org/wiki/Yokohama_Marine_Tower\n"
    "  Lange Nelle (Ostend) -> concrete\n"
    "    https://en.wikipedia.org/wiki/Lange_Nelle\n"
    "  Phare de l'Île Vierge -> granite  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/%C3%8Ele_Vierge_Lighthouse\n"
    "STAGE 3 — interior:\n"
    "  360 steps\n"
    "  of stone\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_steps(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_material(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_steps(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_material(r, _OBS)["score"] == 1.0


def test_height_alternative_satisfies_keystone():
    r = _r("The Île Vierge granite tower is 82.5 metres tall.")
    assert t.validate_keystone_steps(r, _OBS)["score"] == 1.0


def test_fame_decoy_jeddah_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Jeddah Light, Yokohama Marine Tower, Lange Nelle (Ostend) and the Phare de l'Île Vierge all "
        "checked. I take the famous Jeddah Light as tallest; it stands 133 metres."
    )
    r = _r(wrong)
    assert t.validate_keystone_steps(r, _OBS)["score"] == 0.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_material(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_bare_360_and_embedded():
    # A bare compass 360 (no steps/stone nearby) must NOT satisfy; 82.50 lacks a trailing boundary.
    assert t.validate_keystone_steps(_r("a compass bearing of 360 degrees"), _OBS)["score"] == 0.0
    assert t.validate_keystone_steps(_r("height 82.50 marker"), _OBS)["score"] == 0.0
    # But '360 steps' or 'steps ... 360' does satisfy.
    assert t.validate_keystone_steps(_r("360 steps to the top"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    text = "I checked only Jeddah Light and the Phare de l'Île Vierge; not the other two towers."
    r = _r(text)
    assert abs(t.validate_elimination_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_steps(r, _OBS)["score"] == 0.0


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
         "deliverables": ["Île Vierge: 360 steps of stone", "survivor: Île Vierge"]}
    assert t.validate_keystone_steps(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor"]
    assert struct["waves"][2] == ["interior_figure"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("lh_jeddah", "lh_yokohama", "lh_ostend", "lh_vierge"):
        assert "{" + key + "}" in by_id["survivor"]["instruction"]
    assert "{survivor}" in by_id["interior_figure"]["instruction"]
    assert "step" in by_id["interior_figure"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("360", "82.5", "365", "271"):
        assert leak not in blob, f"plan leaks {leak!r}"
