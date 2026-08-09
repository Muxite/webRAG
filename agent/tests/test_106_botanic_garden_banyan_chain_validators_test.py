"""
Offline unit tests for the Royal Botanic Gardens branch-then-chain task (test 106) — free, no LLM.

Covers the leak-resistant keystone gate (the Great Banyan's prop-root count 3,772 / canopy 18,918
m²), the UN-gated elimination-coverage diagnostic (four gardens, capped by visits), the
keystone-gated survivor/tree and citation secondaries, single- and multi-line layouts, and the fame
decoy (electing Kew). The compiled plan is a genuine branch-then-chain DAG (4 -> 1 -> 1) that
templates upstream, is self-describing, and leaks nothing (including not naming the Great Banyan).
"""
from agent.app.idea_tests import test_106_tier5_botanic_garden_banyan_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: Kew (https://en.wikipedia.org/wiki/Royal_Botanic_Gardens,_Kew) holds no such record; "
    "Edinburgh (https://en.wikipedia.org/wiki/Royal_Botanic_Garden_Edinburgh) none; Sydney "
    "(https://en.wikipedia.org/wiki/Royal_Botanic_Garden,_Sydney) none; the Kolkata / Calcutta "
    "garden (https://en.wikipedia.org/wiki/Acharya_Jagadish_Chandra_Bose_Indian_Botanic_Garden) "
    "holds the Great Banyan (https://en.wikipedia.org/wiki/The_Great_Banyan) — the survivor. "
    "Stage 3: the Great Banyan has 3,772 prop roots (canopy 18,918 m²)."
)

_FULL_MULTI = (
    "STAGE 1 — record tree?\n"
    "  Royal Botanic Gardens, Kew -> no\n"
    "    https://en.wikipedia.org/wiki/Royal_Botanic_Gardens,_Kew\n"
    "  Royal Botanic Garden Edinburgh -> no\n"
    "    https://en.wikipedia.org/wiki/Royal_Botanic_Garden_Edinburgh\n"
    "  Royal Botanic Garden, Sydney -> no\n"
    "    https://en.wikipedia.org/wiki/Royal_Botanic_Garden,_Sydney\n"
    "  Acharya J. C. Bose Indian Botanic Garden, Kolkata -> Great Banyan  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Acharya_Jagadish_Chandra_Bose_Indian_Botanic_Garden\n"
    "STAGE 3 — Great Banyan figure:\n"
    "    https://en.wikipedia.org/wiki/The_Great_Banyan\n"
    "  3,772\n"
    "  prop roots\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_banyan(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_tree(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_banyan(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_tree(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_canopy_area_alternative_satisfies_keystone():
    r = _r("The Great Banyan occupies a canopy area of about 18,918 square metres.")
    assert t.validate_keystone_banyan(r, _OBS)["score"] == 1.0


def test_fame_decoy_kew_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Kew, Edinburgh, Sydney and the Kolkata garden all checked. I take the famous Kew as the "
        "answer; its arboretum has thousands of trees over 132 hectares."
    )
    r = _r(wrong)
    assert t.validate_keystone_banyan(r, _OBS)["score"] == 0.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_tree(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_numbers():
    assert t.validate_keystone_banyan(_r("code 13,772 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_banyan(_r("area 18,9180 marker"), _OBS)["score"] == 0.0
    assert t.validate_keystone_banyan(_r("circumference 486 m"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = "I checked only Kew and the Kolkata garden; not the other two gardens."
    r = _r(text)
    assert abs(t.validate_elimination_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_banyan(r, _OBS)["score"] == 0.0


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
         "deliverables": ["Great Banyan: 3,772 prop roots", "survivor: Kolkata garden"]}
    assert t.validate_keystone_banyan(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor"]
    assert struct["waves"][2] == ["tree_figure"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("garden_kew", "garden_edinburgh", "garden_sydney", "garden_kolkata"):
        assert "{" + key + "}" in by_id["survivor"]["instruction"]
    assert "{survivor}" in by_id["tree_figure"]["instruction"]
    assert "prop-root" in by_id["tree_figure"]["expect"].lower() or \
           "canopy" in by_id["tree_figure"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # Must not leak the figure NOR pre-name the round-2 discovery (the Great Banyan).
    for leak in ("3772", "3,772", "18918", "18,918", "486", "banyan"):
        assert leak not in blob, f"plan leaks {leak!r}"
