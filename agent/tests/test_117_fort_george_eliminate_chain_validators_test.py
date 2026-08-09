"""Offline unit tests for test 117 (Fort George -> Highland fort construction budget). Free, no LLM.

Covers the leak-resistant keystone gate (£92,673), the UN-gated candidate-coverage diagnostic, the
keystone-gated survivor/citation secondaries, single- and multi-line layout, the famous-decoy failure
mode (picking the Ontario War-of-1812 fort), keystone token rejecting near-miss numbers, visit gating,
and the compiled plan (branch-then-chain, templated, self-describing, leaks nothing).
"""
from agent.app.idea_tests import test_117_tier5_fort_george_eliminate_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: Fort George, Ontario is the War of 1812 fort at Niagara "
    "(https://en.wikipedia.org/wiki/Fort_George_(Ontario)); Fort George near Ardersier, Inverness is the "
    "Georgian bastion fort built 1748 after Culloden (Jacobite rising) "
    "(https://en.wikipedia.org/wiki/Fort_George,_Highland); Fort George (Manhattan) is the colonial "
    "New York fort (https://en.wikipedia.org/wiki/Fort_George_(Manhattan)); Fort George, Guernsey is the "
    "Channel Islands garrison at Saint Peter Port (https://en.wikipedia.org/wiki/Fort_George,_Guernsey). "
    "Stage 2: survivor is the Highland fort. Stage 3: its original budget was £92,673 19s 1d."
)

_FULL_MULTI = (
    "STAGE 1:\n"
    "  Fort George, Ontario -> War of 1812, Niagara, canada\n"
    "    https://en.wikipedia.org/wiki/Fort_George_(Ontario)\n"
    "  Fort George, Highland (Ardersier, Inverness) -> Georgian bastion, 1748, culloden, jacobite\n"
    "    https://en.wikipedia.org/wiki/Fort_George,_Highland\n"
    "  Fort George (Manhattan) -> colonial New York fort\n"
    "    https://en.wikipedia.org/wiki/Fort_George_(Manhattan)\n"
    "  Fort George, Guernsey -> Channel Islands garrison, Saint Peter Port\n"
    "    https://en.wikipedia.org/wiki/Fort_George,_Guernsey\n"
    "STAGE 2 survivor: Highland fort\n"
    "STAGE 3 budget:\n  £92,673\n  19s 1d\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone(r, _OBS)["score"] == 1.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone(r, _OBS)["score"] == 1.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_famous_decoy_survivor_gates_to_zero_but_keeps_coverage():
    wrong = (
        "Fort George, Ontario -> War of 1812 Niagara canada; Fort George, Highland (Ardersier Inverness) "
        "-> Georgian bastion 1748 culloden jacobite; Fort George (Manhattan) -> colonial New York; Fort "
        "George, Guernsey -> Channel Islands Saint Peter. I pick the battle-famous Ontario War of 1812 "
        "fort; its garrison held some hundreds of men."
    )
    r = _r(wrong)
    assert t.validate_keystone(r, _OBS)["score"] == 0.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_near_miss_numbers():
    assert t.validate_keystone(_r("cost was 192,673 pounds"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("final cost more than 200,000"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("original budget £92,673 19s 1d"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    text = (
        "Fort George, Ontario -> War of 1812 Niagara; Fort George, Highland (Ardersier) -> Georgian "
        "bastion 1748 culloden. I did not investigate the remaining two forts."
    )
    r = _r(text)
    assert abs(t.validate_candidate_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone(r, _OBS)["score"] == 0.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0


def test_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_candidate_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_candidate_coverage(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_candidate_coverage(r, {"visit": {"count": 4}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Original budget: £92,673", "survivor: Highland"]}
    assert t.validate_keystone(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["election"]
    assert struct["waves"][2] == ["keystone_budget"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_ontario", "cand_highland", "cand_newyork", "cand_guernsey"):
        assert "{" + key + "}" in by_id["election"]["instruction"]
    assert "{election}" in by_id["keystone_budget"]["instruction"]
    assert "budget" in by_id["keystone_budget"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("92,673", "92673", "200,000"):
        assert leak not in blob, f"plan leaks {leak!r}"
