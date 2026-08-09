"""Offline unit tests for test 120 (Cleopatra's Needles -> NY obelisk transit days). Free, no LLM.

Covers the leak-resistant keystone gate (112-day transit), the UN-gated candidate-coverage
diagnostic, the keystone-gated survivor/citation secondaries, single- and multi-line layout, the
famous-decoy failure mode (picking the London Needle) and the Paris/Vatican mis-identification traps,
keystone token rejecting near-miss numbers, visit gating, and the compiled plan (branch-then-chain,
templated, self-describing, leaks nothing).
"""
from agent.app.idea_tests import test_120_tier5_obelisk_eliminate_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: Cleopatra's Needle, London stands on the Thames Embankment "
    "(https://en.wikipedia.org/wiki/Cleopatra's_Needle,_London); Cleopatra's Needle, New York City "
    "stands in Central Park in the western hemisphere on Greywacke Knoll "
    "(https://en.wikipedia.org/wiki/Cleopatra's_Needle_(New_York_City)); the Luxor Obelisk on the Place "
    "de la Concorde in Paris is NOT a Cleopatra's Needle "
    "(https://en.wikipedia.org/wiki/Luxor_Obelisk); the Vatican Obelisk in St. Peter's Square, Rome is "
    "NOT a Cleopatra's Needle (https://en.wikipedia.org/wiki/Vatican_Obelisk). Stage 2: survivor is the "
    "New York obelisk. Stage 3: it took 112 days to move it to Central Park."
)

_FULL_MULTI = (
    "STAGE 1:\n"
    "  Cleopatra's Needle, London -> a Needle, on the thames embankment (eastern hemisphere)\n"
    "    https://en.wikipedia.org/wiki/Cleopatra's_Needle,_London\n"
    "  Cleopatra's Needle, New York City -> a Needle, central park, new york, greywacke, western hemisphere\n"
    "    https://en.wikipedia.org/wiki/Cleopatra's_Needle_(New_York_City)\n"
    "  Luxor Obelisk (Paris) -> concorde, luxor, paris; not a Needle\n"
    "    https://en.wikipedia.org/wiki/Luxor_Obelisk\n"
    "  Vatican Obelisk (Rome) -> vatican, st. peter's square, rome; not a Needle\n"
    "    https://en.wikipedia.org/wiki/Vatican_Obelisk\n"
    "STAGE 2 survivor: New York obelisk\n"
    "STAGE 3 transit:\n  112\n  days\n"
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
        "Cleopatra's Needle, London -> a Needle on the thames embankment; Cleopatra's Needle, New York "
        "City -> a Needle in central park, new york, western hemisphere; Luxor Obelisk paris concorde -> "
        "not a Needle; Vatican Obelisk rome st. peter -> not a Needle. I pick the famous London Needle; it "
        "stands about 21 m high."
    )
    r = _r(wrong)
    assert t.validate_keystone(r, _OBS)["score"] == 0.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_near_miss_numbers():
    assert t.validate_keystone(_r("the ship carried 1120 days of supplies"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("it weighs about 200 tons"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("it took 112 days to move it"), _OBS)["score"] == 1.0
    assert t.validate_keystone(_r("a 112-day journey"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    text = (
        "Cleopatra's Needle, London -> a Needle on the thames embankment; Cleopatra's Needle, New York "
        "City -> a Needle in central park, western hemisphere. I did not investigate the remaining two obelisks."
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
         "deliverables": ["Transit took 112 days", "survivor: New York obelisk"]}
    assert t.validate_keystone(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["election"]
    assert struct["waves"][2] == ["keystone_days"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_london", "cand_newyork", "cand_paris", "cand_vatican"):
        assert "{" + key + "}" in by_id["election"]["instruction"]
    assert "{election}" in by_id["keystone_days"]["instruction"]
    assert "days" in by_id["keystone_days"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("112 days", "112-day", "112 day"):
        assert leak not in blob, f"plan leaks {leak!r}"
