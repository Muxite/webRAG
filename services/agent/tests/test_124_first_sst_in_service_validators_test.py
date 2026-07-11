"""Offline unit tests for test 124 (first SST airliner in service -> Tu-144 -> cruising speed)."""
from agent.app.idea_tests import test_124_tier5_first_sst_in_service_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Stage 1: Concorde entered service 21 January 1976, the Anglo-French airliner "
    "(https://en.wikipedia.org/wiki/Concorde); the Boeing 2707 was cancelled and never flew "
    "(https://en.wikipedia.org/wiki/Boeing_2707); the Lockheed L-2000 was a cancelled proposal, "
    "never built (https://en.wikipedia.org/wiki/Lockheed_L-2000); the Tupolev Tu-144 entered service "
    "26 December 1975 flying mail and freight (https://en.wikipedia.org/wiki/Tupolev_Tu-144) — the "
    "survivor, first SST in service. Stage 2: its cruising speed is 2,125 km/h."
)

_FULL_MULTI = (
    "STAGE 1 — aircraft:\n"
    "  Concorde -> entered service 21 January 1976\n"
    "    https://en.wikipedia.org/wiki/Concorde\n"
    "  Boeing 2707 -> cancelled, never flew\n"
    "    https://en.wikipedia.org/wiki/Boeing_2707\n"
    "  Lockheed L-2000 -> cancelled proposal, never built\n"
    "    https://en.wikipedia.org/wiki/Lockheed_L-2000\n"
    "  Tupolev Tu-144 -> first SST in service, 26 December 1975 mail/freight  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Tupolev_Tu-144\n"
    "STAGE 2 — cruising speed:\n"
    "  2,125\n"
    "  km/h\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_speed(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_speed(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0


def test_speed_alt_forms_satisfy_keystone():
    assert t.validate_keystone_speed(_r("cruise 1,320 mph"), _OBS)["score"] == 1.0
    assert t.validate_keystone_speed(_r("2125 km/h cruise"), _OBS)["score"] == 1.0


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Concorde -> entered service 1976 Anglo-French; Boeing 2707 -> cancelled never flew; "
        "Lockheed L-2000 -> cancelled proposal; Tupolev Tu-144 -> in service 1975 mail freight. I "
        "take the famous Concorde; it cruised at about 2,158 km/h."
    )
    r = _r(wrong)
    assert t.validate_keystone_speed(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_speed(_r("Concorde 2,158 km/h"), _OBS)["score"] == 0.0
    assert t.validate_keystone_speed(_r("max speed 2,500 km/h"), _OBS)["score"] == 0.0
    assert t.validate_keystone_speed(_r("12,125 units"), _OBS)["score"] == 0.0


def test_partial_branch_exploration():
    text = ("Concorde -> entered service 1976; Tupolev Tu-144 -> in service 1975 mail freight. Did "
            "not check the Boeing 2707 or Lockheed L-2000.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_speed(r, _OBS)["score"] == 0.0


def test_branch_exploration_requires_visits():
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_branch_exploration(r, {"visit": {"count": 4}})["score"] == 1.0


def test_visits_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is False


def test_compiled_plan_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 5
    assert struct["edge_count"] == 4
    assert struct["wave_widths"] == [4, 1]
    assert struct["waves"][1] == ["survivor_speed"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_concorde", "cand_boeing2707", "cand_lockheed2000", "cand_tu144"):
        assert "{" + key + "}" in by_id["survivor_speed"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("2,125", "2125", "1,320 mph", "1,147 kn"):
        assert leak not in blob, f"plan leaks {leak!r}"
