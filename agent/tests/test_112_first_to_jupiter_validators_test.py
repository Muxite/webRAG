"""Offline unit tests for test 112 (Jupiter probes -> Pioneer 10 -> antenna diameter)."""
from agent.app.idea_tests import test_112_tier5_first_to_jupiter_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: Voyager 1 flew by Jupiter in 1979 and is now interstellar "
    "(https://en.wikipedia.org/wiki/Voyager_1); Pioneer 11 flew by Jupiter in 1974 and went on to "
    "Saturn (https://en.wikipedia.org/wiki/Pioneer_11); the Galileo orbiter was first to orbit "
    "Jupiter in 1995 (https://en.wikipedia.org/wiki/Galileo_(spacecraft)); Pioneer 10 was the first "
    "spacecraft to fly by Jupiter in 1973 (https://en.wikipedia.org/wiki/Pioneer_10) — the "
    "survivor. Stage 2: its high-gain antenna dish is 2.74 m in diameter."
)

_FULL_MULTI = (
    "STAGE 1 — probes:\n"
    "  Voyager 1 -> Jupiter 1979, interstellar\n"
    "    https://en.wikipedia.org/wiki/Voyager_1\n"
    "  Pioneer 11 -> Jupiter 1974, then Saturn\n"
    "    https://en.wikipedia.org/wiki/Pioneer_11\n"
    "  Galileo orbiter -> first to orbit Jupiter 1995\n"
    "    https://en.wikipedia.org/wiki/Galileo_(spacecraft)\n"
    "  Pioneer 10 -> first to fly by Jupiter 1973  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Pioneer_10\n"
    "STAGE 2 — antenna:\n"
    "  2.74\n"
    "  m\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_antenna(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_antenna(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0


def test_launch_mass_alternative_satisfies_keystone():
    assert t.validate_keystone_antenna(_r("Pioneer 10 launch mass 258 kg"), _OBS)["score"] == 1.0


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Voyager 1 -> Jupiter 1979 interstellar; Pioneer 11 -> Jupiter 1974 then Saturn; Galileo "
        "orbiter -> orbit Jupiter 1995; Pioneer 10 -> first to fly by Jupiter 1973. I take the "
        "famous Voyager 1; its dish is 3.7 m."
    )
    r = _r(wrong)
    assert t.validate_keystone_antenna(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_antenna(_r("dish diameter 3.7 m"), _OBS)["score"] == 0.0
    assert t.validate_keystone_antenna(_r("mass 260 kg approx"), _OBS)["score"] == 0.0


def test_partial_branch_exploration():
    text = ("Voyager 1 -> Jupiter 1979 interstellar; Pioneer 10 -> first to fly by Jupiter 1973. "
            "Did not check Pioneer 11 or Galileo.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_antenna(r, _OBS)["score"] == 0.0


def test_branch_exploration_requires_visits():
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_branch_exploration(r, {"visit": {"count": 4}})["score"] == 1.0


def test_visits_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False


def test_compiled_plan_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 5
    assert struct["edge_count"] == 4
    assert struct["wave_widths"] == [4, 1]
    assert struct["waves"][1] == ["survivor_antenna"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_voyager1", "cand_pioneer10", "cand_pioneer11", "cand_galileo"):
        assert "{" + key + "}" in by_id["survivor_antenna"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("2.74", "258 kg", "258kg"):
        assert leak not in blob, f"plan leaks {leak!r}"
