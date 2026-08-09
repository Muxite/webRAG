"""Offline unit tests for test 114 (oldest metros -> Budapest Line 1 -> length/stations)."""
from agent.app.idea_tests import test_114_tier5_oldest_continental_metro_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: the London Underground is the oldest metro, opened 1863 but off the continent "
    "(https://en.wikipedia.org/wiki/London_Underground); the Athens ISAP Line 1 opened 1869 as a "
    "steam railway, electrified later (https://en.wikipedia.org/wiki/Line_1_(Athens_Metro)); the "
    "Chicago 'L' is an elevated railway from 1892 (https://en.wikipedia.org/wiki/Chicago_%22L%22); "
    "Budapest Line 1 (the Millennium Underground) opened 1896, the first electrified underground on "
    "the European mainland, a UNESCO site (https://en.wikipedia.org/wiki/Line_1_(Budapest_Metro)) — "
    "the survivor. Stage 2: it is 4.4 km long with 11 stations."
)

_FULL_MULTI = (
    "STAGE 1 — systems:\n"
    "  London Underground -> oldest metro, 1863, off the continent\n"
    "    https://en.wikipedia.org/wiki/London_Underground\n"
    "  Athens ISAP Line 1 -> 1869 steam railway, electrified later\n"
    "    https://en.wikipedia.org/wiki/Line_1_(Athens_Metro)\n"
    "  Chicago 'L' -> elevated railway, 1892\n"
    "    https://en.wikipedia.org/wiki/Chicago_%22L%22\n"
    "  Budapest Line 1 (Millennium Underground) -> 1896, first on the mainland, UNESCO  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Line_1_(Budapest_Metro)\n"
    "STAGE 2 — scale:\n"
    "  4.4 km\n"
    "  11 stations\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_scale(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_scale(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0


def test_stations_alternative_satisfies_keystone():
    assert t.validate_keystone_scale(_r("the line has 11 stations"), _OBS)["score"] == 1.0


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "London Underground -> oldest metro 1863; Athens ISAP Line 1 -> 1869 steam railway; Chicago "
        "'L' -> elevated railway 1892; Budapest Line 1 -> 1896 first on the mainland UNESCO. I take "
        "the famous London Underground; it is 402 km long."
    )
    r = _r(wrong)
    assert t.validate_keystone_scale(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_scale(_r("402 km, 272 stations"), _OBS)["score"] == 0.0
    assert t.validate_keystone_scale(_r("opened in 1896"), _OBS)["score"] == 0.0


def test_partial_branch_exploration():
    text = ("London Underground -> oldest 1863; Budapest Line 1 -> 1896 mainland UNESCO. "
            "Did not check Athens or Chicago.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_scale(r, _OBS)["score"] == 0.0


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
    assert struct["waves"][1] == ["survivor_scale"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_london", "cand_budapest", "cand_athens", "cand_chicago"):
        assert "{" + key + "}" in by_id["survivor_scale"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("4.4 km", "4.4km", "11 station", "1896"):
        assert leak not in blob, f"plan leaks {leak!r}"
