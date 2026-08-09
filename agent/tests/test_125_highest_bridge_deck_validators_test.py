"""Offline unit tests for test 125 (world's highest bridge by deck height -> Huajiang Canyon Bridge -> deck height)."""
from agent.app.idea_tests import test_125_tier5_highest_bridge_deck_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Stage 1: the Golden Gate Bridge is a famous iconic suspension bridge in San Francisco with a low "
    "deck (https://en.wikipedia.org/wiki/Golden_Gate_Bridge); the Millau Viaduct is the world's "
    "tallest bridge by structural height (pier 343 m) in France "
    "(https://en.wikipedia.org/wiki/Millau_Viaduct); the Duge Bridge (Beipanjiang) was the former "
    "highest bridge (565 m), surpassed in 2025 (https://en.wikipedia.org/wiki/Duge_Bridge); the "
    "Huajiang Canyon Bridge is the world's highest bridge with its deck above the river in Guizhou, "
    "opened 2025 (https://en.wikipedia.org/wiki/Huajiang_Canyon_Bridge) — the survivor. Stage 2: its "
    "deck height is 625 m above the river."
)

_FULL_MULTI = (
    "STAGE 1 — bridges:\n"
    "  Golden Gate Bridge -> iconic suspension bridge, low deck\n"
    "    https://en.wikipedia.org/wiki/Golden_Gate_Bridge\n"
    "  Millau Viaduct -> world's tallest bridge, structural pier 343 m\n"
    "    https://en.wikipedia.org/wiki/Millau_Viaduct\n"
    "  Duge Bridge -> former highest (565 m), surpassed in 2025\n"
    "    https://en.wikipedia.org/wiki/Duge_Bridge\n"
    "  Huajiang Canyon Bridge -> current highest bridge, deck height above the river, Guizhou 2025  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Huajiang_Canyon_Bridge\n"
    "STAGE 2 — deck height:\n"
    "  625\n"
    "  m\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_deck_height(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_deck_height(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0


def test_imperial_alternative_satisfies_keystone():
    assert t.validate_keystone_deck_height(_r("deck 2,051 ft above the river"), _OBS)["score"] == 1.0


def test_spelled_out_metric_unit_satisfies_keystone():
    """Unit-tolerance fix (F26): a correctly grounded answer phrased with the spelled-out unit
    ("625 metres"/"625 meters"/"625-metre") must not false-fail merely for not echoing the page's
    bare "m" abbreviation."""
    assert t.validate_keystone_deck_height(_r("deck height 625 metres above the river"), _OBS)["score"] == 1.0
    assert t.validate_keystone_deck_height(_r("its deck sits 625 meters up"), _OBS)["score"] == 1.0
    assert t.validate_keystone_deck_height(_r("a 625-metre deck height"), _OBS)["score"] == 1.0


def test_ungrounded_correct_value_scores_near_zero():
    """Right keystone value present, but zero visits (no grounding) -> keystone and every
    keystone-gated secondary must collapse to 0, even though the value string matches."""
    r = _r(_FULL_SINGLE)
    ungrounded = {"visit": {"count": 0}}
    assert t.validate_keystone_deck_height(r, ungrounded)["score"] == 0.0
    assert t.validate_survivor(r, ungrounded)["score"] == 0.0
    assert t.validate_citations(r, ungrounded)["score"] == 0.0
    overall = sum(v["score"] for v in [
        t.validate_keystone_deck_height(r, ungrounded),
        t.validate_survivor(r, ungrounded),
        t.validate_citations(r, ungrounded),
    ]) / 3.0
    assert overall < 0.75


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Golden Gate Bridge -> iconic suspension bridge low deck; Millau Viaduct -> world's tallest "
        "structural pier 343 m; Duge Bridge -> former highest 565 m surpassed 2025; Huajiang Canyon "
        "Bridge -> current highest deck above the river Guizhou 2025. I take the famous Millau "
        "Viaduct; its deck sits about 270 m above the river."
    )
    r = _r(wrong)
    assert t.validate_keystone_deck_height(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_deck_height(_r("Duge former highest 565 m"), _OBS)["score"] == 0.0
    assert t.validate_keystone_deck_height(_r("main span 1,420 m"), _OBS)["score"] == 0.0
    assert t.validate_keystone_deck_height(_r("Millau deck 270 m"), _OBS)["score"] == 0.0


def test_partial_branch_exploration():
    text = ("Millau Viaduct -> world's tallest structural pier; Huajiang Canyon Bridge -> current "
            "highest deck above the river Guizhou 2025. Did not check Golden Gate or Duge.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_deck_height(r, _OBS)["score"] == 0.0


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
    assert struct["waves"][1] == ["survivor_deck"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_goldengate", "cand_millau", "cand_duge", "cand_huajiang"):
        assert "{" + key + "}" in by_id["survivor_deck"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("625", "2,051", "2051"):
        assert leak not in blob, f"plan leaks {leak!r}"
