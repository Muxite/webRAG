"""Offline unit tests for test 126 (first cartridge-based handheld -> Microvision -> LCD resolution)."""
from agent.app.idea_tests import test_126_tier5_first_cartridge_handheld_survivor as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Stage 1: the Nintendo Game Boy is a cartridge handheld released in 1989 "
    "(https://en.wikipedia.org/wiki/Game_Boy); the Atari Lynx is a colour cartridge handheld from "
    "1989 (https://en.wikipedia.org/wiki/Atari_Lynx); the Sega Game Gear is a colour cartridge "
    "handheld from 1990 (https://en.wikipedia.org/wiki/Game_Gear); the Milton Bradley Microvision "
    "was the first handheld with interchangeable cartridges, released 1979 "
    "(https://en.wikipedia.org/wiki/Microvision) — the survivor. Stage 2: its LCD resolution is "
    "16x16 pixels."
)

_FULL_MULTI = (
    "STAGE 1 — consoles:\n"
    "  Nintendo Game Boy -> 1989\n"
    "    https://en.wikipedia.org/wiki/Game_Boy\n"
    "  Atari Lynx -> 1989 colour\n"
    "    https://en.wikipedia.org/wiki/Atari_Lynx\n"
    "  Sega Game Gear -> 1990 colour\n"
    "    https://en.wikipedia.org/wiki/Game_Gear\n"
    "  Milton Bradley Microvision -> first cartridge handheld, 1979  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Microvision\n"
    "STAGE 2 — LCD resolution:\n"
    "  16 x 16\n"
    "  pixels\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0


def test_spacing_and_unicode_times_satisfy_keystone():
    assert t.validate_keystone_resolution(_r("resolution 16 x 16"), _OBS)["score"] == 1.0
    assert t.validate_keystone_resolution(_r("screen 16×16 pixels"), _OBS)["score"] == 1.0


def test_spelled_out_joiner_satisfies_keystone():
    """Unit-tolerance fix (F26): a correctly grounded answer phrased with the spelled-out joiner
    ("16 by 16") must not false-fail merely for not using the "x"/"×" operator."""
    assert t.validate_keystone_resolution(_r("a resolution of 16 by 16 pixels"), _OBS)["score"] == 1.0


def test_ungrounded_correct_value_scores_near_zero():
    """Right keystone value present, but zero visits (no grounding) -> keystone and every
    keystone-gated secondary must collapse to 0, even though the value string matches."""
    r = _r(_FULL_SINGLE)
    ungrounded = {"visit": {"count": 0}}
    assert t.validate_keystone_resolution(r, ungrounded)["score"] == 0.0
    assert t.validate_survivor(r, ungrounded)["score"] == 0.0
    assert t.validate_citations(r, ungrounded)["score"] == 0.0
    overall = sum(v["score"] for v in [
        t.validate_keystone_resolution(r, ungrounded),
        t.validate_survivor(r, ungrounded),
        t.validate_citations(r, ungrounded),
    ]) / 3.0
    assert overall < 0.75


def test_famous_decoy_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Nintendo Game Boy -> 1989; Atari Lynx -> 1989 colour; Sega Game Gear -> 1990 colour; Milton "
        "Bradley Microvision -> 1979 first cartridge handheld. I take the famous Game Boy; its screen "
        "is 160x144 pixels."
    )
    r = _r(wrong)
    assert t.validate_keystone_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_rejects_wrong_numbers():
    assert t.validate_keystone_resolution(_r("Game Boy 160x144"), _OBS)["score"] == 0.0
    assert t.validate_keystone_resolution(_r("Lynx 160x102"), _OBS)["score"] == 0.0
    assert t.validate_keystone_resolution(_r("released in 1979"), _OBS)["score"] == 0.0


def test_partial_branch_exploration():
    text = ("Nintendo Game Boy -> 1989; Milton Bradley Microvision -> 1979 first cartridge handheld. "
            "Did not open the remaining two consoles.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_resolution(r, _OBS)["score"] == 0.0


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
    assert struct["waves"][1] == ["survivor_resolution"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_gameboy", "cand_lynx", "cand_gamegear", "cand_microvision"):
        assert "{" + key + "}" in by_id["survivor_resolution"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("16x16", "16 x 16", "16×16"):
        assert leak not in blob, f"plan leaks {leak!r}"
