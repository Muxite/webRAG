"""
Offline unit tests for the branch-to-eliminate-then-chain task (test 098) — free, no LLM.

Covers the leak-resistant keystone gate (the Cape observatory's McClean/Victoria refractor aperture,
24 in / 610 mm), the UN-gated branch-exploration diagnostic (how many of the four Royal Observatories
were resolved to their hemisphere, retained even when the terminus is wrong), the keystone-gated
survivor/instrument and citation secondaries, both single- and multi-line layout, and the adversarial
failure modes:
  * a memory-anchored agent that elects the FAMOUS Greenwich as the survivor and reports a wrong
    downstream figure -> keystone 0 while breadth is retained;
  * a plausible-but-wrong final figure (a different aperture on the page) -> keystone 0;
  * the keystone token rejecting bare/embedded numbers.
Plus the compiled plan is a genuine branch-then-chain DAG (4 -> 1 -> 1), templates upstream results,
is self-describing, and leaks no hemisphere / survivor / instrument / aperture.
"""
import re

from agent.app.idea_tests import test_098_tier5_royal_observatory_southern as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: of the four Royal Observatories, Greenwich lies in the northern hemisphere at 51°N "
    "(https://en.wikipedia.org/wiki/Royal_Observatory,_Greenwich); Edinburgh is northern at 55°N "
    "(https://en.wikipedia.org/wiki/Royal_Observatory,_Edinburgh); the Royal Observatory of Belgium "
    "at Uccle is northern at 50°N (https://en.wikipedia.org/wiki/Royal_Observatory_of_Belgium); and "
    "the Royal Observatory, Cape of Good Hope is in the SOUTHERN hemisphere in South Africa "
    "(https://en.wikipedia.org/wiki/Royal_Observatory,_Cape_of_Good_Hope) — the survivor. Stage 2: "
    "its historic photographic refractor is the McClean telescope (the Victoria telescope). Stage 3: "
    "that refractor's photographic object glass has an aperture of 24 inches (610 mm)."
)

_FULL_MULTI = (
    "STAGE 1 — Royal Observatories and their hemispheres:\n"
    "  Greenwich -> northern (51 N)\n"
    "    https://en.wikipedia.org/wiki/Royal_Observatory,_Greenwich\n"
    "  Edinburgh -> northern (55 N)\n"
    "    https://en.wikipedia.org/wiki/Royal_Observatory,_Edinburgh\n"
    "  Royal Observatory of Belgium (Uccle) -> northern (50 N)\n"
    "    https://en.wikipedia.org/wiki/Royal_Observatory_of_Belgium\n"
    "  Cape of Good Hope -> SOUTHERN (South Africa)  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Royal_Observatory,_Cape_of_Good_Hope\n"
    "STAGE 2 — historic refractor:\n"
    "  McClean telescope (Victoria telescope)\n"
    "STAGE 3 — photographic aperture:\n"
    "  24-inch\n"
    "  610 mm\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_aperture(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_aperture(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_mm_alternative_satisfies_keystone():
    r = _r("The McClean/Victoria refractor's photographic object glass is 610 mm.")
    assert t.validate_keystone_aperture(r, _OBS)["score"] == 1.0


def test_famous_decoy_survivor_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Royal Observatories: Greenwich -> northern 51 N; Edinburgh -> northern 55 N; Royal Observatory "
        "of Belgium (Uccle) -> northern 50 N; Cape of Good Hope -> southern South Africa. I take the "
        "famous Greenwich as the survivor; its great equatorial has a 28-inch aperture."
    )
    r = _r(wrong)
    assert t.validate_keystone_aperture(r, _OBS)["score"] == 0.0     # no 24 in / 610 mm
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0     # all four hemispheres resolved
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 0.0     # gated on keystone
    assert t.validate_citations(r, _OBS)["score"] == 0.0            # gated on keystone


def test_plausible_wrong_aperture_fails():
    # Reports a different aperture that appears on the page (e.g. the 40-inch reflector) — keystone must fail.
    text = (
        "Survivor: Cape of Good Hope observatory. Its historic McClean refractor... but I report the "
        "40-inch reflector aperture instead."
    )
    r = _r(text)
    assert t.validate_keystone_aperture(r, _OBS)["passed"] is False
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 0.0    # gated on keystone


def test_keystone_token_rejects_bare_and_embedded_numbers():
    assert t.validate_keystone_aperture(_r("founded in 1820, 24 telescopes total"), _OBS)["score"] == 0.0
    assert t.validate_keystone_aperture(_r("catalogue 624-inch xj"), _OBS)["score"] == 0.0  # 624 not 24
    assert t.validate_keystone_aperture(_r("aperture 1610 mm"), _OBS)["score"] == 0.0        # 1610 not 610


def test_partial_branch_exploration_scores_fraction():
    text = (
        "Greenwich -> northern 51 N; Cape of Good Hope -> southern South Africa. I did not check "
        "Edinburgh or the Belgian observatory."
    )
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_aperture(r, _OBS)["score"] == 0.0
    assert t.validate_survivor_and_chain(r, _OBS)["score"] == 0.0


def test_branch_exploration_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["passed"] is False
    assert abs(t.validate_branch_exploration(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_branch_exploration(r, {"visit": {"count": 4}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 4}})["score"] - (4 / 5)) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["McClean refractor photographic aperture: 24-inch", "survivor: Cape of Good Hope"]}
    assert t.validate_keystone_aperture(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor_instrument"]
    assert struct["waves"][2] == ["refractor_aperture"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("obs_greenwich", "obs_edinburgh", "obs_belgium", "obs_cape"):
        assert "{" + key + "}" in by_id["survivor_instrument"]["instruction"]
    assert "{survivor_instrument}" in by_id["refractor_aperture"]["instruction"]
    assert "refractor" in by_id["survivor_instrument"]["instruction"].lower()
    assert "aperture" in by_id["refractor_aperture"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("mcclean", "victoria", "24-inch", "24 inch", "610", "southern hemisphere is",
                 "cape wins", "cape of good hope is the"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # 'south'/'north' appear only as GIVEN criterion words, never asserting the answer verdict.
    assert "24" not in blob
