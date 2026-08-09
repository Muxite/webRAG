"""
Offline unit tests for the branch-to-eliminate-then-chain task (test 095) — free, no LLM.

Covers the leak-resistant keystone gate (the Dorset Stour's catchment, 1,240 km²), the UN-gated
branch-exploration diagnostic (how many of the four Rivers Avon were resolved to their mouth,
retained even when the terminus is wrong), the keystone-gated survivor/Stour and citation
secondaries, the correct answer in both single- and multi-line layout, and the adversarial
failure modes the task is built to expose:
  * a memory-anchored agent that elects the FAMOUS decoy (Shakespeare's Warwickshire Avon) as the
    survivor and reports its wrong downstream number -> keystone 0 while breadth is retained;
  * a plausible-but-wrong final figure (the Dorset Stour's LENGTH, 98 km, not its catchment) ->
    keystone 0;
  * the keystone token rejecting embedded/near-miss numbers.
Plus the compiled plan is a genuine branch-then-chain DAG (wave of 4 -> 1 -> 1), templates the
upstream results, is self-describing, and leaks no mouth / survivor / Stour / catchment figure.
"""
import re

from agent.app.idea_tests import test_095_tier5_branch_eliminate_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: of the four Rivers Avon, the Bristol Avon empties into the Severn Estuary at "
    "Avonmouth (https://en.wikipedia.org/wiki/River_Avon,_Bristol); the Warwickshire "
    "(Shakespeare's) Avon into the River Severn at Tewkesbury "
    "(https://en.wikipedia.org/wiki/River_Avon,_Warwickshire); the Scottish Strathspey Avon into "
    "the River Spey at Ballindalloch (https://en.wikipedia.org/wiki/River_Avon,_Strathspey); and "
    "the Hampshire (Salisbury) Avon into the ENGLISH CHANNEL at Christchurch "
    "(https://en.wikipedia.org/wiki/River_Avon,_Hampshire) — the survivor. Stage 2: at Christchurch "
    "it merges with the River Stour, Dorset (https://en.wikipedia.org/wiki/River_Stour,_Dorset). "
    "Stage 3: that Stour's catchment area is 1,240 km² (480 sq mi)."
)

_FULL_MULTI = (
    "STAGE 1 — Rivers Avon and their mouths:\n"
    "  Bristol Avon -> Severn Estuary (Avonmouth)\n"
    "    https://en.wikipedia.org/wiki/River_Avon,_Bristol\n"
    "  Warwickshire (Shakespeare's) Avon -> River Severn (Tewkesbury)\n"
    "    https://en.wikipedia.org/wiki/River_Avon,_Warwickshire\n"
    "  Strathspey (Scottish) Avon -> River Spey (Ballindalloch)\n"
    "    https://en.wikipedia.org/wiki/River_Avon,_Strathspey\n"
    "  Hampshire (Salisbury) Avon -> English Channel (Christchurch)  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/River_Avon,_Hampshire\n"
    "STAGE 2 — merges with:\n"
    "  River Stour, Dorset\n"
    "    https://en.wikipedia.org/wiki/River_Stour,_Dorset\n"
    "STAGE 3 — catchment area:\n"
    "  1,240\n"
    "  km2\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_catchment(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_stour(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy layout: the keystone (\b1,?240\b on its own line), the four-way
    # breadth, the survivor/Stour and the citations must all still register.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_catchment(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_stour(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_sq_mile_alternative_satisfies_keystone():
    # Reporting the imperial catchment "480 sq mi" is also the correct figure.
    r = _r("The River Stour, Dorset has a catchment/basin area of 480 sq mi.")
    assert t.validate_keystone_catchment(r, _OBS)["score"] == 1.0


def test_famous_decoy_survivor_gates_to_zero_but_keeps_breadth():
    # The classic shallow slip: a memory-anchored agent still resolves all four mouths (branch
    # exploration full) but WRONGLY elects Shakespeare's Warwickshire Avon as the "survivor",
    # follows it to the Severn (no Stour), and reports a wrong downstream figure. The keystone must
    # be 0; the un-gated breadth is retained; the gated secondaries zero out.
    wrong = (
        "Rivers Avon: Bristol Avon -> Severn Estuary (Avonmouth); Warwickshire (Shakespeare's) Avon "
        "-> River Severn at Tewkesbury; Hampshire/Salisbury Avon -> English Channel at Christchurch; "
        "Strathspey Scottish Avon -> River Spey at Ballindalloch. I take the famous Warwickshire "
        "Avon as the survivor; its length is 137 km."
    )
    r = _r(wrong)
    assert t.validate_keystone_catchment(r, _OBS)["score"] == 0.0     # no 1,240 / 480 sq
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0      # all four mouths resolved
    assert t.validate_survivor_and_stour(r, _OBS)["score"] == 0.0      # gated on keystone
    assert t.validate_citations(r, _OBS)["score"] == 0.0             # gated on keystone


def test_plausible_wrong_final_figure_fails():
    # Correct chain resolution but reports the Dorset Stour's LENGTH (98 km / 61 mi) instead of its
    # CATCHMENT (1,240 km²) — a plausible confusion between two infobox figures. Keystone must fail.
    text = (
        "Survivor: Hampshire (Salisbury) Avon -> English Channel at Christchurch, where it merges "
        "with the River Stour, Dorset. The Dorset Stour is 98 km (61 mi) long."
    )
    r = _r(text)
    assert t.validate_keystone_catchment(r, _OBS)["passed"] is False
    assert t.validate_survivor_and_stour(r, _OBS)["score"] == 0.0     # gated on keystone


def test_keystone_token_rejects_embedded_and_bare_numbers():
    # A larger number merely containing "1240" ("21,240") must not match; a bare "480" without a
    # 'sq'/'square' unit (e.g. an elevation) must not match; the length figures must not match.
    assert t.validate_keystone_catchment(_r("basin code 21,240 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_catchment(_r("source elevation 480 m"), _OBS)["score"] == 0.0
    assert t.validate_keystone_catchment(_r("length 98 km (61 mi)"), _OBS)["score"] == 0.0


def test_partial_branch_exploration_scores_fraction():
    # Only two of the four Avons resolved to their mouth -> breadth registers exactly 2/4, and with
    # no keystone the gated secondaries are 0.
    text = (
        "Bristol Avon -> Severn Estuary (Avonmouth); Hampshire/Salisbury Avon -> English Channel at "
        "Christchurch. I did not check the Warwickshire or Scottish Avons."
    )
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_catchment(r, _OBS)["score"] == 0.0
    assert t.validate_survivor_and_stour(r, _OBS)["score"] == 0.0


def test_branch_exploration_requires_visits_not_just_text():
    # Regression: a model can narrate all four Avon mouths from the mandate/memory with ZERO page
    # visits. The mouth is a PAGE-ONLY infobox fact, so text-presence alone must not bank breadth —
    # the credited count is capped by the visit count, so zero visits scores 0 despite a full answer.
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_branch_exploration(r, {"visit": {"count": 0}})["passed"] is False
    # A partial number of visits caps the credited breadth at the visits made (min of 4 hits, 2 visits).
    assert abs(t.validate_branch_exploration(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    # Enough visits to back all four -> full credit, as before.
    assert t.validate_branch_exploration(r, {"visit": {"count": 4}})["score"] == 1.0


def test_no_cross_crediting_between_severn_mouthed_avons():
    # Both the Bristol and Warwickshire Avons share a 'Severn' mouth token, but each is credited
    # only with its OWN distinct name token: naming just Warwickshire's mouth must not credit Bristol.
    text = "The Warwickshire (Shakespeare's) Avon flows into the River Severn at Tewkesbury."
    r = _r(text)
    res = t.validate_branch_exploration(r, _OBS)
    assert abs(res["score"] - 0.25) < 1e-9   # exactly one of four


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 4}})["score"] - (4 / 5)) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_catchment(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_catchment(r, ungrounded_obs)["passed"] is False
    assert t.validate_survivor_and_stour(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_catchment(r, ungrounded_obs)["score"],
        t.validate_branch_exploration(r, ungrounded_obs)["score"],
        t.validate_survivor_and_stour(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_deliverables_slot_is_primary_for_keystone():
    # Keystone reads deliverables[0] when present, even if final_deliverable lacks the figure.
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Dorset Stour catchment: 1,240 km²", "survivor: Hampshire Avon"]}
    assert t.validate_keystone_catchment(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    # 4 fan-out edges (each Avon -> survivor) + 1 chain edge (survivor -> catchment)
    assert struct["edge_count"] == 5
    # Genuine tree: a wide first wave (the 4-way branch), then two single-leaf waves (the chain).
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor_confluence"]
    assert struct["waves"][2] == ["stour_catchment"]
    # NOT a pure fan-out and NOT a pure chain — the shape this task is designed to force.
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    # Wave-2 leaf templates all four upstream mouths; wave-3 leaf templates the survivor result.
    for key in ("avon_bristol", "avon_warwickshire", "avon_hampshire", "avon_scottish"):
        assert "{" + key + "}" in by_id["survivor_confluence"]["instruction"]
    assert "{survivor_confluence}" in by_id["stour_catchment"]["instruction"]
    # Self-describing: each dependent leaf restates its own subject so the aggregator can bind it.
    assert "stour" in by_id["survivor_confluence"]["instruction"].lower()
    assert "catchment" in by_id["stour_catchment"]["expect"].lower()
    assert "km" in by_id["stour_catchment"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: names the GIVEN candidate Avons and the GIVEN English-Channel criterion, but
    # leaks NO mouth (the fact to find), NOT which Avon survives, NOT the Dorset Stour, and NOT the
    # catchment figure. Word-bounded so "spey" (the Scottish Avon's mouth) is caught while the GIVEN
    # candidate name "Strathspey" is allowed.
    for leak in ("severn", "spey", "estuary", "tewkesbury", "avonmouth", "ballindalloch",
                 "christchurch", "dorset", "1,240", "1240", "480", "98 km", "61 mi"):
        assert re.search(r"\b" + re.escape(leak) + r"\b", blob) is None, f"plan leaks {leak!r}"
