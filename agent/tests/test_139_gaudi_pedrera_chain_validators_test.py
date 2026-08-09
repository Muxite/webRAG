"""
Offline unit tests for the Gaudí -> Casa Milà stop/continue chain (test 139) — no LLM.

Keystone gate (Casa Milà per-floor area, 1,323 m2), UN-gated chain-coverage (capped by visits),
gated terminal-resolution + citations, single/multi-line layouts, and the two Bucket-C failure
modes: STOP-EARLY (the Sagrada Família's own height) and OVER-HOP (a different Gaudí building on the
SAME avenue, Casa Batlló). Compiled plan is a genuine dag chain that templates its predecessor and
leaks nothing.
"""
from agent.app.idea_tests import test_139_tier5_gaudi_pedrera_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}

_FULL_SINGLE = (
    "Hop 1: the Sagrada Família (https://en.wikipedia.org/wiki/Sagrada_Familia) was designed by "
    "Antoni Gaudí (https://en.wikipedia.org/wiki/Antoni_Gaudi). Hop 2: his Casa Milà, nicknamed La "
    "Pedrera, at 92 Passeig de Gràcia (https://en.wikipedia.org/wiki/Casa_Mila). Hop 3: its floor "
    "area is 1,323 m2 per floor (on a plot of 1,620 m2)."
)

_FULL_MULTI = (
    "HOP 1 — architect:\n"
    "  Sagrada Família -> Antoni Gaudí\n"
    "    https://en.wikipedia.org/wiki/Sagrada_Familia\n"
    "    https://en.wikipedia.org/wiki/Antoni_Gaudi\n"
    "HOP 2 — terminal (La Pedrera, Passeig de Gràcia):\n"
    "  Casa Milà\n"
    "    https://en.wikipedia.org/wiki/Casa_Mila\n"
    "HOP 3 — floor area:\n"
    "  1,323\n"
    "  square metres per floor\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_area(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_area(r, _OBS)["score"] == 1.0
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_area(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_area(r, ungrounded_obs)["passed"] is False
    assert t.validate_terminal_resolution(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_chain_coverage(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_area(r, ungrounded_obs)["score"],
        t.validate_chain_coverage(r, ungrounded_obs)["score"],
        t.validate_terminal_resolution(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_stop_early_gates_to_zero_but_keeps_coverage():
    wrong = "The Sagrada Família, by Antoni Gaudí, has a tallest tower of 172.5 m."
    r = _r(wrong)
    assert t.validate_keystone_area(r, _OBS)["score"] == 0.0
    assert abs(t.validate_chain_coverage(r, _OBS)["score"] - 2 / 3) < 1e-9
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_over_hop_gates_to_zero():
    wrong = (
        "Gaudí also designed Casa Batlló, further down Passeig de Gràcia, with a colourful façade "
        "of 32 balconies."
    )
    r = _r(wrong)
    assert t.validate_keystone_area(r, _OBS)["score"] == 0.0
    assert t.validate_terminal_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_area(_r("code 13230 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_area(_r("marker 1320"), _OBS)["score"] == 0.0
    assert t.validate_keystone_area(_r("plot 1,620 m2"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = "I only investigated the Sagrada Família and Antoni Gaudí; I did not reach La Pedrera."
    r = _r(text)
    # NOTE: 'La Pedrera' in the sentence matches the terminal token, so this names all three.
    assert t.validate_chain_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_keystone_area(r, _OBS)["score"] == 0.0


def test_partial_coverage_two_of_three():
    text = "I only investigated the Sagrada Família and Antoni Gaudí; I stopped at the architect."
    r = _r(text)
    assert abs(t.validate_chain_coverage(r, _OBS)["score"] - 2 / 3) < 1e-9


def test_chain_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_chain_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_chain_coverage(r, {"visit": {"count": 2}})["score"] - 2 / 3) < 1e-9
    assert t.validate_chain_coverage(r, {"visit": {"count": 3}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Casa Milà floor area: 1,323 m2 per floor", "architect: Gaudí"]}
    assert t.validate_keystone_area(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_dag_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["edge_count"] == 2
    assert struct["wave_widths"] == [1, 1, 1]
    assert struct["is_dag_chain"] is True
    assert struct["is_pure_fanout"] is False


def test_compiled_plan_templates_upstream_and_leaks_nothing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{creator}" in by_id["other_work"]["instruction"]
    assert "{other_work}" in by_id["figure"]["instruction"]
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("1,323", "1323", "1,620", "172.5"):
        assert leak not in blob, f"plan leaks {leak!r}"
