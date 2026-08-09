"""
Offline unit tests for the Ytterby-element branch-then-chain task (test 108) — free, no LLM.

Covers the leak-resistant keystone gate (the survivor's melting point 824 °C / boiling 1196 °C),
the UN-gated elimination-coverage diagnostic (four elements with word-bounded distinct tokens,
capped by visits), the keystone-gated survivor/property and citation secondaries, single- and
multi-line layouts, the fame decoy (electing yttrium), and — critically — the substring hazard that
"ytterbium" contains "terbium"/"erbium". The compiled plan is a genuine branch-then-chain DAG
(4 -> 1 -> 1) that templates upstream, is self-describing, and leaks nothing.
"""
from agent.app.idea_tests import test_108_tier5_ytterby_element_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: Yttrium (https://en.wikipedia.org/wiki/Yttrium) Z=39; Terbium "
    "(https://en.wikipedia.org/wiki/Terbium) Z=65; Erbium (https://en.wikipedia.org/wiki/Erbium) "
    "Z=68; Ytterbium (https://en.wikipedia.org/wiki/Ytterbium) Z=70 — the highest, the survivor. "
    "Stage 3: ytterbium's melting point is 824 °C."
)

_FULL_MULTI = (
    "STAGE 1 — atomic number:\n"
    "  Yttrium -> 39\n"
    "    https://en.wikipedia.org/wiki/Yttrium\n"
    "  Terbium -> 65\n"
    "    https://en.wikipedia.org/wiki/Terbium\n"
    "  Erbium -> 68\n"
    "    https://en.wikipedia.org/wiki/Erbium\n"
    "  Ytterbium -> 70  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Ytterbium\n"
    "STAGE 3 — melting point:\n"
    "  824\n"
    "  degrees Celsius\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_melting(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_property(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_melting(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_property(r, _OBS)["score"] == 1.0


def test_boiling_point_alternative_satisfies_keystone():
    r = _r("Ytterbium's boiling point is 1196 °C.")
    assert t.validate_keystone_melting(r, _OBS)["score"] == 1.0


def test_fame_decoy_yttrium_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Yttrium, terbium, erbium and ytterbium all checked. I take the familiar yttrium; its "
        "melting point is 1526 °C."
    )
    r = _r(wrong)
    assert t.validate_keystone_melting(r, _OBS)["score"] == 0.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_property(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_name_tokens_do_not_cross_credit_via_substring():
    # "ytterbium" contains the substrings "terbium"/"erbium" — but the word-bounded tokens must NOT
    # fire on it, so naming ONLY ytterbium credits exactly one of four (not three).
    r = _r("The answer is ytterbium.")
    assert abs(t.validate_elimination_coverage(r, _OBS)["score"] - 0.25) < 1e-9


def test_keystone_token_rejects_embedded_and_atomic_numbers():
    assert t.validate_keystone_melting(_r("code 8240 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_melting(_r("value 11196 marker"), _OBS)["score"] == 0.0
    assert t.validate_keystone_melting(_r("atomic number 70, then 68"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = "I checked only yttrium and ytterbium; not the other two elements."
    r = _r(text)
    assert abs(t.validate_elimination_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_melting(r, _OBS)["score"] == 0.0


def test_elimination_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_elimination_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_elimination_coverage(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_elimination_coverage(r, {"visit": {"count": 4}})["score"] == 1.0


def test_citations_count_not_inflated_by_ytterbium_erbium_overlap():
    # The full answer cites all four distinct pages; overlap of 'erbium' inside 'ytterbium' must not
    # push the count above 4 (>=2 passes).
    r = _r(_FULL_SINGLE)
    res = t.validate_citations(r, _OBS)
    assert res["passed"] is True


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 4}})["score"] - (4 / 5)) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    must collapse the keystone gate (and everything gated on it) to 0, not just the value match."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_melting(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_melting(r, ungrounded_obs)["passed"] is False
    assert t.validate_survivor_and_property(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citations(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_melting(r, ungrounded_obs)["score"],
        t.validate_elimination_coverage(r, ungrounded_obs)["score"],
        t.validate_survivor_and_property(r, ungrounded_obs)["score"],
        t.validate_citations(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Ytterbium melting point: 824 °C", "survivor: Ytterbium"]}
    assert t.validate_keystone_melting(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor"]
    assert struct["waves"][2] == ["melting_point"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("el_yttrium", "el_terbium", "el_erbium", "el_ytterbium"):
        assert "{" + key + "}" in by_id["survivor"]["instruction"]
    assert "{survivor}" in by_id["melting_point"]["instruction"]
    assert "melting" in by_id["melting_point"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("824", "1196", " 70", " 39", " 65", " 68"):
        assert leak not in blob, f"plan leaks {leak!r}"
