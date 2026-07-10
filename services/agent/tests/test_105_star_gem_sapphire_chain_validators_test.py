"""
Offline unit tests for the 'Star of ...' gem branch-then-chain task (test 105) — free, no LLM.

Covers the leak-resistant keystone gate (the survivor sapphire's carat weight, 563.35 ct), the
UN-gated elimination-coverage diagnostic (four gems, capped by visits), the keystone-gated
survivor/type and citation secondaries, single- and multi-line layouts, and the adversarial modes:
the fame decoy (electing the Cullinan diamond) and the sapphire distractor (Star of Bombay), plus
token rejection. The compiled plan is a genuine branch-then-chain DAG (4 -> 1 -> 1) that templates
upstream, is self-describing, and leaks nothing.
"""
from agent.app.idea_tests import test_105_tier5_star_gem_sapphire_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Stage 1: Star of Africa / Cullinan I (https://en.wikipedia.org/wiki/Cullinan_Diamond) is a "
    "diamond; Star of the South (https://en.wikipedia.org/wiki/Star_of_the_South) is a diamond; "
    "Star of Bombay (https://en.wikipedia.org/wiki/Star_of_Bombay) is a sapphire at the Smithsonian; "
    "Star of India (https://en.wikipedia.org/wiki/Star_of_India) is a sapphire at the American "
    "Museum of Natural History — the survivor. Stage 3: its weight is 563.35 carats."
)

_FULL_MULTI = (
    "STAGE 1 — mineral type / home:\n"
    "  Star of Africa (Cullinan I) -> diamond\n"
    "    https://en.wikipedia.org/wiki/Cullinan_Diamond\n"
    "  Star of the South -> diamond\n"
    "    https://en.wikipedia.org/wiki/Star_of_the_South\n"
    "  Star of Bombay -> sapphire (Smithsonian)\n"
    "    https://en.wikipedia.org/wiki/Star_of_Bombay\n"
    "  Star of India -> sapphire (American Museum of Natural History)  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Star_of_India\n"
    "STAGE 3 — carat weight:\n"
    "  563.35\n"
    "  carats\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_carat(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_type(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_carat(r, _OBS)["score"] == 1.0
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_type(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_fame_decoy_diamond_gates_to_zero_but_keeps_breadth():
    wrong = (
        "Star of Africa (Cullinan I), Star of the South, Star of Bombay, Star of India all checked. "
        "I take the famous Star of Africa (Cullinan I) diamond; it weighs 530.2 carats."
    )
    r = _r(wrong)
    assert t.validate_keystone_carat(r, _OBS)["score"] == 0.0      # 530.2 != 563.35
    assert t.validate_elimination_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_type(r, _OBS)["score"] == 0.0   # gated
    assert t.validate_citations(r, _OBS)["score"] == 0.0          # gated


def test_sapphire_distractor_bombay_gates_to_zero():
    # The sharper slip: elect the OTHER sapphire (Star of Bombay, at the Smithsonian) and report its
    # ~182 ct weight. The keystone (563.35) must reject it.
    wrong = "The sapphire is the Star of Bombay at the Smithsonian; it weighs 182 carats."
    r = _r(wrong)
    assert t.validate_keystone_carat(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_embedded_and_near_miss():
    assert t.validate_keystone_carat(_r("code 1563.35 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone_carat(_r("weight 563.350 marker"), _OBS)["score"] == 0.0
    assert t.validate_keystone_carat(_r("a bare 563 carats"), _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = "I checked only the Star of Africa and the Star of India; not the other two gems."
    r = _r(text)
    assert abs(t.validate_elimination_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_carat(r, _OBS)["score"] == 0.0


def test_elimination_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_elimination_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_elimination_coverage(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_elimination_coverage(r, {"visit": {"count": 4}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 4}})["score"] - (4 / 5)) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Star of India: 563.35 carats", "survivor: Star of India"]}
    assert t.validate_keystone_carat(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["survivor"]
    assert struct["waves"][2] == ["carat_weight"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("gem_africa", "gem_south", "gem_bombay", "gem_india"):
        assert "{" + key + "}" in by_id["survivor"]["instruction"]
    assert "{survivor}" in by_id["carat_weight"]["instruction"]
    assert "carat" in by_id["carat_weight"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("563.35", "112.67", "182 carat", "530.2"):
        assert leak not in blob, f"plan leaks {leak!r}"
