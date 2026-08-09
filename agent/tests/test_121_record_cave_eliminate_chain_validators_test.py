"""Offline unit tests for test 121 (record caves -> Krubera mapped length). Free, no LLM.

Covers the leak-resistant keystone gate (mapped length 16.058 km, DISTINCT from the headline depth),
the UN-gated candidate-coverage diagnostic, the keystone-gated survivor/citation secondaries, single-
and multi-line layout, the famous-decoy failure mode (picking the longest, Mammoth), the
echo-the-headline-depth failure (reporting Krubera's depth 2,224 m instead of its length), keystone
token rejecting near-miss numbers, visit gating, and the compiled plan (branch-then-chain, templated,
self-describing, leaks nothing).
"""
from agent.app.idea_tests import test_121_tier5_record_cave_eliminate_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_FULL_SINGLE = (
    "Stage 1: Mammoth Cave in Kentucky is the longest cave system, 686 km "
    "(https://en.wikipedia.org/wiki/Mammoth_Cave); the Sarawak Chamber (Gua Nasib Bagus) in Mulu, "
    "Malaysia is the largest cave chamber by area on Borneo "
    "(https://en.wikipedia.org/wiki/Sarawak_Chamber); Optymistychna Cave in Ukraine is the longest "
    "gypsum cave in Europe, 264 km (https://en.wikipedia.org/wiki/Optymistychna_Cave); Krubera (Voronya) "
    "Cave in the Arabika Massif, Abkhazia is the deepest known cave, 2,224 m "
    "(https://en.wikipedia.org/wiki/Krubera_Cave). Stage 2: survivor is Krubera. Stage 3: its total "
    "mapped length is 16.058 km."
)

_FULL_MULTI = (
    "STAGE 1:\n"
    "  Mammoth Cave -> longest cave system, kentucky, 686\n"
    "    https://en.wikipedia.org/wiki/Mammoth_Cave\n"
    "  Sarawak Chamber (Gua Nasib Bagus) -> largest chamber, borneo, mulu, malaysia\n"
    "    https://en.wikipedia.org/wiki/Sarawak_Chamber\n"
    "  Optymistychna Cave -> longest gypsum cave, ukraine, 264, europe\n"
    "    https://en.wikipedia.org/wiki/Optymistychna_Cave\n"
    "  Krubera (Voronya) Cave -> deepest, arabika, abkhazia, 2,224\n"
    "    https://en.wikipedia.org/wiki/Krubera_Cave\n"
    "STAGE 2 survivor: Krubera\n"
    "STAGE 3 mapped length:\n  16.058\n  km\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone(r, _OBS)["score"] == 1.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone(r, _OBS)["score"] == 1.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_famous_decoy_survivor_gates_to_zero_but_keeps_coverage():
    wrong = (
        "Mammoth Cave -> longest cave system, kentucky, 686; Sarawak Chamber (Gua Nasib Bagus) -> largest "
        "chamber, borneo, mulu, malaysia; Optymistychna Cave -> longest gypsum cave, ukraine, 264, europe; "
        "Krubera (Voronya) Cave -> deepest, arabika, abkhazia, 2,224. I pick the famous longest cave, "
        "Mammoth; it is over 686 km long."
    )
    r = _r(wrong)
    assert t.validate_keystone(r, _OBS)["score"] == 0.0
    assert t.validate_candidate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_echo_the_headline_depth_fails_keystone():
    # Correct survivor (Krubera) but reports its headline DEPTH (2,224 m), not its mapped LENGTH.
    text = (
        "Survivor: Krubera (Voronya) Cave, the deepest known cave in the Arabika Massif; its depth is "
        "2,224 m (2,199 m confirmed)."
    )
    r = _r(text)
    assert t.validate_keystone(r, _OBS)["passed"] is False
    assert t.validate_survivor(r, _OBS)["score"] == 0.0


def test_keystone_token_rejects_near_miss_numbers():
    assert t.validate_keystone(_r("code 216.058 xj"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("its depth is 2,224 m"), _OBS)["score"] == 0.0
    assert t.validate_keystone(_r("mapped length 16.058 km"), _OBS)["score"] == 1.0
    assert t.validate_keystone(_r("mapped length 16,058 m"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    text = (
        "Mammoth Cave -> longest cave, kentucky; Krubera (Voronya) Cave -> deepest, arabika, abkhazia, "
        "2,224. I did not check Sarawak or Optymistychna."
    )
    r = _r(text)
    assert abs(t.validate_candidate_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone(r, _OBS)["score"] == 0.0
    assert t.validate_survivor(r, _OBS)["score"] == 0.0


def test_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_candidate_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_candidate_coverage(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_candidate_coverage(r, {"visit": {"count": 4}})["score"] == 1.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Mapped length: 16.058 km", "survivor: Krubera"]}
    assert t.validate_keystone(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 6
    assert struct["edge_count"] == 5
    assert struct["wave_widths"] == [4, 1, 1]
    assert struct["waves"][1] == ["election"]
    assert struct["waves"][2] == ["keystone_length"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {l["id"]: l for l in plan["leaves"]}
    for key in ("cand_mammoth", "cand_sarawak", "cand_optymistychna", "cand_krubera"):
        assert "{" + key + "}" in by_id["election"]["instruction"]
    assert "{election}" in by_id["keystone_length"]["instruction"]
    assert "length" in by_id["keystone_length"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("16.058", "16,058"):
        assert leak not in blob, f"plan leaks {leak!r}"
