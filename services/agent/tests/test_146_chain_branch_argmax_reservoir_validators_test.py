"""
Offline unit tests for the chained-branch argmax task (test 146) — free, no LLM.

Covers the grounding-gated keystone (Smallwood Reservoir as the largest by SURFACE AREA, ~6,527 km²),
the UN-gated branch-coverage diagnostic (how many of the four 2-hop chains were completed, retained
even when the final comparison is wrong), the keystone-gated winner-chain and citation secondaries,
the correct answer in both single- and multi-line layout, and the adversarial failures this compound
task is built to expose:
  * the CAPACITY decoy — an agent that ranks by stored volume elects the Manicouagan Reservoir
    (137.9 km³ vs Smallwood's 32.64 km³) -> keystone 0 while breadth is retained;
  * the FAME decoy — Hoover Dam / Lake Mead, "the largest reservoir in the United States", which is
    the smallest of the four by area;
  * a right-winner/wrong-number answer (Smallwood's VOLUME instead of its area);
  * text-without-visits (the visit cap) and the 0-visit ungrounded-guess gate.
Plus the compiled plan is a genuine branch-and-chain DAG (4 -> 4), templates each branch's own hop-1
result, and leaks no reservoir name, area or volume.
"""
import re

from agent.app.idea_tests import test_146_tier5_chain_branch_argmax_reservoir as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 8}}


_FULL_SINGLE = (
    "Largest by surface area: the Smallwood Reservoir, 6,527 km². Full chain table: "
    "Churchill Falls Generating Station -> Smallwood Reservoir -> 6,527 km² "
    "(https://en.wikipedia.org/wiki/Churchill_Falls_Generating_Station, "
    "https://en.wikipedia.org/wiki/Smallwood_Reservoir); "
    "Daniel-Johnson Dam -> Manicouagan Reservoir -> 1,942 km² "
    "(https://en.wikipedia.org/wiki/Daniel-Johnson_Dam, "
    "https://en.wikipedia.org/wiki/Manicouagan_Reservoir); "
    "W. A. C. Bennett Dam -> Williston Lake -> 1,761 km² "
    "(https://en.wikipedia.org/wiki/W._A._C._Bennett_Dam, "
    "https://en.wikipedia.org/wiki/Williston_Lake); "
    "Hoover Dam -> Lake Mead -> 640 km² (247 sq mi) "
    "(https://en.wikipedia.org/wiki/Hoover_Dam, https://en.wikipedia.org/wiki/Lake_Mead)"
)

_FULL_MULTI = (
    "LARGEST BY SURFACE AREA\n"
    "  Smallwood Reservoir\n"
    "  6,527\n"
    "  km2\n"
    "ROWS (project -> reservoir -> surface area)\n"
    "  Churchill Falls Generating Station -> Smallwood Reservoir -> 6,527 km2\n"
    "    https://en.wikipedia.org/wiki/Churchill_Falls_Generating_Station\n"
    "    https://en.wikipedia.org/wiki/Smallwood_Reservoir\n"
    "  Daniel-Johnson Dam -> Manicouagan Reservoir -> 1,942 km2\n"
    "    https://en.wikipedia.org/wiki/Manicouagan_Reservoir\n"
    "  W. A. C. Bennett Dam -> Williston Lake -> 1,761 km2\n"
    "    https://en.wikipedia.org/wiki/Williston_Lake\n"
    "  Hoover Dam -> Lake Mead -> 640 km2\n"
    "    https://en.wikipedia.org/wiki/Lake_Mead\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_branch_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy layout: the superlative proximity window uses [^.;], which is
    # newline-tolerant, so the keystone must still bind across the line break.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_branch_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_station_page_area_variant_also_accepted():
    # The Churchill Falls station page quotes 6,988 km² for the same reservoir; a correctly grounded
    # answer that read the figure there must not false-fail.
    r = _r("The largest by surface area is the Smallwood Reservoir at 6,988 km².")
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0


def test_capacity_decoy_gates_to_zero_but_keeps_breadth():
    # The engineered trap: ranking by stored VOLUME elects the Manicouagan Reservoir. All four
    # branches were still chained, so the un-gated breadth diagnostic is retained at 4/4 while the
    # keystone and every gated secondary collapse to 0.
    wrong = (
        "By capacity the largest reservoir is the Manicouagan Reservoir (137.9 km³), so that is my "
        "answer. Rows: Churchill Falls -> Smallwood Reservoir -> 6,527 km²; Daniel-Johnson Dam -> "
        "Manicouagan Reservoir -> 1,942 km²; W. A. C. Bennett Dam -> Williston Lake -> 1,761 km²; "
        "Hoover Dam -> Lake Mead -> 640 km²."
    )
    r = _r(wrong)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_branch_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_chain(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_fame_decoy_lake_mead_fails_keystone():
    wrong = (
        "Lake Mead is the largest reservoir in the United States, so the answer is Lake Mead, "
        "640 km² (Hoover Dam)."
    )
    r = _r(wrong)
    assert t.validate_keystone_argmax(r, _OBS)["passed"] is False


def test_right_winner_wrong_figure_fails_keystone():
    # Correct branch elected but the reported number is Smallwood's water VOLUME (32.64 km³), not
    # its surface area — the keystone must reject it.
    r = _r("The largest is the Smallwood Reservoir, which holds 32.64 km³ of water.")
    assert t.validate_keystone_argmax(r, _OBS)["passed"] is False
    assert t.validate_winner_chain(r, _OBS)["score"] == 0.0


def test_keystone_rejects_bare_and_embedded_numbers():
    # A near-miss numeric token must not satisfy the keystone: 16,527 embeds 6,527 without a word
    # boundary; a bare 2,520 without a 'sq'/'square' unit (e.g. an elevation) must not match; and
    # naming the winner without any area figure must not match.
    assert t.validate_keystone_argmax(_r("largest: Smallwood, code 16,527"), _OBS)["score"] == 0.0
    assert t.validate_keystone_argmax(_r("largest: Smallwood, 2,520 m elevation"), _OBS)["score"] == 0.0
    assert t.validate_keystone_argmax(_r("the largest is the Smallwood Reservoir"), _OBS)["score"] == 0.0


def test_partial_branch_coverage_scores_exact_fraction():
    # Only two of the four branches chained to reservoir+area -> exactly 2/4; with no keystone the
    # gated secondaries are 0.
    text = (
        "Churchill Falls -> Smallwood Reservoir -> 6,527 km²; W. A. C. Bennett Dam -> Williston "
        "Lake -> 1,761 km². I did not finish the Daniel-Johnson or Hoover branches."
    )
    r = _r(text)
    assert abs(t.validate_branch_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_winner_chain(r, _OBS)["score"] == 0.0


def test_branch_coverage_requires_visits_not_just_text():
    # Both hops are page-only, so a 0-visit narration of all four chains must bank nothing, and a
    # partially-grounded run is capped at the number of visits actually made.
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_branch_coverage(r, {"visit": {"count": 0}})["passed"] is False
    assert abs(t.validate_branch_coverage(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_branch_coverage(r, {"visit": {"count": 4}})["score"] == 1.0


def test_visit_gate_and_scale():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 5}})["passed"] is True
    assert abs(t.validate_visits(r, {"visit": {"count": 6}})["score"] - 0.75) < 1e-9


def test_ungrounded_correct_value_gates_to_zero():
    """A perfect answer with ZERO page visits is a parametric-memory guess: the keystone and every
    gated secondary must collapse, and the overall mean must land well under the 0.75 pass bar."""
    r = _r(_FULL_SINGLE)
    obs0 = {"visit": {"count": 0}}
    assert t.validate_keystone_argmax(r, obs0)["score"] == 0.0
    assert t.validate_winner_chain(r, obs0)["score"] == 0.0
    assert t.validate_citations(r, obs0)["score"] == 0.0
    scores = [fn(r, obs0)["score"] for fn in t.get_validation_functions()]
    assert sum(scores) / len(scores) == 0.0


def test_wrong_keystone_run_is_bimodal_not_a_partial_pass():
    # The capacity-decoy run keeps full breadth but must still land far below the 0.75 bar.
    wrong = (
        "By capacity the largest reservoir is the Manicouagan Reservoir (137.9 km³). Rows: "
        "Churchill Falls -> Smallwood Reservoir -> 6,527 km²; Daniel-Johnson Dam -> Manicouagan "
        "Reservoir -> 1,942 km²; W. A. C. Bennett Dam -> Williston Lake -> 1,761 km²; Hoover Dam -> "
        "Lake Mead -> 640 km²."
    )
    r = _r(wrong)
    scores = [fn(r, _OBS)["score"] for fn in t.get_validation_functions()]
    mean = sum(scores) / len(scores)
    assert mean < 0.5
    # ...while the full answer is a clean 1.0 — bimodal, no 0.44 trap in between.
    full = [fn(_r(_FULL_SINGLE), _OBS)["score"] for fn in t.get_validation_functions()]
    assert sum(full) / len(full) == 1.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Largest by area: Smallwood Reservoir — 6,527 km²",
                          "Churchill Falls -> Smallwood Reservoir"]}
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)                      # must not raise
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 8
    assert struct["edge_count"] == 4            # one hop1 -> hop2 edge per branch, no cross edges
    assert struct["wave_widths"] == [4, 4]
    assert struct["is_pure_fanout"] is False    # NOT a flat one-round fan-out
    assert struct["is_dag_chain"] is False      # NOT a single linear chain
    assert set(struct["waves"][0]) == {"res_churchill", "res_manic", "res_williston", "res_mead"}
    assert set(struct["waves"][1]) == {"area_churchill", "area_manic", "area_williston", "area_mead"}


def test_compiled_plan_templates_its_own_branch_only():
    plan = t.get_compiled_plan()
    by_id = {leaf["id"]: leaf for leaf in plan["leaves"]}
    for key in ("churchill", "manic", "williston", "mead"):
        instr = by_id[f"area_{key}"]["instruction"]
        assert "{res_" + key + "}" in instr
        # a hop-2 leaf must not template any OTHER branch's hop-1 result (the chains are independent)
        for other in ("churchill", "manic", "williston", "mead"):
            if other != key:
                assert "{res_" + other + "}" not in instr
        assert by_id[f"area_{key}"]["depends_on"] == [f"res_{key}"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(leaf) for leaf in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the four GIVEN projects and the GIVEN comparison attribute may appear, but no
    # reservoir name (hop 1's answer), no area (hop 2's answer), no volume (the decoy axis) and no
    # hint of which branch wins.
    for leak in ("smallwood", "manicouagan", "williston", "mead", "6,527", "6527", "6,988",
                 "1,942", "1942", "1,761", "1761", "640", "247", "32.64", "137.9"):
        assert re.search(r"\b" + re.escape(leak) + r"\b", blob) is None, f"plan leaks {leak!r}"
