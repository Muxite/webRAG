"""
Offline unit tests for the chained-branch argmax task B (test 149) — free, no LLM.

The replication partner of test 146 in a different domain. Covers the grounding-gated keystone (the
VLT at Paranal, 8.2 m primary mirror, asserted as the biggest), the UN-gated branch-coverage
diagnostic (how many of the four 2-hop chains completed, retained when the comparison is wrong), the
keystone-gated winner-chain and citation secondaries, single- and multi-line layout, and the
adversarial failures:
  * the two FAME decoys — the Hale Telescope (world's largest 1949-1993) and the Hooker telescope
    (world's largest 1917-1949), both of which lose here;
  * the MULTI-UNIT trap — reporting a combined/equivalent aperture instead of one unit's 8.2 m;
  * a right-winner/no-figure answer, text-without-visits, and the 0-visit ungrounded guess.
Plus the compiled plan is a genuine branch-and-chain DAG (4 -> 4) whose hop-2 leaves template only
their OWN branch, and it leaks no telescope name and no diameter.
"""
import re

from agent.app.idea_tests import test_149_tier5_chain_branch_argmax_telescope as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 8}}


_FULL_SINGLE = (
    "Biggest primary mirror: the Very Large Telescope at Paranal, 8.2 m per unit telescope. Rows: "
    "Paranal Observatory -> Very Large Telescope -> 8.2 m "
    "(https://en.wikipedia.org/wiki/Paranal_Observatory, "
    "https://en.wikipedia.org/wiki/Very_Large_Telescope); "
    "Palomar Observatory -> Hale Telescope -> 5.08 m (200 inch) "
    "(https://en.wikipedia.org/wiki/Palomar_Observatory, "
    "https://en.wikipedia.org/wiki/Hale_Telescope); "
    "Kitt Peak National Observatory -> Nicholas U. Mayall Telescope -> 4.0 m (158 inch) "
    "(https://en.wikipedia.org/wiki/Kitt_Peak_National_Observatory); "
    "Mount Wilson Observatory -> Hooker telescope -> 2.54 m (100 inch) "
    "(https://en.wikipedia.org/wiki/Mount_Wilson_Observatory)"
)

_FULL_MULTI = (
    "BIGGEST PRIMARY MIRROR\n"
    "  Very Large Telescope (Paranal Observatory)\n"
    "  8.2\n"
    "  m per unit telescope\n"
    "ROWS (observatory -> largest telescope -> primary mirror)\n"
    "  Paranal Observatory -> Very Large Telescope -> 8.2 m\n"
    "    https://en.wikipedia.org/wiki/Very_Large_Telescope\n"
    "  Palomar Observatory -> Hale Telescope -> 5.08 m\n"
    "    https://en.wikipedia.org/wiki/Hale_Telescope\n"
    "  Kitt Peak National Observatory -> Nicholas U. Mayall Telescope -> 4.0 m\n"
    "    https://en.wikipedia.org/wiki/Kitt_Peak_National_Observatory\n"
    "  Mount Wilson Observatory -> Hooker telescope -> 2.54 m\n"
    "    https://en.wikipedia.org/wiki/Mount_Wilson_Observatory\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_branch_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_branch_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_chain(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_fame_decoy_hale_gates_to_zero_but_keeps_breadth():
    # The classic memory anchor: "the Hale Telescope was the largest in the world". All four chains
    # were still walked, so breadth is retained while the keystone and gated checks collapse.
    wrong = (
        "The largest is the Hale Telescope at Palomar, 5.08 m (200 inch). Rows: Paranal Observatory "
        "-> Very Large Telescope -> 8.2 m; Kitt Peak National Observatory -> Nicholas U. Mayall "
        "Telescope -> 4.0 m; Mount Wilson Observatory -> Hooker telescope -> 2.54 m."
    )
    r = _r(wrong)
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0
    assert t.validate_branch_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_chain(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_fame_decoy_hooker_fails_keystone():
    wrong = "Mount Wilson's Hooker telescope (2.54 m, 100 inch) is the answer."
    assert t.validate_keystone_argmax(_r(wrong), _OBS)["passed"] is False


def test_multi_unit_trap_fails_keystone():
    # Reporting a combined/equivalent aperture instead of ONE unit's primary mirror.
    wrong = "The largest is the VLT at Paranal, with a combined equivalent aperture of 16.4 m."
    assert t.validate_keystone_argmax(_r(wrong), _OBS)["passed"] is False


def test_right_winner_without_figure_fails_keystone():
    assert t.validate_keystone_argmax(_r("The largest telescope is the VLT at Paranal."), _OBS)["passed"] is False


def test_keystone_rejects_embedded_numbers():
    assert t.validate_keystone_argmax(_r("largest: VLT, serial 18.25"), _OBS)["score"] == 0.0
    assert t.validate_keystone_argmax(_r("largest: VLT, 5.08 m"), _OBS)["score"] == 0.0


def test_partial_branch_coverage_scores_exact_fraction():
    text = ("Paranal -> Very Large Telescope -> 8.2 m; Mount Wilson -> Hooker telescope -> 2.54 m. "
            "The Palomar and Kitt Peak branches were not completed.")
    r = _r(text)
    assert abs(t.validate_branch_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 0.0


def test_branch_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_branch_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_branch_coverage(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_branch_coverage(r, {"visit": {"count": 4}})["score"] == 1.0


def test_visit_gate_and_scale():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 5}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 7}})["score"] == 1.0


def test_ungrounded_correct_value_gates_to_zero():
    r = _r(_FULL_SINGLE)
    obs0 = {"visit": {"count": 0}}
    scores = [fn(r, obs0)["score"] for fn in t.get_validation_functions()]
    assert sum(scores) / len(scores) == 0.0


def test_scores_are_bimodal():
    wrong = (
        "The largest is the Hale Telescope at Palomar, 5.08 m. Rows: Paranal Observatory -> Very "
        "Large Telescope -> 8.2 m; Kitt Peak National Observatory -> Nicholas U. Mayall Telescope "
        "-> 4.0 m; Mount Wilson Observatory -> Hooker telescope -> 2.54 m."
    )
    wrong_scores = [fn(_r(wrong), _OBS)["score"] for fn in t.get_validation_functions()]
    full_scores = [fn(_r(_FULL_SINGLE), _OBS)["score"] for fn in t.get_validation_functions()]
    assert sum(wrong_scores) / len(wrong_scores) < 0.5
    assert sum(full_scores) / len(full_scores) == 1.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Biggest: Very Large Telescope — 8.2 m", "site: Paranal Observatory"]}
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0


def test_fixture_margins_are_wide_and_distinct():
    dias = sorted(b["diameter_m"] for b in t.BRANCHES)
    assert dias == [2.54, 4.0, 5.08, 8.2]
    assert t.WINNER["diameter_m"] == max(dias)
    runner_up = sorted(dias)[-2]
    assert t.WINNER["diameter_m"] / runner_up > 1.6      # >60% margin over the runner-up
    for a, b in zip(dias, dias[1:]):                     # neighbours are >25% apart
        assert b / a > 1.25


def test_hop1_asks_for_the_largest_OPTICAL_telescope_uniformly():
    """Regression guard for the hop-1 ambiguity found in adversarial review (2026-08-07).

    Kitt Peak's article never calls the Mayall "the largest telescope"; it says the largest OPTICAL
    instruments are the Mayall and the WIYN, and separately that "the ARO 12m Radio Telescope is
    also at the location". An UNQUALIFIED "largest telescope at that site" therefore has a
    defensible answer of 12 m, which beats the winner's 8.2 m and silently flips the argmax — a
    correct-process run scoring 0. The qualifier must be present in the statement AND in every
    hop-1 leaf, and must be UNIFORM across the four branches so it cannot hint which site needs it
    (all four winners are optical, so it narrows nothing)."""
    stmt = t.get_task_statement().lower()
    assert "largest optical telescope" in stmt
    assert "radio telescope" in stmt, "the statement must say which instrument class to ignore"
    hop1 = [lf for lf in t.get_compiled_plan()["leaves"] if lf["id"].startswith("tel_")]
    assert len(hop1) == len(t.BRANCHES)
    for leaf in hop1:
        assert "largest optical telescope" in leaf["instruction"].lower(), leaf["id"]
        assert "radio telescope" in leaf["instruction"].lower(), leaf["id"]
    # UNIFORMITY: strip the branch's own givens and every hop-1 instruction must be byte-identical,
    # so the qualifier cannot be read as a hint about one particular site.
    skeletons = set()
    for leaf, b in zip(hop1, t.BRANCHES):
        # ``desc`` embeds ``site``, so it must be substituted FIRST or the longer span is mangled.
        skeletons.add(leaf["instruction"].replace(b["desc"], "<DESC>").replace(b["site"], "<SITE>"))
    assert len(skeletons) == 1, f"hop-1 instructions differ beyond their GIVEN site/description: {skeletons}"


def test_page_printed_metric_values_are_credited():
    """The pages print "100-inch (2.5 m)" and "200-inch (5.1 m)", NOT the exact conversions 2.54 /
    5.08 (verified live 2026-08-07 — neither string occurs in the wikitext of any of the four
    pages; Mount Wilson's prose even says "the 2.5 meter telescope"). An agent that answers the
    question as asked — the diameter IN METRES — must be credited for the value the page shows."""
    hooker = next(b for b in t.BRANCHES if b["key"] == "wilson")["dia_rx"]
    kitt = next(b for b in t.BRANCHES if b["key"] == "kittpeak")["dia_rx"]
    for shown in ("Hooker telescope: 2.5 m", "the 2.5-meter telescope", "2.54 m", "100-inch"):
        assert re.search(hooker, shown, re.I), f"page-printed value not credited: {shown!r}"
    for near_miss in ("1.5 m", "12.5 m", "3.5 m"):
        assert re.search(hooker, near_miss, re.I) is None, f"over-matches {near_miss!r}"
    # ...and the Kitt Peak metric alternative must not fire on any other "X.4 m" figure: a bare
    # ``\b4\s*m`` treats the '.' in "2.4 m" as a word boundary and credited the MDM 2.4 m Hiltner
    # (printed on the very same Kitt Peak page) as if it were the Mayall's mirror.
    for good in ("Mayall 4-meter", "4.0 m", "the 4 m Mayall", "158 inches", "four-meter"):
        assert re.search(kitt, good, re.I), good
    for bad in ("the MDM 2.4 m Hiltner Telescope", "a 3.4 m mirror", "WIYN 3.5-meter", "24 m"):
        assert re.search(kitt, bad, re.I) is None, f"Kitt Peak dia_rx over-matches {bad!r}"


def test_full_answer_with_page_printed_metric_values_still_scores_all():
    """End-to-end companion: the same answer written with the metric values the pages actually
    print (2.5 m / 5.1 m) scores exactly as the exact-conversion fixture does."""
    r = _r(_FULL_SINGLE.replace("2.54 m", "2.5 m").replace("5.08 m", "5.1 m"))
    assert t.validate_keystone_argmax(r, _OBS)["score"] == 1.0
    assert t.validate_branch_coverage(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_branch_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 8
    assert struct["edge_count"] == 4
    assert struct["wave_widths"] == [4, 4]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_its_own_branch_only():
    plan = t.get_compiled_plan()
    by_id = {leaf["id"]: leaf for leaf in plan["leaves"]}
    for key in ("paranal", "palomar", "kittpeak", "wilson"):
        instr = by_id[f"dia_{key}"]["instruction"]
        assert "{tel_" + key + "}" in instr
        for other in ("paranal", "palomar", "kittpeak", "wilson"):
            if other != key:
                assert "{tel_" + other + "}" not in instr
        assert by_id[f"dia_{key}"]["depends_on"] == [f"tel_{key}"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(leaf) for leaf in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("hale", "mayall", "hooker", "vlt", "very large telescope", "8.2", "5.08", "5.1",
                 "2.54", "158", "200-inch", "100-inch"):
        assert re.search(re.escape(leak), blob) is None, f"plan leaks {leak!r}"
