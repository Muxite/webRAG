"""
Offline unit tests for the survivor -> reconcile -> subset-sum task (test 148) — free, no LLM.

Covers the grounding-gated keystone (the computed total 10,398 km, band [10,330, 10,410]), BOTH
un-gated breadth diagnostics (candidate mouths resolved; component lengths gathered — each retained
when the arithmetic is wrong), the keystone-gated reconciliation and citation secondaries, the
answer in single- and multi-line layout, and one adversarial case per axis this compound task
stacks:
  * STAGE-2 failure — the widely quoted combined river-SYSTEM length (5,410 km) used instead of the
    river's own 3,700 km -> total 12,108, must be rejected;
  * STAGE-3 length-rule failure — the too-short Tobol counted -> 11,989, rejected;
  * STAGE-3 basin-rule failure — the out-of-basin Vilyuy counted -> 13,048, rejected;
  * drop-one failure — the Ishim missed -> 7,948, rejected;
  * STAGE-1 failure — a wrong survivor never reaches the band at all;
plus the band's exact edges, the 3,650-variant total, text-without-visits and the 0-visit guess.
The compiled plan is checked for shape (8 -> 1), templating and leak-freedom.
"""
import re

from agent.app.idea_tests import test_148_tier5_survivor_reconcile_subset_sum as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 8}}


_FULL_SINGLE = (
    "Keystone total: 10,398 km. Stage 1 — mouths: the Ob empties into the Gulf of Ob (Kara Sea) "
    "[SURVIVOR] (https://en.wikipedia.org/wiki/Ob_(river)); the Lena into the Lena Delta and the "
    "Laptev Sea (https://en.wikipedia.org/wiki/Lena_(river)); the Amur into the Strait of Tartary "
    "(https://en.wikipedia.org/wiki/Amur); the Kolyma into the East Siberian Sea "
    "(https://en.wikipedia.org/wiki/Kolyma). Stage 2 — two lengths circulate: the river itself is "
    "3,700 km, while 5,410 km is the combined Ob-Irtysh system measured through its longest "
    "tributary, so I reject the 5,410 km system figure. Stage 3 — the Irtysh is 4,248 km and flows "
    "into the Ob (in basin, >2,000 km: counted) (https://en.wikipedia.org/wiki/Irtysh); the Ishim "
    "is 2,450 km and flows into the Irtysh (in basin, >2,000 km: counted) "
    "(https://en.wikipedia.org/wiki/Ishim_(river)); the Tobol is 1,591 km (in basin but too short: "
    "not counted) (https://en.wikipedia.org/wiki/Tobol); the Vilyuy is 2,650 km but flows into the "
    "Lena (out of basin: not counted) (https://en.wikipedia.org/wiki/Vilyuy). "
    "Addition: 3,700 + 4,248 + 2,450 = 10,398 km."
)

_FULL_MULTI = (
    "TOTAL\n"
    "  10,398\n"
    "  km\n"
    "STAGE 1 — candidate mouths\n"
    "  Ob -> Gulf of Ob (Kara Sea)  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Ob_(river)\n"
    "  Lena -> Lena Delta / Laptev Sea\n"
    "  Amur -> Strait of Tartary\n"
    "  Kolyma -> East Siberian Sea\n"
    "STAGE 2 — conflicting lengths\n"
    "  river itself: 3,700 km  (used)\n"
    "  combined system through its longest tributary: 5,410 km  (rejected)\n"
    "STAGE 3 — component checks\n"
    "  Irtysh: 4,248 km, flows into the Ob -> counted\n"
    "    https://en.wikipedia.org/wiki/Irtysh\n"
    "  Ishim: 2,450 km, flows into the Irtysh -> counted\n"
    "  Tobol: 1,591 km, flows into the Irtysh -> not counted (too short)\n"
    "  Vilyuy: 2,650 km, flows into the Lena -> not counted (other basin)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_total(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_component_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_total(r, _OBS)["score"] == 1.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_component_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation(r, _OBS)["score"] == 1.0


def test_keystone_band_edges_and_length_variant():
    # The band accepts the exact total, the 3,650-variant total and +/-10 km of rounding slack —
    # and nothing beyond it.
    assert t.validate_keystone_total(_r("Total: 10,398 km"), _OBS)["passed"] is True
    assert t.validate_keystone_total(_r("Total: 10,348 km"), _OBS)["passed"] is True   # Ob = 3,650
    assert t.validate_keystone_total(_r("Total: 10,330 km"), _OBS)["passed"] is True
    assert t.validate_keystone_total(_r("Total: 10,410 km"), _OBS)["passed"] is True
    assert t.validate_keystone_total(_r("Total: 10,329 km"), _OBS)["passed"] is False
    assert t.validate_keystone_total(_r("Total: 10,411 km"), _OBS)["passed"] is False


def test_system_length_error_gates_to_zero_but_keeps_both_breadth_axes():
    # STAGE-2 failure: the famous "seventh-longest river system" figure (5,410 km) used as the
    # river's own length -> 12,108 km. Both un-gated breadth diagnostics are retained; the keystone
    # and every gated secondary collapse.
    wrong = _FULL_SINGLE.replace("10,398 km. Stage 1", "12,108 km. Stage 1").replace(
        "Addition: 3,700 + 4,248 + 2,450 = 10,398 km.",
        "Addition: 5,410 + 4,248 + 2,450 = 12,108 km.")
    r = _r(wrong)
    assert t.validate_keystone_total(r, _OBS)["score"] == 0.0
    assert t.validate_branch_exploration(r, _OBS)["score"] == 1.0
    assert t.validate_component_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_membership_rule_failures_are_rejected():
    # STAGE-3 failures: counting the too-short Tobol, counting the out-of-basin Vilyuy, or both.
    assert t.validate_keystone_total(_r("Total: 11,989 km"), _OBS)["passed"] is False   # + Tobol
    assert t.validate_keystone_total(_r("Total: 13,048 km"), _OBS)["passed"] is False   # + Vilyuy
    assert t.validate_keystone_total(_r("Total: 14,639 km"), _OBS)["passed"] is False   # + both
    assert t.validate_keystone_total(_r("Total: 7,948 km"), _OBS)["passed"] is False    # - Ishim
    assert t.validate_keystone_total(_r("Total: 6,150 km"), _OBS)["passed"] is False    # - Irtysh


def test_stage1_failure_never_reaches_the_band():
    # A wrong survivor (the Lena, whose Vilyuy would be in-basin) cannot land in band.
    wrong = ("I take the Lena as the survivor: 4,294 km plus the Vilyuy 2,650 km and the Aldan "
             "2,273 km = 9,217 km.")
    assert t.validate_keystone_total(_r(wrong), _OBS)["passed"] is False


def test_keystone_rejects_stage_figures_alone():
    # Reporting only the survivor's length, or only the system length, is not the keystone.
    assert t.validate_keystone_total(_r("3,700 km"), _OBS)["passed"] is False
    assert t.validate_keystone_total(_r("5,410 km"), _OBS)["passed"] is False
    assert t.validate_keystone_total(_r("4,248 km"), _OBS)["passed"] is False


def test_reconciliation_requires_both_the_value_and_the_flag():
    # Gated secondary: 1/2 when the river-itself figure is present but the system figure is never
    # identified as the rejected one.
    partial = "Total: 10,398 km. The Ob is 3,700 km long. Irtysh 4,248 km, Ishim 2,450 km."
    res = t.validate_reconciliation(_r(partial), _OBS)
    assert abs(res["score"] - 0.5) < 1e-9
    flagged = partial + " The 5,410 km figure is the combined system length, which I reject."
    assert t.validate_reconciliation(_r(flagged), _OBS)["score"] == 1.0


def test_partial_coverage_scores_exact_fractions():
    text = ("Ob -> Kara Sea; Amur -> Strait of Tartary. Components: Irtysh 4,248 km; Tobol 1,591 km. "
            "No total computed.")
    r = _r(text)
    assert abs(t.validate_branch_exploration(r, _OBS)["score"] - 0.5) < 1e-9
    assert abs(t.validate_component_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_total(r, _OBS)["score"] == 0.0
    assert t.validate_reconciliation(r, _OBS)["score"] == 0.0


def test_breadth_diagnostics_require_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    obs0 = {"visit": {"count": 0}}
    assert t.validate_branch_exploration(r, obs0)["score"] == 0.0
    assert t.validate_component_coverage(r, obs0)["score"] == 0.0
    obs2 = {"visit": {"count": 2}}
    assert abs(t.validate_branch_exploration(r, obs2)["score"] - 0.5) < 1e-9
    assert abs(t.validate_component_coverage(r, obs2)["score"] - 0.5) < 1e-9


def test_visit_gate_and_scale():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 5}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 8}})["score"] == 1.0


def test_ungrounded_correct_value_gates_to_zero():
    r = _r(_FULL_SINGLE)
    obs0 = {"visit": {"count": 0}}
    scores = [fn(r, obs0)["score"] for fn in t.get_validation_functions()]
    assert sum(scores) / len(scores) == 0.0


def test_scores_are_bimodal():
    wrong = _FULL_SINGLE.replace("10,398 km. Stage 1", "12,108 km. Stage 1").replace(
        "Addition: 3,700 + 4,248 + 2,450 = 10,398 km.",
        "Addition: 5,410 + 4,248 + 2,450 = 12,108 km.")
    wrong_scores = [fn(_r(wrong), _OBS)["score"] for fn in t.get_validation_functions()]
    full_scores = [fn(_r(_FULL_SINGLE), _OBS)["score"] for fn in t.get_validation_functions()]
    assert sum(wrong_scores) / len(wrong_scores) <= 0.5
    assert sum(full_scores) / len(full_scores) == 1.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["10,398 km", "survivor: the Ob (Kara Sea)"]}
    assert t.validate_keystone_total(r, _OBS)["score"] == 1.0


def test_fixture_arithmetic_is_self_consistent():
    counted = [c for c in t.COMPONENTS if c["in_basin"] and c["length"] > t.LENGTH_THRESHOLD]
    assert [c["name"] for c in counted] == ["Irtysh", "Ishim"]
    assert t.SURVIVOR_LENGTH + sum(c["length"] for c in counted) == 10398
    assert t.KEYSTONE_TOTAL in t.KEYSTONE_BAND
    # every single-rule failure must land OUTSIDE the accepted band
    for wrong_total in (12108, 11989, 13048, 14639, 7948, 6150):
        assert wrong_total not in t.KEYSTONE_BAND


def test_compiled_plan_validates_and_is_gather_then_dependent():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 9
    assert struct["edge_count"] == 4                 # only the four mouth leaves feed the survivor leaf
    assert struct["wave_widths"] == [8, 1]
    assert struct["waves"][1] == ["survivor_lengths"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_mouths_only():
    plan = t.get_compiled_plan()
    by_id = {leaf["id"]: leaf for leaf in plan["leaves"]}
    instr = by_id["survivor_lengths"]["instruction"]
    for c in t.CANDIDATES:
        assert "{mouth_" + c["key"] + "}" in instr
    # the component leaves are independent of the elimination — they must not template anything
    for c in t.COMPONENTS:
        assert "{" not in by_id[f"comp_{c['key']}"]["instruction"]
        assert by_id[f"comp_{c['key']}"]["depends_on"] == []


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(leaf) for leaf in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # GIVENS that may appear: the four candidate rivers, the four component rivers, the Kara Sea
    # criterion and the 2,000 km threshold. Nothing else — no mouth, no length, no total.
    for leak in ("laptev", "tartary", "okhotsk", "gulf of ob", "east siberian", "3,700", "3700",
                 "5,410", "5410", "4,248", "4248", "2,450", "2450", "1,591", "1591", "2,650",
                 "2650", "10,398", "10398"):
        assert re.search(re.escape(leak), blob) is None, f"plan leaks {leak!r}"
    assert "kara sea" in blob and "2,000" in blob
