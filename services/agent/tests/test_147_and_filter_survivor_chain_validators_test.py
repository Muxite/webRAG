"""
Offline unit tests for the AND-filter -> survivor -> disambiguated-chain task (test 147) — free, no LLM.

Covers the grounding-gated keystone (the outflow section's length, 165 km), the UN-gated
filter-coverage diagnostic (how many of the six lakes had BOTH constraint attributes gathered,
retained even when the chain terminus is wrong), the keystone-gated survivor/section and citation
secondaries, the answer in both single- and multi-line layout, and the adversarial failures this
compound task is built to expose:
  * DROP-A-CONSTRAINT — "largest lake" elects Balaton (600 km², 12.2 m deep) and "deepest lake"
    elects Como (425 m, 146 km²); both must gate to 0 while breadth is retained;
  * CHAIN SHORT-CIRCUIT — right survivor but the whole river's 1,230 km is reported;
  * WRONG SECTION — the INFLOW section (93.5 km) instead of the outflow section;
  * wrong unit ("165 m"), text-without-visits, and the 0-visit ungrounded guess.
Plus the compiled plan is a genuine filter-then-chain DAG (6 -> 1 -> 1), templates its upstream
results, and leaks no attribute value, no river and no section name.
"""
import re

from agent.app.idea_tests import test_147_tier5_and_filter_survivor_chain as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 7}}


_FULL_SINGLE = (
    "Keystone: the outflow section is 165 km long. Stage 1 — both constraints: Lake Constance "
    "536 km² / 251 m (area>250 yes, depth>200 yes) -> SURVIVOR "
    "(https://en.wikipedia.org/wiki/Lake_Constance); Lake Balaton 600 km² / 12.2 m (yes, no) "
    "(https://en.wikipedia.org/wiki/Lake_Balaton); Lake Neusiedl 315 km² / 1.8 m (yes, no) "
    "(https://en.wikipedia.org/wiki/Lake_Neusiedl); Lake Maggiore 212.5 km² / 372 m (no, yes) "
    "(https://en.wikipedia.org/wiki/Lake_Maggiore); Lake Como 146 km² / 425 m (no, yes) "
    "(https://en.wikipedia.org/wiki/Lake_Como); Lake Neuchâtel 218.3 km² / 152 m (no, no) "
    "(https://en.wikipedia.org/wiki/Lake_Neuchatel). Stage 2 — the outflow is the High Rhine "
    "(the Alpine Rhine is the inflow). Stage 3 — the High Rhine is 165 km long "
    "(https://en.wikipedia.org/wiki/High_Rhine)."
)

_FULL_MULTI = (
    "STAGE 1 — two-way check\n"
    "  Lake Constance: area=536 km2 (>250? yes), depth=251 m (>200? yes)  [SURVIVOR]\n"
    "    https://en.wikipedia.org/wiki/Lake_Constance\n"
    "  Lake Balaton: area=600 km2 (>250? yes), depth=12.2 m (>200? no)\n"
    "    https://en.wikipedia.org/wiki/Lake_Balaton\n"
    "  Lake Neusiedl: area=315 km2 (>250? yes), depth=1.8 m (>200? no)\n"
    "  Lake Maggiore: area=212.5 km2 (>250? no), depth=372 m (>200? yes)\n"
    "  Lake Como: area=146 km2 (>250? no), depth=425 m (>200? yes)\n"
    "  Lake Neuchâtel: area=218.3 km2 (>250? no), depth=152 m (>200? no)\n"
    "STAGE 2 — outflow section\n"
    "  High Rhine\n"
    "    https://en.wikipedia.org/wiki/High_Rhine\n"
    "STAGE 3 — its length\n"
    "  165\n"
    "  km\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_section_length(r, _OBS)["score"] == 1.0
    assert t.validate_filter_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_section(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # The keystone regex is newline-tolerant ("165\nkm"), and coverage sees the table rows.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_section_length(r, _OBS)["score"] == 1.0
    assert t.validate_filter_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_section(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["passed"] is True


def test_imperial_form_satisfies_keystone():
    r = _r("The High Rhine runs 103 mi from the lake's outflow to Basel.")
    assert t.validate_keystone_section_length(r, _OBS)["score"] == 1.0


def test_dropped_depth_constraint_gates_to_zero_but_keeps_breadth():
    # "Largest lake" shortcut -> Balaton, whose outflow length is reported instead. All six lakes
    # were still measured, so the un-gated breadth stays at 6/6 while everything else collapses.
    wrong = (
        "The largest lake is Lake Balaton (600 km², 12.2 m), so I take it as the survivor; its "
        "outflow the Sió is 121 km long. Table: Lake Constance 536 km² / 251 m; Lake Neusiedl "
        "315 km² / 1.8 m; Lake Maggiore 212.5 km² / 372 m; Lake Como 146 km² / 425 m; Lake "
        "Neuchâtel 218.3 km² / 152 m."
    )
    r = _r(wrong)
    assert t.validate_keystone_section_length(r, _OBS)["score"] == 0.0
    assert t.validate_filter_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_survivor_and_section(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_dropped_area_constraint_fails_keystone():
    wrong = "The deepest lake is Lake Como (425 m); its outflow the Adda is 313 km long."
    assert t.validate_keystone_section_length(_r(wrong), _OBS)["passed"] is False


def test_chain_short_circuit_to_whole_river_fails_keystone():
    # Right survivor, but the whole river's length is grabbed instead of the outflow section's.
    wrong = (
        "Lake Constance (536 km², 251 m) satisfies both constraints; its outflow is the Rhine, "
        "which is 1,230 km long."
    )
    r = _r(wrong)
    assert t.validate_keystone_section_length(r, _OBS)["passed"] is False
    assert t.validate_survivor_and_section(r, _OBS)["score"] == 0.0


def test_inflow_section_trap_fails_keystone():
    # The disambiguation trap: the Alpine Rhine (93.5 km) flows INTO the lake, not out of it.
    wrong = "Lake Constance's Rhine section is the Alpine Rhine, 93.5 km long."
    assert t.validate_keystone_section_length(_r(wrong), _OBS)["passed"] is False


def test_keystone_rejects_wrong_unit_and_embedded_numbers():
    # Requires a LENGTH unit: "165 m" (metres) is not the answer, and 165 embedded in a larger
    # number must not match.
    assert t.validate_keystone_section_length(_r("165 m above sea level"), _OBS)["score"] == 0.0
    assert t.validate_keystone_section_length(_r("gauge 3165 km marker"), _OBS)["score"] == 0.0
    assert t.validate_keystone_section_length(_r("the Rhine is 1,230 km"), _OBS)["score"] == 0.0


def test_keystone_rejects_grouped_numbers_ending_in_the_keystone_digits():
    """Boundary-artefact guard, mirroring 146's ``16,527`` case (added in adversarial review,
    2026-08-07). ``\\b`` treats the thousands separator as a word boundary, so ``\\b165\\s*km``
    used to accept "1,165 km" — the digits of a LARGER grouped number satisfying the keystone.
    ``(?<![\\d,.])`` closes it, on both the metric and the imperial alternative."""
    for grouped in ("1,165 km", "2,165 km", "1.165 km", "9,103 mi", "1,103 miles"):
        assert t.validate_keystone_section_length(_r(grouped), _OBS)["score"] == 0.0, grouped
    # ...while every legitimate rendering the High Rhine article produces still passes.
    for real in ("165 km", "165\nkm", "165 kilometres", "(165 km)", "= 165 km", "103 mi"):
        assert t.validate_keystone_section_length(_r(real), _OBS)["score"] == 1.0, real


def test_partial_filter_coverage_scores_exact_fraction():
    text = (
        "Lake Constance 536 km² / 251 m; Lake Como 146 km² / 425 m; Lake Balaton 600 km² / 12.2 m. "
        "I ran out of budget before Neusiedl, Maggiore and Neuchâtel."
    )
    r = _r(text)
    assert abs(t.validate_filter_coverage(r, _OBS)["score"] - 0.5) < 1e-9
    assert t.validate_keystone_section_length(r, _OBS)["score"] == 0.0


def test_filter_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_filter_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_filter_coverage(r, {"visit": {"count": 3}})["score"] - 0.5) < 1e-9
    assert t.validate_filter_coverage(r, {"visit": {"count": 6}})["score"] == 1.0


def test_visit_gate_and_scale():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True


def test_ungrounded_correct_value_gates_to_zero():
    r = _r(_FULL_SINGLE)
    obs0 = {"visit": {"count": 0}}
    assert t.validate_keystone_section_length(r, obs0)["score"] == 0.0
    assert t.validate_survivor_and_section(r, obs0)["score"] == 0.0
    assert t.validate_citations(r, obs0)["score"] == 0.0
    scores = [fn(r, obs0)["score"] for fn in t.get_validation_functions()]
    assert sum(scores) / len(scores) == 0.0


def test_scores_are_bimodal():
    wrong = (
        "The largest lake is Lake Balaton (600 km², 12.2 m); its outflow the Sió is 121 km long. "
        "Lake Constance 536 km² / 251 m; Lake Neusiedl 315 km² / 1.8 m; Lake Maggiore 212.5 km² / "
        "372 m; Lake Como 146 km² / 425 m; Lake Neuchâtel 218.3 km² / 152 m."
    )
    wrong_scores = [fn(_r(wrong), _OBS)["score"] for fn in t.get_validation_functions()]
    full_scores = [fn(_r(_FULL_SINGLE), _OBS)["score"] for fn in t.get_validation_functions()]
    assert sum(wrong_scores) / len(wrong_scores) < 0.5
    assert sum(full_scores) / len(full_scores) == 1.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["High Rhine: 165 km", "survivor: Lake Constance"]}
    assert t.validate_keystone_section_length(r, _OBS)["score"] == 1.0


def test_and_filter_fixture_has_exactly_one_survivor_and_needs_both_constraints():
    # Each constraint alone must be satisfied by MORE than one lake, or the conjunction is decorative.
    area_only = [e["name"] for e in t.LAKES if e["area_over"]]
    depth_only = [e["name"] for e in t.LAKES if e["depth_over"]]
    both = [e["name"] for e in t.LAKES if e["area_over"] and e["depth_over"]]
    assert len(area_only) == 3 and len(depth_only) == 3
    assert both == ["Lake Constance"]
    # ...and the fixture booleans must agree with the recorded numbers and thresholds.
    for e in t.LAKES:
        assert (e["area"] > t.AREA_THRESHOLD) is e["area_over"]
        assert (e["depth"] > t.DEPTH_THRESHOLD) is e["depth_over"]


def test_compiled_plan_validates_and_is_filter_then_chain():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 8
    assert struct["edge_count"] == 7          # 6 filter edges + 1 chain edge
    assert struct["wave_widths"] == [6, 1, 1]
    assert struct["waves"][1] == ["survivor_outflow"]
    assert struct["waves"][2] == ["section_length"]
    assert struct["is_pure_fanout"] is False
    assert struct["is_dag_chain"] is False


def test_compiled_plan_templates_upstream_and_is_self_describing():
    plan = t.get_compiled_plan()
    by_id = {leaf["id"]: leaf for leaf in plan["leaves"]}
    for e in t.LAKES:
        assert "{lake_" + e["key"] + "}" in by_id["survivor_outflow"]["instruction"]
    assert "{survivor_outflow}" in by_id["section_length"]["instruction"]
    assert "outflow" in by_id["survivor_outflow"]["instruction"].lower()
    assert "length" in by_id["section_length"]["expect"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(leaf) for leaf in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # The six GIVEN lakes and the two GIVEN thresholds may appear; nothing else may.
    for leak in ("rhine", "hochrhein", "165", "103", "536", "251", "600", "12.2", "315", "1.8",
                 "212.5", "372", "146", "425", "218.3", "152", "93.5", "1,230"):
        assert re.search(r"\b" + re.escape(leak) + r"\b", blob) is None, f"plan leaks {leak!r}"
    # ...and the thresholds that ARE given must still be present (the plan must state the rule).
    assert "250" in blob and "200" in blob
