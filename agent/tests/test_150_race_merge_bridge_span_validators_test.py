"""
Offline unit tests for race-and-merge task A (test 150) — free, no LLM.

Covers the grounding-gated keystone (the 1,310 m MAIN span, bound to span vocabulary), the
UN-gated route-coverage diagnostic (how many of the three redundant routes were actually reached,
retained when the value is wrong), the keystone-gated agreement/citation secondaries, single- and
multi-line layout, and the adversarial failures specific to the race shape:
  * the TOTAL-LENGTH decoy (1,380 m) printed one line above the span in the same infobox;
  * the SWAP (span and total length reported the wrong way round);
  * the LINEAR walk — one route only: correct answer, but breadth 1/3 and nothing merged;
  * URLs recited without visits, and the 0-visit ungrounded guess.
Plus the compiled plan is a genuine single-wave race (3 leaves, 0 edges), every leaf carries the
mechanical DONE/NOT-FOUND contract, and it leaks no figure.
"""
import re

from agent.app.idea_tests import test_150_tier5_race_merge_bridge_span as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}


_FULL_SINGLE = (
    "Main span of the Hardanger Bridge: 1,310 metres. All three independent routes agree on this "
    "figure: https://en.wikipedia.org/wiki/Hardanger_Bridge (infobox 'Longest span 1,310 metres'), "
    "https://en.wikipedia.org/wiki/List_of_longest_suspension_bridge_spans (its row gives 1,310 m), "
    "and https://no.wikipedia.org/wiki/Hardangerbrua (hovedspenn 1 310 meter). "
    "Note the bridge's total length is a different figure and was not reported as the span."
)

_FULL_MULTI = (
    "HARDANGER BRIDGE — MAIN SPAN\n"
    "  1,310\n"
    "  metres\n"
    "ROUTES REACHED (all three agree)\n"
    "  route 1 subject page   -> 1,310 m\n"
    "    https://en.wikipedia.org/wiki/Hardanger_Bridge\n"
    "  route 2 ranked list    -> 1,310 m\n"
    "    https://en.wikipedia.org/wiki/List_of_longest_suspension_bridge_spans\n"
    "  route 3 Norwegian page -> 1 310 meter (hovedspenn)\n"
    "    https://no.wikipedia.org/wiki/Hardangerbrua\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_main_span(r, _OBS)["score"] == 1.0
    assert t.validate_route_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_route_agreement(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_main_span(r, _OBS)["score"] == 1.0
    assert t.validate_route_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_route_agreement(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_total_length_decoy_gates_to_zero_but_keeps_breadth():
    """The infobox trap: the 1,380 m TOTAL length reported as the span. All three routes were still
    reached, so the un-gated breadth diagnostic is retained while every gated check collapses."""
    wrong = (
        "Main span of the Hardanger Bridge: 1,380 metres. Routes: "
        "https://en.wikipedia.org/wiki/Hardanger_Bridge, "
        "https://en.wikipedia.org/wiki/List_of_longest_suspension_bridge_spans, "
        "https://no.wikipedia.org/wiki/Hardangerbrua — they agree."
    )
    r = _r(wrong)
    assert t.validate_keystone_main_span(r, _OBS)["score"] == 0.0
    assert t.validate_route_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_route_agreement(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_swapped_labels_fail_keystone():
    """Span and total length reported the wrong way round — the value 1,310 IS present, so only the
    label binding can catch this."""
    swapped = ("Hardanger Bridge — main span 1,380 m, total length 1,310 m "
               "(https://en.wikipedia.org/wiki/Hardanger_Bridge)")
    assert t.validate_keystone_main_span(_r(swapped), _OBS)["passed"] is False


def test_unlabelled_or_wrong_figures_fail_keystone():
    assert t.validate_keystone_main_span(_r("Hardanger Bridge: 1,310"), _OBS)["passed"] is False
    assert t.validate_keystone_main_span(_r("Main span: 201.5 m (Hardanger Bridge)"), _OBS)["passed"] is False
    assert t.validate_keystone_main_span(_r("Main span: 1,310 m"), _OBS)["passed"] is False  # no subject
    # a longer number that merely ends in 1310 must not be credited
    assert t.validate_keystone_main_span(_r("Hardanger Bridge span id 41310 m"), _OBS)["passed"] is False


def test_keystone_accepts_every_rendering_the_three_routes_print():
    for shown in ("main span 1,310 metres", "longest span: 1310 m", "hovedspenn 1 310 meter",
                  "1,310 m main span", "Main span\n1,310 m"):
        assert t.validate_keystone_main_span(_r("Hardanger Bridge — " + shown), _OBS)["passed"] is True, shown


def test_linear_single_route_walk_is_correct_but_narrow():
    """The behaviour the race mechanism is meant to beat: one route walked, right answer, no
    fallback and nothing merged. Keystone full, breadth 1/3, agreement 0."""
    one = ("Main span of the Hardanger Bridge: 1,310 m "
           "(https://en.wikipedia.org/wiki/Hardanger_Bridge)")
    r = _r(one)
    assert t.validate_keystone_main_span(r, _OBS)["score"] == 1.0
    assert abs(t.validate_route_coverage(r, _OBS)["score"] - 1 / 3) < 1e-9
    assert t.validate_route_coverage(r, _OBS)["passed"] is False
    assert t.validate_route_agreement(r, _OBS)["score"] == 0.0
    assert abs(t.validate_citations(r, _OBS)["score"] - 1 / 3) < 1e-9


def test_two_routes_without_agreement_statement_scores_half():
    two = ("Main span of the Hardanger Bridge: 1,310 m. Sources: "
           "https://en.wikipedia.org/wiki/Hardanger_Bridge and "
           "https://no.wikipedia.org/wiki/Hardangerbrua")
    r = _r(two)
    assert abs(t.validate_route_coverage(r, _OBS)["score"] - 2 / 3) < 1e-9
    assert t.validate_route_agreement(r, _OBS)["score"] == 0.5


def test_route_coverage_requires_visits_not_just_text():
    r = _r(_FULL_SINGLE)
    assert t.validate_route_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert abs(t.validate_route_coverage(r, {"visit": {"count": 2}})["score"] - 2 / 3) < 1e-9
    assert t.validate_route_coverage(r, {"visit": {"count": 3}})["score"] == 1.0


def test_visit_gate_and_scale():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 0}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["score"] == 1.0


def test_ungrounded_correct_value_gates_to_zero():
    r = _r(_FULL_SINGLE)
    obs0 = {"visit": {"count": 0}}
    scores = [fn(r, obs0)["score"] for fn in t.get_validation_functions()]
    assert sum(scores) / len(scores) == 0.0


def test_scores_are_bimodal():
    wrong = (
        "Main span of the Hardanger Bridge: 1,380 metres. Routes: "
        "https://en.wikipedia.org/wiki/Hardanger_Bridge, "
        "https://en.wikipedia.org/wiki/List_of_longest_suspension_bridge_spans, "
        "https://no.wikipedia.org/wiki/Hardangerbrua — they agree."
    )
    wrong_scores = [fn(_r(wrong), _OBS)["score"] for fn in t.get_validation_functions()]
    full_scores = [fn(_r(_FULL_SINGLE), _OBS)["score"] for fn in t.get_validation_functions()]
    assert sum(wrong_scores) / len(wrong_scores) < 0.5
    assert sum(full_scores) / len(full_scores) == 1.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Hardanger Bridge — main span 1,310 m",
                          "routes: https://no.wikipedia.org/wiki/Hardangerbrua"]}
    assert t.validate_keystone_main_span(r, _OBS)["score"] == 1.0


def test_metadata_and_shape_are_declared():
    md = t.get_test_metadata()
    assert md["test_id"] == "150"
    assert md["level"] == "graph"
    assert len(t.ROUTES) == 3
    assert len({r["key"] for r in t.ROUTES}) == 3
    # the task statement must state the defining property of the race shape: one route suffices
    stmt = t.get_task_statement().lower()
    assert "any one of them is sufficient" in stmt
    assert "concurrently" in stmt


def test_routes_are_non_overlapping_pages():
    """Redundancy only means anything if the three routes are genuinely different pages: no route's
    slug pattern may match another route's canonical URL."""
    urls = ["https://en.wikipedia.org/wiki/hardanger_bridge",
            "https://en.wikipedia.org/wiki/list_of_longest_suspension_bridge_spans",
            "https://no.wikipedia.org/wiki/hardangerbrua"]
    for i, route in enumerate(t.ROUTES):
        for j, url in enumerate(urls):
            hit = bool(re.search(route["slug"], url))
            assert hit == (i == j), f"route {route['key']} slug vs {url}"


def test_compiled_plan_validates_and_is_a_single_wave_race():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["edge_count"] == 0
    assert struct["wave_widths"] == [3]
    assert struct["is_pure_fanout"] is True      # redundant siblings, resolved concurrently
    assert struct["is_dag_chain"] is False


def test_compiled_plan_leaves_carry_the_race_contract():
    """Each racing leaf must (a) be confined to its own route so the siblings stay independent and
    (b) emit a mechanical DONE/NOT-FOUND signal, which is what the merge point picks a winner on."""
    for leaf in t.get_compiled_plan()["leaves"]:
        instr = leaf["instruction"].lower()
        assert "route result:" in instr, leaf["id"]
        assert "not found" in instr, leaf["id"]
        assert "do not open the other routes" in instr, leaf["id"]
        assert leaf["depends_on"] == []
    agg = t.get_compiled_plan()["aggregation"].lower()
    assert "not found" in agg and "do not" in agg


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(leaf) for leaf in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("1,310", "1310", "1 310", "4,300", "4,298", "1,380", "1380", "1 380", "4,530",
                 "201.5", "rank 21"):
        assert re.search(re.escape(leak), blob) is None, f"plan leaks {leak!r}"
