"""
Offline unit tests for race-and-merge task B (test 151) — free, no LLM.

The cross-domain replication partner of test 150's suite. Covers the grounding-gated keystone (the
2007 award year bound to the laureate/medal with no other year in between), the UN-gated
route-coverage diagnostic (retained when the year is wrong), the keystone-gated agreement and
citation secondaries, single- and multi-line layout, and the decoys this fixture was chosen for:
  * 1995 — the Smithsonian/Enola Gay resignation, the strongest parametric anchor on his page;
  * 1931 (birth) and 1987 (APS fellowship), both printed in the same honours block;
  * 2006 / 2008 — the recipient rows immediately above and below his in the medal table;
  * asteroid 12143, a five-digit near-miss for a naive four-digit year regex.
Plus the linear single-route walk, URLs without visits, the 0-visit guess, and the compiled plan's
single-wave race shape / DONE-NOT-FOUND contract / zero leakage.
"""
import re

from agent.app.idea_tests import test_151_tier5_race_merge_bruce_medal_year as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}


_FULL_SINGLE = (
    "Martin Harwit received the Catherine Wolfe Bruce Gold Medal in 2007. All three independent "
    "routes agree: https://en.wikipedia.org/wiki/Martin_Harwit (honours list), "
    "https://en.wikipedia.org/wiki/Catherine_Wolfe_Bruce_Gold_Medal (recipients table), and "
    "https://phys-astro.sonoma.edu/node/71 (medalists sorted by award date)."
)

_FULL_MULTI = (
    "BRUCE MEDAL — AWARD YEAR FOR MARTIN HARWIT\n"
    "  2007\n"
    "ROUTES REACHED (all three agree)\n"
    "  route 1 laureate page -> 2007\n"
    "    https://en.wikipedia.org/wiki/Martin_Harwit\n"
    "  route 2 medal page    -> 2007\n"
    "    https://en.wikipedia.org/wiki/Catherine_Wolfe_Bruce_Gold_Medal\n"
    "  route 3 Sonoma State  -> 2007 (listed as Martin Otto Harwit)\n"
    "    https://phys-astro.sonoma.edu/node/71\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_award_year(r, _OBS)["score"] == 1.0
    assert t.validate_route_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_route_agreement(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_award_year(r, _OBS)["score"] == 1.0
    assert t.validate_route_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_route_agreement(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_smithsonian_anchor_gates_to_zero_but_keeps_breadth():
    """The 1995 Enola Gay resignation is the fact he is most written about — the parametric answer.
    All three routes were still reached, so breadth survives while every gated check collapses."""
    wrong = (
        "Martin Harwit received the Bruce Medal in 1995. Routes: "
        "https://en.wikipedia.org/wiki/Martin_Harwit, "
        "https://en.wikipedia.org/wiki/Catherine_Wolfe_Bruce_Gold_Medal, "
        "https://phys-astro.sonoma.edu/node/71 — they agree."
    )
    r = _r(wrong)
    assert t.validate_keystone_award_year(r, _OBS)["score"] == 0.0
    assert t.validate_route_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_route_agreement(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_adjacent_table_rows_fail_keystone():
    for wrong in ("Martin Harwit received the Bruce Medal in 2006.",
                  "Martin Harwit received the Bruce Medal in 2008.",
                  "Harwit — Bruce Medal, 1987.",
                  "Harwit — Bruce Medal, 1931."):
        assert t.validate_keystone_award_year(_r(wrong), _OBS)["passed"] is False, wrong


def test_year_must_be_bound_to_the_laureate_or_medal():
    # a bare year with the name far away, another year in between: not a bound answer
    far = ("Martin Harwit was born in 1931 and directed the National Air and Space Museum until "
           "1995 and many things happened in 2007 elsewhere")
    assert t.validate_keystone_award_year(_r(far), _OBS)["passed"] is False
    # no subject named at all
    assert t.validate_keystone_award_year(_r("The medal was awarded in 2007."), _OBS)["passed"] is False


def test_asteroid_number_is_not_read_as_a_year():
    """12143 Harwit is printed in the same honours block; a naive \\d{4} match sees '2143'."""
    assert t.validate_keystone_award_year(_r("Harwit has asteroid 12143 named after him."),
                                          _OBS)["passed"] is False
    assert t.validate_keystone_award_year(_r("Harwit, asteroid 12007 named after him."),
                                          _OBS)["passed"] is False


def test_correct_answer_that_also_mentions_neighbouring_rows_still_passes():
    """A thorough racer may quote the rows around his in the table; the veto must stop at the FIRST
    year after the medal's name, so quoting neighbours cannot false-fail a correct answer."""
    text = ("Martin Harwit received the Bruce Medal in 2007 "
            "(https://en.wikipedia.org/wiki/Catherine_Wolfe_Bruce_Gold_Medal). "
            "The neighbouring recipients in the table were 2006 Frank J Low and 2008 Sidney van "
            "den Bergh.")
    assert t.validate_keystone_award_year(_r(text), _OBS)["passed"] is True


def test_honours_list_layout_still_passes():
    """Layout copied from his page: an unrelated earlier year sits just before the medal line."""
    text = ("Martin Harwit — honours: Fellow of the American Physical Society 1987; "
            "Bruce Medal 2007; asteroid 12143 Harwit.")
    assert t.validate_keystone_award_year(_r(text), _OBS)["passed"] is True


def test_middle_name_rendering_from_the_specialist_site_is_credited():
    text = ("Martin Otto Harwit, Bruce Medal 2007 (https://phys-astro.sonoma.edu/node/71)")
    assert t.validate_keystone_award_year(_r(text), _OBS)["passed"] is True


def test_linear_single_route_walk_is_correct_but_narrow():
    one = ("Martin Harwit received the Bruce Medal in 2007 "
           "(https://en.wikipedia.org/wiki/Martin_Harwit)")
    r = _r(one)
    assert t.validate_keystone_award_year(r, _OBS)["score"] == 1.0
    assert abs(t.validate_route_coverage(r, _OBS)["score"] - 1 / 3) < 1e-9
    assert t.validate_route_coverage(r, _OBS)["passed"] is False
    assert t.validate_route_agreement(r, _OBS)["score"] == 0.0
    assert abs(t.validate_citations(r, _OBS)["score"] - 1 / 3) < 1e-9


def test_two_routes_without_agreement_statement_scores_half():
    two = ("Martin Harwit — Bruce Medal 2007. Sources: "
           "https://en.wikipedia.org/wiki/Martin_Harwit ; "
           "https://phys-astro.sonoma.edu/node/71")
    r = _r(two)
    assert abs(t.validate_route_coverage(r, _OBS)["score"] - 2 / 3) < 1e-9
    assert t.validate_route_agreement(r, _OBS)["score"] == 0.5


def test_bruce_medal_redirect_slug_is_accepted_as_a_citation():
    text = ("Martin Harwit — Bruce Medal 2007 (https://en.wikipedia.org/wiki/Bruce_Medal)")
    assert abs(t.validate_route_coverage(_r(text), _OBS)["score"] - 1 / 3) < 1e-9


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
        "Martin Harwit received the Bruce Medal in 1995. Routes: "
        "https://en.wikipedia.org/wiki/Martin_Harwit, "
        "https://en.wikipedia.org/wiki/Catherine_Wolfe_Bruce_Gold_Medal, "
        "https://phys-astro.sonoma.edu/node/71 — they agree."
    )
    wrong_scores = [fn(_r(wrong), _OBS)["score"] for fn in t.get_validation_functions()]
    full_scores = [fn(_r(_FULL_SINGLE), _OBS)["score"] for fn in t.get_validation_functions()]
    assert sum(wrong_scores) / len(wrong_scores) < 0.5
    assert sum(full_scores) / len(full_scores) == 1.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Harwit — Bruce Medal awarded 2007",
                          "routes: https://phys-astro.sonoma.edu/node/71"]}
    assert t.validate_keystone_award_year(r, _OBS)["score"] == 1.0


def test_metadata_and_shape_are_declared():
    md = t.get_test_metadata()
    assert md["test_id"] == "151"
    assert md["level"] == "graph"
    assert len(t.ROUTES) == 3
    assert len({r["key"] for r in t.ROUTES}) == 3
    stmt = t.get_task_statement().lower()
    assert "any one of them is sufficient" in stmt
    assert "concurrently" in stmt
    # one of the three routes must be off-Wikipedia, else the "independent publishers" claim is empty
    assert any("wikipedia" not in r["name"].lower() for r in t.ROUTES)


def test_routes_are_non_overlapping_pages():
    urls = ["https://en.wikipedia.org/wiki/martin_harwit",
            "https://en.wikipedia.org/wiki/catherine_wolfe_bruce_gold_medal",
            "https://phys-astro.sonoma.edu/node/71"]
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
    assert struct["is_pure_fanout"] is True
    assert struct["is_dag_chain"] is False


def test_compiled_plan_leaves_carry_the_race_contract():
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
    for leak in ("2007", "2006", "2008", "1995", "1987", "1931", "12143"):
        assert re.search(re.escape(leak), blob) is None, f"plan leaks {leak!r}"
    # no bare four-digit year of any kind may appear in a racing leaf
    assert re.search(r"\b(?:19|20)\d\d\b", blob) is None, "plan leaks a year"
