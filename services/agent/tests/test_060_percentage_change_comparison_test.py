"""
Offline unit tests for the percentage-change comparison task (test 060) — free.

Cover the keystone gate (Frisco = larger PERCENTAGE change, the absolute-vs-relative trap), in
single- and multi-line layout; the wrong answer that names Phoenix (the larger ABSOLUTE change)
gating every credit-bearing check to zero; the "names both, no winner" case failing the keystone;
the UN-gated coverage diagnostic (four census figures, retained even when the comparison is
botched and reported as an exact fraction); the gated percentage-value check rejecting a value
outside the +/- 2-point tolerance; the visit gate (no-visits -> 0); and that the compiled plan is
a well-formed four-leaf pure-fan-out DAG that leaks no value, percentage or winner.
"""
from agent.app.idea_tests import test_060_tier5_percentage_change_comparison as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}


_FULL_SINGLE = (
    "Frisco, Texas grew from 116,989 (2010 census) to 200,509 (2020 census) "
    "(https://en.wikipedia.org/wiki/Frisco,_Texas), a change of +71.4%. "
    "Phoenix, Arizona grew from 1,445,632 (2010 census) to 1,608,139 (2020 census) "
    "(https://en.wikipedia.org/wiki/Phoenix,_Arizona), a change of +11.2%. "
    "Although Phoenix added more people in absolute terms, Frisco changed more by percentage."
)

_FULL_MULTI = (
    "Frisco, Texas:\n"
    "  2010 census: 116,989\n"
    "  2020 census: 200,509\n"
    "  percentage change: +71.4%\n"
    "  source: https://en.wikipedia.org/wiki/Frisco,_Texas\n"
    "Phoenix, Arizona:\n"
    "  2010 census: 1,445,632\n"
    "  2020 census: 1,608,139\n"
    "  percentage change: +11.2%\n"
    "  source: https://en.wikipedia.org/wiki/Phoenix,_Arizona\n"
    "Larger percentage change:\n"
    "  Frisco\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_percent_winner(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_percentage(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer, newline-heavy layout: the keystone ("Frisco" under "Larger percentage change:"),
    # the four figures, the percentage value and both citations must all still register.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_percent_winner(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_percentage(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_deliverables_list_primary_slot_drives_keystone():
    # Keystone reads deliverables[0]; coverage/value read the whole deliverables list.
    r = {
        "deliverables": [
            "Frisco changed more by percentage than Phoenix.",
            "Frisco's percentage change is about 71.4%.",
            "Frisco 116,989 -> 200,509; Phoenix 1,445,632 -> 1,608,139.",
            "https://en.wikipedia.org/wiki/Frisco,_Texas ; "
            "https://en.wikipedia.org/wiki/Phoenix,_Arizona",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_percent_winner(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_percentage(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_absolute_trap_answer_gates_to_zero():
    # The cheap-model error: compares absolute deltas and names Phoenix (larger absolute change).
    # Even though both cities and all four figures appear, every credit-bearing check is 0 except
    # the UN-gated coverage diagnostic.
    wrong = (
        "Frisco grew from 116,989 to 200,509 (added 83,520); Phoenix grew from 1,445,632 to "
        "1,608,139 (added 162,507). Phoenix added more people than Frisco, so Phoenix changed "
        "more. Sources: https://en.wikipedia.org/wiki/Frisco,_Texas and "
        "https://en.wikipedia.org/wiki/Phoenix,_Arizona."
    )
    r = _r(wrong)
    assert t.validate_keystone_percent_winner(r, _OBS)["score"] == 0.0
    assert t.validate_winner_percentage(r, _OBS)["score"] == 0.0   # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0            # gated on keystone
    assert t.validate_coverage(r, _OBS)["score"] == 1.0            # UN-gated: four figures retained


def test_names_both_without_picking_fails_keystone():
    # Lists both cities but does not assert which changed more -> keystone not satisfied.
    r = _r(
        "The two cities are Frisco, Texas and Phoenix, Arizona; both grew between 2010 and 2020."
    )
    assert t.validate_keystone_percent_winner(r, _OBS)["score"] == 0.0


def test_wrong_base_percentage_outside_tolerance_zeroes_value():
    # Correct winner (keystone passes) but the percentage is computed off the wrong base
    # ((later-earlier)/later = 41.7%), which is outside the +/- 2-point window around 71.4%.
    text = (
        "Frisco changed more by percentage. Frisco: 116,989 -> 200,509, a 41.7% change "
        "(https://en.wikipedia.org/wiki/Frisco,_Texas). "
        "Phoenix: 1,445,632 -> 1,608,139 (https://en.wikipedia.org/wiki/Phoenix,_Arizona)."
    )
    r = _r(text)
    assert t.validate_keystone_percent_winner(r, _OBS)["passed"]
    assert t.validate_winner_percentage(r, _OBS)["score"] == 0.0   # 41.7% not within 69.4-73.4%
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    # Only Frisco's two figures gathered (keystone still asserted) -> coverage = 2/4.
    text = (
        "Frisco changed more by percentage: from 116,989 (2010) to 200,509 (2020), about 71.4% "
        "(https://en.wikipedia.org/wiki/Frisco,_Texas)."
    )
    r = _r(text)
    assert abs(t.validate_coverage(r, _OBS)["score"] - 0.5) < 1e-9   # 2 of 4 figures
    assert t.validate_keystone_percent_winner(r, _OBS)["passed"]
    assert t.validate_winner_percentage(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]
    assert abs(t.validate_visits(r, {"visit": {"count": 1}})["score"] - 0.5) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 2}})["score"] == 1.0


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    of Frisco as the larger-percentage city must collapse the keystone gate (and everything gated
    on it) to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_percent_winner(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_percent_winner(r, ungrounded_obs)["passed"] is False
    assert t.validate_winner_percentage(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    # UN-gated coverage is unaffected by the grounding gate (it scans raw text, not the keystone).
    assert t.validate_coverage(r, ungrounded_obs)["score"] == 1.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_percent_winner(r, ungrounded_obs)["score"],
        t.validate_winner_percentage(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_compiled_plan_validates_and_is_four_leaf_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 4
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True           # four independent values, one parallel wave
    assert struct["waves"] == [["frisco_2010", "frisco_2020", "phoenix_2010", "phoenix_2020"]]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the two GIVEN cities and two GIVEN census years may appear, but no population
    # value, no percentage, and no statement of which city won.
    for leak in ("116", "989", "200,509", "200509", "1,445", "445632",
                 "1,608", "608,139", "608139", "71", "11.2"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # No pre-stated winner: neither city sits next to a 'changed more / larger %' assertion.
    assert not t._FRISCO_WINS.search(plan["aggregation"])
