"""
Offline unit tests for the MLB Negro-Leagues batting-leader conflicting-source task (test 132) —
free, no LLM.

Covers the keystone gate (Josh Gibson AND .371/.372) that MUST reject the reflex historical answer
(Ty Cobb / .367) and any averaged value; the UN-gated reconciliation coverage diagnostic (both
leaders surfaced by name, retained when the pick is wrong, gated on read-evidence); the keystone-
gated revision-identification and citation secondaries; single- and multi-line layout; and the
compiled plan (2 -> 1) that leaks nothing.
"""
from agent.app.idea_tests import test_132_tier5_negro_leagues_batting_leader as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}

_CITE = "https://en.wikipedia.org/wiki/List_of_Major_League_Baseball_career_batting_average_leaders"

_FULL_SINGLE = (
    "Until MLB incorporated Negro Leagues statistics in 2024, Ty Cobb (.367) was the consensus "
    "leader; he was supplanted by Josh Gibson, who is now the all-time career batting-average "
    f"leader at .371. Source: {_CITE}"
)

_FULL_MULTI = (
    "MLB career batting-average leader reconciliation:\n"
    "  former consensus (pre-2024): Ty Cobb, .367\n"
    "  current #1 (after 2024 Negro Leagues incorporation): Josh Gibson\n"
    "    .371\n"
    "Gibson supplanted Cobb under the 2024 revision.\n"
    f"  {_CITE}\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_leader(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["passed"] is True
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_leader(r, _OBS)["score"] == 1.0
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["passed"] is True


def test_accepts_mlb_dot_com_372_rounding():
    r = _r(f"Josh Gibson is the all-time leader at .372, ahead of Ty Cobb (.367). {_CITE}")
    assert t.validate_keystone_leader(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_value_scores_near_zero():
    """Right keystone value present, but zero visits (no grounding) and no source citation in text
    -> keystone and every keystone-gated secondary must collapse to 0, even though the value string
    matches."""
    r = _r(_FULL_SINGLE)
    ungrounded = {"visit": {"count": 0}}
    assert t.validate_keystone_leader(r, ungrounded)["score"] == 0.0
    assert t.validate_identifies_correct_source(r, ungrounded)["score"] == 0.0
    assert t.validate_citation(r, ungrounded)["score"] == 0.0
    overall = sum(v["score"] for v in [
        t.validate_keystone_leader(r, ungrounded),
        t.validate_identifies_correct_source(r, ungrounded),
        t.validate_citation(r, ungrounded),
    ]) / 3.0
    assert overall < 0.75


def test_wrong_source_pick_gates_but_keeps_coverage():
    wrong = (
        "The list shows Josh Gibson and Ty Cobb near the top. I report Ty Cobb as the all-time "
        f"career batting-average leader at .367. {_CITE}"
    )
    r = {"output": {"final_deliverable": "Leader: Ty Cobb, .367"},
         "deliverables": ["Leader: Ty Cobb, .367", wrong]}
    assert t.validate_keystone_leader(r, _OBS)["score"] == 0.0        # reflex wrong answer
    assert t.validate_reconciliation_coverage(r, _OBS)["score"] == 1.0  # both names present, retained
    assert t.validate_identifies_correct_source(r, _OBS)["score"] == 0.0  # gated
    assert t.validate_citation(r, _OBS)["score"] == 0.0              # gated


def test_averaged_value_gates_to_zero():
    avg = _r(f"Averaging the two batting averages gives about .369 for Josh Gibson. {_CITE}")
    assert t.validate_keystone_leader(avg, _OBS)["score"] == 0.0


def test_keystone_requires_both_name_and_number():
    assert t.validate_keystone_leader(_r("The leader is .371"), _OBS)["score"] == 0.0   # no name
    assert t.validate_keystone_leader(_r("The leader is Josh Gibson"), _OBS)["score"] == 0.0  # no number
    assert t.validate_keystone_leader(_r("Ty Cobb, .367"), _OBS)["score"] == 0.0        # reflex
    assert t.validate_keystone_leader(_r("Josh Gibson, .371"), _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    r = _r("The current leader is Josh Gibson.")  # only revised leader named
    assert abs(t.validate_reconciliation_coverage(r, _OBS)["score"] - 0.5) < 1e-9


def test_coverage_requires_read_evidence():
    r = _r(_FULL_SINGLE.replace(_CITE, ""))
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_reconciliation_coverage(r, {"visit": {"count": 1}})["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_deliverables_slot_is_primary_for_keystone():
    r = {"output": {"final_deliverable": "see structured answer"},
         "deliverables": ["Career batting leader: Josh Gibson, .371", "Cobb .367 was former leader"]}
    assert t.validate_keystone_leader(r, _OBS)["score"] == 1.0


def test_compiled_plan_validates_and_is_two_then_one():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 3
    assert struct["wave_widths"] == [2, 1]
    assert struct["waves"][1] == ["reconcile"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("gibson", ".371", ".372"):
        assert leak not in blob, f"plan leaks {leak!r}"
