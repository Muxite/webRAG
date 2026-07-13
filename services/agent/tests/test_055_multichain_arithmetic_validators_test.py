"""
Offline unit tests for the multi-chain + terminal-arithmetic task (test 055) — free.

Cover the computed keystone gate (|1865 - 1746| = 119), the UN-gated breadth diagnostic (both
founding years gathered, retained even when the arithmetic is botched), the keystone-gated
secondaries (each chain's university resolved + citation), correct answer in both single- and
multi-line layout, a wrong/parametric answer that gates to zero, and that the compiled plan is a
genuine two-chain DAG (two parallel waves, two dependency edges) that templates the upstream
universities and leaks no author / university / year / difference.
"""
from agent.app.idea_tests import test_055_tier5_multichain_arithmetic as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 4}}


_FULL_SINGLE = (
    "Chain A: 'The Shining' -> Stephen King -> University of Maine "
    "(https://en.wikipedia.org/wiki/University_of_Maine), founded 1865. "
    "Chain B: 'The Great Gatsby' -> F. Scott Fitzgerald -> Princeton University "
    "(https://en.wikipedia.org/wiki/Princeton_University), founded 1746. "
    "Absolute difference of the founding years = 1865 - 1746 = 119 years."
)

_FULL_MULTI = (
    "Chain A:\n"
    "  Author of 'The Shining': Stephen King.\n"
    "  University: University of Maine (https://en.wikipedia.org/wiki/University_of_Maine).\n"
    "  Founding year: 1865.\n"
    "Chain B:\n"
    "  Author of 'The Great Gatsby': F. Scott Fitzgerald.\n"
    "  University: Princeton University (https://en.wikipedia.org/wiki/Princeton_University).\n"
    "  Founding year: 1746.\n"
    "Absolute difference between the two founding years:\n"
    "119\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_difference(r, _OBS)["score"] == 1.0
    assert t.validate_breadth_years(r, _OBS)["score"] == 1.0
    assert t.validate_chains_resolved(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy layout: the keystone (\b119\b on its own line), the breadth
    # years, the chain universities and the citations must all still register.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_difference(r, _OBS)["score"] == 1.0
    assert t.validate_breadth_years(r, _OBS)["score"] == 1.0
    assert t.validate_chains_resolved(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_value_gates_keystone_but_keeps_breadth():
    # GROUNDING-GATE fix: the correct keystone VALUE STRING alone must NOT earn credit if the
    # agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory
    # guess must collapse the keystone gate (and everything gated on it), while the UN-gated
    # breadth diagnostic (both founding years) is deliberately preserved.
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_difference(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_difference(r, ungrounded_obs)["passed"] is False
    assert t.validate_chains_resolved(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_breadth_years(r, ungrounded_obs)["score"] == 1.0
    all_scores = [f(r, ungrounded_obs)["score"] for f in t.get_validation_functions()]
    assert sum(all_scores) / len(all_scores) < 0.75


def test_wrong_parametric_answer_gates_to_zero():
    # A confident-but-wrong answer: wrong universities/years and a wrong difference. Even though
    # it *mentions* both universities, every credit-bearing check must be 0 (keystone absent).
    wrong = (
        "From memory: Stephen King went to the University of Maine (founded 1862) and "
        "F. Scott Fitzgerald went to Princeton (founded 1742), so the difference is 120 years."
    )
    r = _r(wrong)
    assert t.validate_keystone_difference(r, _OBS)["score"] == 0.0
    assert t.validate_breadth_years(r, _OBS)["score"] == 0.0      # neither real year (1865/1746)
    assert t.validate_chains_resolved(r, _OBS)["score"] == 0.0    # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0          # gated on keystone


def test_missing_keystone_keeps_breadth_but_gates_secondaries():
    # Both chains resolved (universities + years + URLs) but the terminal subtraction slipped
    # (= 121, not 119). The UN-gated breadth diagnostic stays full; the gated secondaries zero out.
    text = _FULL_SINGLE.replace("= 119 years", "= 121 years")
    r = _r(text)
    assert not t.validate_keystone_difference(r, _OBS)["passed"]
    assert t.validate_breadth_years(r, _OBS)["score"] == 1.0       # both years still gathered
    assert t.validate_chains_resolved(r, _OBS)["score"] == 0.0     # gated on keystone
    assert t.validate_citation(r, _OBS)["score"] == 0.0           # gated on keystone


def test_partial_breadth_scores_fraction():
    # Only chain A's year gathered (1865), but the difference is asserted -> keystone can pass
    # while breadth registers exactly one of two inputs.
    text = (
        "University of Maine was founded in 1865 "
        "(https://en.wikipedia.org/wiki/University_of_Maine); the computed difference is 119."
    )
    r = _r(text)
    assert abs(t.validate_breadth_years(r, _OBS)["score"] - 0.5) < 1e-9   # only 1865
    assert t.validate_keystone_difference(r, _OBS)["passed"]


def test_low_visits_scores_fraction():
    r = _r(_FULL_SINGLE)
    assert abs(t.validate_visits(r, {"visit": {"count": 2}})["score"] - 0.5) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_compiled_plan_validates_and_is_a_two_chain_dag():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 4
    assert struct["edge_count"] == 2
    # Two parallel chains: a university wave feeding a founding-year wave.
    assert struct["waves"] == [["a_univ", "b_univ"], ["a_year", "b_year"]]
    assert struct["edges"] == ["a_univ->a_year", "b_univ->b_year"]
    # Each year leaf templates its upstream university id (real dependency edges, not duplicates).
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{a_univ}" in by_id["a_year"]["instruction"]
    assert "{b_univ}" in by_id["b_year"]["instruction"]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: names the two GIVEN novels but no author, university, year, or the answer.
    for leak in ("king", "fitzgerald", "maine", "princeton", "1865", "1746", "119"):
        assert leak not in blob, f"plan leaks {leak!r}"
