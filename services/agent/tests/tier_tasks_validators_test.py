"""
Offline unit tests for the Tier-1/2 benchmark task validators (tests 048 & 049).

Free (no LLM/network): feed synthetic deliverables + observability and assert the
keystone/secondary validators score as designed — including the monotonicity lever
that the snippet-only ``minimal`` rung (0 visits) loses the ``visited_both`` credit on
the Tier-2 task while a crawling rung keeps it.
"""
from agent.app.idea_tests import test_048_tier1_single_fact as t1
from agent.app.idea_tests import test_049_tier2_two_page_combine as t2


def _result(text: str):
    return {"output": {"final_deliverable": text}}


# ---- Tier 1 (test 048) ---------------------------------------------------------

def test_t1_keystone_pass_and_citation():
    r = _result("The Eiffel Tower was completed in 1889. Source: https://en.wikipedia.org/wiki/Eiffel_Tower")
    obs = {"visit": {"count": 1}}
    ks = t1.validate_keystone_year(r, obs)
    gr = t1.validate_grounding(r, obs)
    assert ks["passed"] and ks["score"] == 1.0
    assert gr["passed"] and gr["score"] == 1.0


def test_t1_keystone_fail_short_circuits_grounding():
    r = _result("The Eiffel Tower is in Paris.")  # no year
    obs = {"visit": {"count": 1}}
    assert not t1.validate_keystone_year(r, obs)["passed"]
    gr = t1.validate_grounding(r, obs)
    assert not gr["passed"] and gr["score"] == 0.0


# ---- Tier 2 (test 049) ---------------------------------------------------------

_FULL_ANSWER = (
    "The Statue of Liberty was dedicated in 1886, while the Eiffel Tower was completed in 1889. "
    "So the Statue of Liberty was completed first, by 3 years. "
    "Sources: https://en.wikipedia.org/wiki/Statue_of_Liberty and "
    "https://en.wikipedia.org/wiki/Eiffel_Tower"
)


def test_t2_full_answer_scores_all_checks():
    r = _result(_FULL_ANSWER)
    obs = {"visit": {"count": 2}}
    checks = {c["check"]: c for c in (
        t2.validate_keystone_combination(r, obs),
        t2.validate_ordering(r, obs),
        t2.validate_grounding(r, obs),
        t2.validate_visited_both(r, obs),
    )}
    assert checks["keystone_combination"]["score"] == 1.0
    assert checks["ordering"]["score"] == 1.0
    assert checks["grounding"]["score"] == 1.0
    assert checks["visited_both"]["score"] == 1.0


def test_t2_minimal_rung_loses_visit_credit_even_if_correct():
    # Snippet-only rung can sometimes state the right answer but never visited a page.
    r = _result(_FULL_ANSWER)
    obs = {"visit": {"count": 0}}
    assert t2.validate_keystone_combination(r, obs)["passed"]
    vb = t2.validate_visited_both(r, obs)
    assert not vb["passed"] and vb["score"] == 0.0


def test_t2_missing_gap_fails_keystone_and_short_circuits():
    r = _result("Eiffel Tower 1889, Statue of Liberty 1886.")  # both years but no derived gap
    obs = {"visit": {"count": 2}}
    assert not t2.validate_keystone_combination(r, obs)["passed"]
    # secondary checks short-circuit to 0 when keystone is absent
    assert t2.validate_ordering(r, obs)["score"] == 0.0
    assert t2.validate_grounding(r, obs)["score"] == 0.0
    assert t2.validate_visited_both(r, obs)["score"] == 0.0


def test_t2_grounding_partial_one_source():
    text = _FULL_ANSWER.replace("https://en.wikipedia.org/wiki/Eiffel_Tower", "the tower page")
    r = _result(text)
    obs = {"visit": {"count": 2}}
    gr = t2.validate_grounding(r, obs)
    assert gr["score"] == 0.5 and not gr["passed"]
