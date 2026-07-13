"""
Offline unit tests for the odd-one-out / negation task (test 069) — free, no LLM.

Cover the negation-robust keystone gate (Bosnia and Herzegovina = the one that does NOT satisfy
"is landlocked"), in single- and multi-line layout and via the deliverables[0] primary slot; the
negation-flip decoy (naming Bosnia but asserting it IS landlocked) and the landlocked-rival decoys
(Austria / Slovakia asserted as the odd one out) gating every credit-bearing check to zero while
the UN-gated coverage diagnostic is retained; the "lists all five without picking" case failing the
keystone; the gated why-exception check; partial coverage scoring an exact fraction; the visit
gate; the GROUNDING-GATE requirement (a correct-but-ungrounded answer must collapse to near-zero);
and that the compiled plan is a well-formed five-leaf pure-fan-out DAG that leaks nothing about
which country is the exception.
"""
from agent.app.idea_tests import test_069_tier5_odd_one_out as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Landlocked status read from each country's lead/infobox: Austria is landlocked "
    "(https://en.wikipedia.org/wiki/Austria); Bosnia and Herzegovina has a 20 km coastline on the "
    "Adriatic Sea at Neum, so it is NOT landlocked "
    "(https://en.wikipedia.org/wiki/Bosnia_and_Herzegovina); North Macedonia is landlocked "
    "(https://en.wikipedia.org/wiki/North_Macedonia); Serbia is landlocked "
    "(https://en.wikipedia.org/wiki/Serbia); Slovakia is landlocked "
    "(https://en.wikipedia.org/wiki/Slovakia). The odd one out -- the one that does NOT satisfy "
    "'is landlocked' -- is Bosnia and Herzegovina."
)

_FULL_MULTI = (
    "Austria: landlocked -- https://en.wikipedia.org/wiki/Austria\n"
    "Bosnia and Herzegovina: NOT landlocked -- coastline on the Adriatic Sea (Neum) "
    "-- https://en.wikipedia.org/wiki/Bosnia_and_Herzegovina\n"
    "North Macedonia: landlocked -- https://en.wikipedia.org/wiki/North_Macedonia\n"
    "Serbia: landlocked -- https://en.wikipedia.org/wiki/Serbia\n"
    "Slovakia: landlocked -- https://en.wikipedia.org/wiki/Slovakia\n"
    "Odd one out (not landlocked):\n"
    "  Bosnia and Herzegovina\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_odd_one_out(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_why_exception(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_odd_one_out(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_why_exception(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_negation_flip_decoy_gates_to_zero():
    # Names Bosnia but asserts it IS landlocked -- the classic negation flip.
    wrong = (
        "Bosnia and Herzegovina is landlocked, like Austria, North Macedonia, Serbia and Slovakia "
        "(https://en.wikipedia.org/wiki/Bosnia_and_Herzegovina)."
    )
    r = _r(wrong)
    assert t.validate_keystone_odd_one_out(r, _OBS)["score"] == 0.0
    assert t.validate_why_exception(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_landlocked_rival_named_as_exception_gates_to_zero():
    r = {
        "deliverables": [
            "Austria is the odd one out -- it has a coastline unlike the others.",
            "Austria landlocked; Bosnia and Herzegovina has a coast; others landlocked.",
            "https://en.wikipedia.org/wiki/Austria",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_odd_one_out(r, _OBS)["score"] == 0.0
    assert t.validate_why_exception(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_lists_all_without_picking_fails_keystone():
    r = _r(
        "The five countries are Austria, Bosnia and Herzegovina, North Macedonia, Serbia and "
        "Slovakia; each is in Central or Southeast Europe."
    )
    assert t.validate_keystone_odd_one_out(r, _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = (
        "Bosnia and Herzegovina is the odd one out -- it has a 20 km Adriatic coastline at Neum, "
        "unlike Austria which is landlocked "
        "(https://en.wikipedia.org/wiki/Bosnia_and_Herzegovina)."
    )
    r = _r(text)
    assert abs(t.validate_coverage(r, _OBS)["score"] - (2.0 / 5.0)) < 1e-9
    assert t.validate_keystone_odd_one_out(r, _OBS)["passed"]
    assert t.validate_why_exception(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]
    assert t.validate_visits(r, {"visit": {"count": 5}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"]


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    of Bosnia and Herzegovina as the odd one out must collapse the keystone gate (and everything
    gated on it) to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_odd_one_out(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_odd_one_out(r, ungrounded_obs)["passed"] is False
    assert t.validate_why_exception(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    # UN-gated coverage is unaffected by the grounding gate (it scans raw text, not the keystone).
    assert t.validate_coverage(r, ungrounded_obs)["score"] == 1.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_odd_one_out(r, ungrounded_obs)["score"],
        t.validate_why_exception(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_compiled_plan_validates_and_is_five_leaf_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 5
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the five GIVEN countries and the neutral "landlocked or coastline?" question
    # may appear, but nothing must reveal WHICH one is the exception.
    for leak in ("neum", "adriatic", "20 km", "20-km", "20 kilomet"):
        assert leak not in blob, f"plan leaks {leak!r}"
    assert not t._BOSNIA_IS_ODD.search(plan["aggregation"])
