"""
Offline unit tests for the multi-constraint AND-filter task (test 068) — free, no LLM.

Cover the keystone gate (Slovakia = the unique country satisfying landlocked AND >4M AND euro), in
single- and multi-line layout and via the deliverables[0] primary slot; the single-constraint-drop
decoys (Czech Republic = largest landlocked, Portugal/Greece = big euro countries with coastlines,
Kosovo = landlocked euro user under 4M) gating every credit-bearing check to zero while the
UN-gated coverage diagnostic is retained; the "lists all six without picking" case failing the
keystone; the enumeration-table acceptance path; the gated winner-attributes check; partial
coverage scoring an exact fraction; the visit gate; the GROUNDING-GATE requirement (a
correct-but-ungrounded answer must collapse to near-zero); and that the compiled plan is a
well-formed eighteen-leaf pure-fan-out DAG that leaks no attribute value or winner.
"""
from agent.app.idea_tests import test_068_tier5_multiconstraint_filter as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 6}}


_FULL_SINGLE = (
    "Attributes read from each country's infobox: Slovakia is landlocked, population 5,449,270, "
    "currency euro (https://en.wikipedia.org/wiki/Slovakia); Czech Republic is landlocked, "
    "population 10,915,839, currency Czech koruna (https://en.wikipedia.org/wiki/Czech_Republic); "
    "Hungary is landlocked, population 9,603,634, currency forint "
    "(https://en.wikipedia.org/wiki/Hungary); Portugal has a coastline, population 11,424,031, "
    "currency euro (https://en.wikipedia.org/wiki/Portugal); Greece has a coastline, population "
    "10,372,335, currency euro (https://en.wikipedia.org/wiki/Greece); Kosovo is landlocked, "
    "population 1,585,566, currency euro (https://en.wikipedia.org/wiki/Kosovo). "
    "Slovakia is the only country that is landlocked AND has population over 4 million AND uses "
    "the euro -- it satisfies all three constraints."
)

_FULL_MULTI = (
    "Slovakia: landlocked=yes, population=5,449,270 (yes), currency=euro (yes) "
    "-- https://en.wikipedia.org/wiki/Slovakia\n"
    "Czech Republic: landlocked=yes, population=10,915,839 (yes), currency=Czech koruna (no) "
    "-- https://en.wikipedia.org/wiki/Czech_Republic\n"
    "Hungary: landlocked=yes, population=9,603,634 (yes), currency=forint (no) "
    "-- https://en.wikipedia.org/wiki/Hungary\n"
    "Portugal: landlocked=no (coastline), population=11,424,031 (yes), currency=euro (yes) "
    "-- https://en.wikipedia.org/wiki/Portugal\n"
    "Greece: landlocked=no (coastline), population=10,372,335 (yes), currency=euro (yes) "
    "-- https://en.wikipedia.org/wiki/Greece\n"
    "Kosovo: landlocked=yes, population=1,585,566 (no), currency=euro (yes) "
    "-- https://en.wikipedia.org/wiki/Kosovo\n"
    "Unique all-three satisfier:\n"
    "  Slovakia\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_filter(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_attributes(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_filter(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_winner_attributes(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_czech_landlocked_shortcut_gates_to_zero():
    wrong = (
        "Czech Republic is the largest landlocked country of the six (population 10,915,839, "
        "currency Czech koruna), so it is the answer "
        "(https://en.wikipedia.org/wiki/Czech_Republic). For completeness: Slovakia landlocked, "
        "euro, 5,449,270 (https://en.wikipedia.org/wiki/Slovakia); Hungary landlocked, forint, "
        "9,603,634 (https://en.wikipedia.org/wiki/Hungary); Portugal coastline, euro, 11,424,031 "
        "(https://en.wikipedia.org/wiki/Portugal); Greece coastline, euro, 10,372,335 "
        "(https://en.wikipedia.org/wiki/Greece); Kosovo landlocked, euro, 1,585,566 "
        "(https://en.wikipedia.org/wiki/Kosovo)."
    )
    r = _r(wrong)
    assert t.validate_keystone_filter(r, _OBS)["score"] == 0.0
    assert t.validate_winner_attributes(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_kosovo_decoy_gates_to_zero():
    r = {
        "deliverables": [
            "Kosovo is landlocked and uses the euro, so it is the answer.",
            "Kosovo landlocked, euro, 1,585,566; Slovakia landlocked, euro, 5,449,270.",
            "https://en.wikipedia.org/wiki/Kosovo",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_filter(r, _OBS)["score"] == 0.0
    assert t.validate_winner_attributes(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_lists_all_without_picking_fails_keystone():
    r = _r(
        "The six countries are Slovakia, Czech Republic, Hungary, Portugal, Greece and Kosovo; "
        "each is in Europe."
    )
    assert t.validate_keystone_filter(r, _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = (
        "Slovakia is the unique country satisfying all three constraints: landlocked, population "
        "5,449,270, currency euro (https://en.wikipedia.org/wiki/Slovakia). Czech Republic is "
        "landlocked with population 10,915,839 and currency Czech koruna "
        "(https://en.wikipedia.org/wiki/Czech_Republic)."
    )
    r = _r(text)
    assert abs(t.validate_coverage(r, _OBS)["score"] - (2.0 / 6.0)) < 1e-9
    assert t.validate_keystone_filter(r, _OBS)["passed"]
    assert t.validate_winner_attributes(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]
    assert t.validate_visits(r, {"visit": {"count": 6}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"]


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    of Slovakia as the unique satisfier must collapse the keystone gate (and everything gated on
    it) to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_filter(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_filter(r, ungrounded_obs)["passed"] is False
    assert t.validate_winner_attributes(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    # UN-gated coverage is unaffected by the grounding gate (it scans raw text, not the keystone).
    assert t.validate_coverage(r, ungrounded_obs)["score"] == 1.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_filter(r, ungrounded_obs)["score"],
        t.validate_winner_attributes(r, ungrounded_obs)["score"],
        t.validate_citation(r, ungrounded_obs)["score"],
    ]
    assert sum(scores) / len(scores) < 0.75


def test_compiled_plan_validates_and_is_eighteen_leaf_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 18
    assert struct["edge_count"] == 0
    assert struct["is_pure_fanout"] is True


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("5,449,270", "5449270", "10,915,839", "9,603,634", "11,424,031", "10,372,335",
                 "1,585,566", "euro", "koruna", "forint"):
        # 'euro' is a GIVEN constraint name (appears in the task/plan by design); skip it.
        if leak == "euro":
            continue
        assert leak not in blob, f"plan leaks {leak!r}"
    assert not t._SLOVAKIA_WINS.search(plan["aggregation"])
