"""
Offline unit tests for the median-by-date ordering task (test 067) — free, no LLM.

Cover the keystone gate (Salto Grande Dam = the median by opening year), in single- and multi-line
layout and via the deliverables[0] primary slot; the parametric decoys (Bhumibol Dam = earliest,
Bakun Dam = latest, Daniel-Johnson Dam = the engineering record-holder) gating every credit-bearing
check to zero while the UN-gated coverage diagnostic is retained; the "lists all five without
picking" case failing the keystone; the gated median-year check; partial coverage scoring an exact
fraction; the visit gate; the GROUNDING-GATE requirement (a correct-but-ungrounded answer must
collapse to near-zero); and that the compiled plan is a well-formed five-leaf pure-fan-out DAG that
leaks no opening year or winner.
"""
from agent.app.idea_tests import test_067_tier5_median_by_date as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}


_FULL_SINGLE = (
    "Opening years read from each dam's infobox: Bhumibol Dam 1964 "
    "(https://en.wikipedia.org/wiki/Bhumibol_Dam); Daniel-Johnson Dam 1970 "
    "(https://en.wikipedia.org/wiki/Daniel-Johnson_Dam); Salto Grande Dam 1979 "
    "(https://en.wikipedia.org/wiki/Salto_Grande_Dam); Merowe Dam 2009 "
    "(https://en.wikipedia.org/wiki/Merowe_Dam); Bakun Dam 2011 "
    "(https://en.wikipedia.org/wiki/Bakun_Dam). Sorted chronologically: 1964 < 1970 < 1979 < 2009 "
    "< 2011, so the median (3rd) is the Salto Grande Dam, opened in 1979."
)

_FULL_MULTI = (
    "Opening years, from each infobox:\n"
    "  Bhumibol Dam — 1964 — https://en.wikipedia.org/wiki/Bhumibol_Dam\n"
    "  Daniel-Johnson Dam — 1970 — https://en.wikipedia.org/wiki/Daniel-Johnson_Dam\n"
    "  Salto Grande Dam — 1979 — https://en.wikipedia.org/wiki/Salto_Grande_Dam\n"
    "  Merowe Dam — 2009 — https://en.wikipedia.org/wiki/Merowe_Dam\n"
    "  Bakun Dam — 2011 — https://en.wikipedia.org/wiki/Bakun_Dam\n"
    "Median (3rd) by opening year:\n"
    "  Salto Grande Dam (1979)\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_median(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_median_year(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_median(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_median_year(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_deliverables_list_primary_slot_drives_keystone():
    r = {
        "deliverables": [
            "The Salto Grande Dam is the median by opening year (3rd of five).",
            "Its opening year is 1979.",
            "Bhumibol 1964; Daniel-Johnson 1970; Salto Grande 1979; Merowe 2009; Bakun 2011.",
            "Sources: https://en.wikipedia.org/wiki/Bhumibol_Dam ; "
            "https://en.wikipedia.org/wiki/Daniel-Johnson_Dam ; "
            "https://en.wikipedia.org/wiki/Salto_Grande_Dam ; "
            "https://en.wikipedia.org/wiki/Merowe_Dam ; https://en.wikipedia.org/wiki/Bakun_Dam",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_median(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_median_year(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_earliest_shortcut_gates_to_zero():
    wrong = (
        "Bhumibol Dam is the earliest of the five (1964), so it is the answer "
        "(https://en.wikipedia.org/wiki/Bhumibol_Dam). For completeness: Daniel-Johnson Dam 1970 "
        "(https://en.wikipedia.org/wiki/Daniel-Johnson_Dam); Salto Grande Dam 1979 "
        "(https://en.wikipedia.org/wiki/Salto_Grande_Dam); Merowe Dam 2009 "
        "(https://en.wikipedia.org/wiki/Merowe_Dam); Bakun Dam 2011 "
        "(https://en.wikipedia.org/wiki/Bakun_Dam)."
    )
    r = _r(wrong)
    assert t.validate_keystone_median(r, _OBS)["score"] == 0.0
    assert t.validate_median_year(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_engineering_recordholder_shortcut_gates_to_zero():
    r = {
        "deliverables": [
            "Daniel-Johnson Dam is the world's largest arch-and-buttress dam, so it is the answer.",
            "Daniel-Johnson 1970; Bhumibol 1964; Salto Grande 1979; Merowe 2009; Bakun 2011.",
            "https://en.wikipedia.org/wiki/Daniel-Johnson_Dam",
        ],
        "output": {"final_deliverable": ""},
    }
    assert t.validate_keystone_median(r, _OBS)["score"] == 0.0
    assert t.validate_median_year(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0


def test_lists_all_without_picking_fails_keystone():
    r = _r(
        "The five dams are Bhumibol Dam, Daniel-Johnson Dam, Salto Grande Dam, Merowe Dam and "
        "Bakun Dam; each is a major hydroelectric facility."
    )
    assert t.validate_keystone_median(r, _OBS)["score"] == 0.0


def test_partial_coverage_scores_fraction():
    text = (
        "Salto Grande Dam is the median by opening year, 1979, sandwiched between Daniel-Johnson "
        "Dam (1970) and Merowe Dam (2009) (https://en.wikipedia.org/wiki/Salto_Grande_Dam)."
    )
    r = _r(text)
    assert abs(t.validate_coverage(r, _OBS)["score"] - (3.0 / 5.0)) < 1e-9
    assert t.validate_keystone_median(r, _OBS)["passed"]
    assert t.validate_median_year(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert not t.validate_visits(r, {"visit": {"count": 0}})["passed"]
    assert t.validate_visits(r, {"visit": {"count": 5}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"]


def test_ungrounded_correct_value_gates_to_zero():
    """Grounding requirement: the correct keystone VALUE STRING alone must NOT earn credit if the
    agent never actually visited a page (visit.count == 0) — an ungrounded parametric-memory guess
    of Salto Grande Dam as the median must collapse the keystone gate (and everything gated on it)
    to 0."""
    r = _r(_FULL_SINGLE)
    ungrounded_obs = {"visit": {"count": 0}}
    assert t.validate_keystone_median(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_keystone_median(r, ungrounded_obs)["passed"] is False
    assert t.validate_median_year(r, ungrounded_obs)["score"] == 0.0
    assert t.validate_citation(r, ungrounded_obs)["score"] == 0.0
    # Coverage is CAPPED BY visit count (F29 fix): a 0-visit run banks 0 coverage credit even
    # though the raw text names all five dams, closing the "parametric recall" leak.
    assert t.validate_coverage(r, ungrounded_obs)["score"] == 0.0
    scores = [
        t.validate_visits(r, ungrounded_obs)["score"],
        t.validate_keystone_median(r, ungrounded_obs)["score"],
        t.validate_median_year(r, ungrounded_obs)["score"],
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
    assert struct["waves"] == [[
        "bhumibol_opened", "daniel_johnson_opened", "salto_grande_opened", "merowe_opened",
        "bakun_opened",
    ]]


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("1964", "1970", "1979", "2009", "2011"):
        assert leak not in blob, f"plan leaks {leak!r}"
    assert not t._SALTO_WINS.search(plan["aggregation"])
