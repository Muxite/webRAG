"""
Offline unit tests for the tier-5 two-arm independent comparison task (test 155) — free, no
LLM.

Adversarial cases: a grounded full answer in BOTH a single-line and a multi-line report
layout (both must reach 1.0 on every check), a FLIPPED verdict (the keystone collapses to 0
and the citations secondary short-circuits with it, while the UN-gated breadth diagnostic
retains its full value), an ungrounded parametric-memory answer (visit gate), an
unsubstantiated bare verdict with no page figure, one-arm-only coverage at an exact
fraction, a both-arms-mentioned sentence that must NOT be stitched into a spurious verdict,
and the compiled plan being well-formed, fully parallel and leak-free.
"""
from agent.app.idea_tests import test_155_tier5_two_arm_wingspan_comparison as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}

_ROWS = (
    "Antonov An-225 Mriya -> wingspan 88.4 m "
    "(https://en.wikipedia.org/wiki/Antonov_An-225_Mriya)\n"
    "Hughes H-4 Hercules -> wingspan 97.51 m "
    "(https://en.wikipedia.org/wiki/Hughes_H-4_Hercules)\n"
)

# Multi-line layout: the verdict label and the winner are separated by a NEWLINE, which the
# [^.] proximity window must tolerate.
_FULL_MULTILINE = "Greater wingspan:\nHughes H-4 Hercules.\n\n" + _ROWS

# Single-line layout: everything on one line.
_FULL_SINGLELINE = (
    "The Hughes H-4 Hercules has a larger wingspan than the Antonov An-225 Mriya. Values: "
    + _ROWS.replace("\n", " ")
)


def test_grounded_correct_answer_scores_all_multiline():
    r = _r(_FULL_MULTILINE)
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0
    assert all(f(r, _OBS)["passed"] for f in t.get_validation_functions())


def test_grounded_correct_answer_scores_all_singleline():
    r = _r(_FULL_SINGLELINE)
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_inverted_phrasing_also_credits():
    r = _r("The An-225 is narrower than the Hughes H-4 Hercules.\n\n" + _ROWS)
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 1.0


def test_flipped_verdict_gates_citations_but_keeps_coverage():
    # Names the An-225 as the wider aircraft (the popular-memory trap) -> keystone must be 0.
    flipped = (
        "The Antonov An-225 Mriya has a larger wingspan than the Hughes H-4 Hercules.\n\n"
        + _ROWS
    )
    r = _r(flipped)
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 0.0
    assert t.validate_keystone_verdict(r, _OBS)["passed"] is False
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["passed"] is False
    # UN-gated breadth diagnostic survives: both arms were still gathered.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_flipped_labelled_verdict_line_scores_zero():
    r = _r("Greater wingspan: Antonov An-225 Mriya, not the Hercules.\n\n" + _ROWS)
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 0.0


def test_bare_coverage_rows_are_not_a_verdict():
    # The rows carry both names and both numbers but state no comparison at all.
    r = _r(_ROWS)
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_third_party_sentence_mentioning_both_is_not_a_verdict():
    # A tempered-gap regression: a sentence naming BOTH arms must not be stitched together.
    r = _r(
        "Compared with the Hughes H-4 Hercules, the Boeing 747 is wider than the An-225.\n\n"
        + _ROWS
    )
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 0.0


def test_framed_but_correct_verdict_still_credits():
    # The frame guard must not reject a genuine verdict that merely opens with "Compared with".
    r = _r(
        "Compared with the An-225, the Hughes H-4 Hercules has a wider wingspan than the "
        "Antonov An-225 Mriya.\n\n" + _ROWS
    )
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 1.0


def test_has_the_larger_wingspan_phrasing_credits():
    r = _r("The Hughes H-4 Hercules has the larger wingspan of the two.\n\n" + _ROWS)
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 1.0


def test_has_the_larger_wingspan_phrasing_flipped_scores_zero():
    r = _r("The Antonov An-225 Mriya has the larger wingspan of the two.\n\n" + _ROWS)
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 0.0


def test_ungrounded_verdict_without_page_figure_scores_zero():
    # Correct direction, but no wingspan value from the winner's page -> not grounded.
    r = _r("The Hughes H-4 Hercules has a larger wingspan than the Antonov An-225 Mriya.")
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_no_visits_gates_keystone_and_citations_but_keeps_coverage():
    r = _r(_FULL_MULTILINE)
    ungrounded = {"visit": {"count": 0}}
    assert t.validate_keystone_verdict(r, ungrounded)["score"] == 0.0
    assert t.validate_citations(r, ungrounded)["score"] == 0.0
    assert t.validate_coverage(r, ungrounded)["score"] == 1.0
    scores = [f(r, ungrounded)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) < 0.75


def test_partial_coverage_exact_fraction():
    # Only the An-225 arm was gathered; the verdict was never reached.
    r = _r("Antonov An-225 Mriya -> wingspan 88.4 m.")
    cov = t.validate_coverage(r, _OBS)
    assert cov["score"] == 1 / 2
    assert cov["passed"] is False
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 0.0


def test_visit_gate():
    r = _r(_FULL_MULTILINE)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_unit_variant_extraction_still_credits():
    # Feet-only figures (319 ft 11 in / 290 ft) are the same page facts in other units.
    r = _r(
        "Greater wingspan: Hughes H-4 Hercules at 319 ft 11 in, versus 290 ft for the "
        "An-225 Mriya.\n" + _ROWS.replace("88.4 m", "").replace("97.51 m", "")
    )
    assert t.validate_keystone_verdict(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_compiled_plan_is_two_independent_arms_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    assert len(plan["leaves"]) == 2
    # Shape fairness: NEITHER leaf may declare a dependency on the other arm.
    assert all(not leaf.get("depends_on") for leaf in plan["leaves"])
    assert len({leaf["id"] for leaf in plan["leaves"]}) == 2
    # Neither leaf's instruction may name the other arm's aircraft.
    ids = [leaf["id"] for leaf in plan["leaves"]]
    assert ids == ["antonov_an_225_mriya", "hughes_h_4_hercules"]
    an_leaf, herc_leaf = plan["leaves"]
    assert "hercules" not in an_leaf["instruction"].lower()
    assert "225" not in herc_leaf["instruction"]
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("88.4", "97.5", "319", "290", "spruce"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # Strongest leak test: the plan text itself must not satisfy the keystone regex, i.e. the
    # scaffold never asserts WHICH aircraft wins.
    assert t._verdict_stated(blob) is False
    assert t._KEYSTONE_VALUE.search(blob) is None


def test_metadata_and_exports():
    md = t.get_test_metadata()
    assert md["test_id"] == "155"
    assert md["level"] == "graph"
    assert t.get_llm_validation_function() is None
    assert len(t.get_validation_functions()) == 4
    stmt = t.get_task_statement().lower()
    # Shape-agnostic: posed as an open comparison, never as a prescribed A-then-B recipe.
    assert "first find" not in stmt and "then find" not in stmt
    assert "in any order" in stmt
