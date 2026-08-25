"""
Offline adversarial unit tests for test 304 (stale-vs-current source conflict requiring an
explicit verification hop). Free, no LLM, no network.

Covers: full correct answer in single-line AND multi-line layout (1.0 across every validator),
the "took the first confident source at face value" failure (reports only the supplied page's
3,754 m -> keystone 0, secondaries gated 0, coverage retained), the "reported the current value
but never verified" failure (3,724 m alone, no stale verdict -> keystone 0), the inverted-roles
answer, the ungrounded/no-visit answer, partial coverage as an exact fraction, and the compiled
plan being well-formed and leak-free.
"""
from agent.app.idea_tests import test_304_mech_stale_source_verification as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 3}}
_NO_VISITS = {"visit": {"count": 0}}

_FULL_MULTILINE = (
    "Supplied page (https://www.mountain-forecast.com/peaks/Aoraki-Mount-Cook) publishes:\n"
    "  Elevation: 3754 m\n"
    "Verdict on that figure: SUPERSEDED / out of date.\n"
    "Currently accepted summit elevation:\n"
    "  3,724 m (12,218 ft) - https://en.wikipedia.org/wiki/Aoraki_/_Mount_Cook\n"
    "Earlier generation: 3,764 m before the rock avalanche of 14 December 1991.\n"
    "Resurvey: November 2013 GPS/photogrammetric survey by the University of Otago -\n"
    "  https://www.otago.ac.nz/surveying/potree/pub/mrc/projects/aoraki\n"
    "The sources disagree; I treat the resurvey as authoritative because it is the most recent\n"
    "primary measurement.\n"
)

_FULL_SINGLELINE = (
    "The supplied page https://www.mountain-forecast.com/peaks/Aoraki-Mount-Cook still lists "
    "3754 m, but that figure is superseded. The current accepted elevation is 3,724 m, per "
    "https://en.wikipedia.org/wiki/Aoraki_/_Mount_Cook. This discrepancy arises because 3,764 m "
    "was the height before the 1991 rock avalanche, and a 2013 University of Otago resurvey "
    "(https://www.otago.ac.nz/surveying/potree/pub/mrc/projects/aoraki) established the current "
    "value; the more recent primary survey is authoritative."
)


def _all_scores(result, obs):
    return [f(result, obs)["score"] for f in t.get_validation_functions()]


def test_full_answer_multiline_scores_everything():
    r = _r(_FULL_MULTILINE)
    assert t.validate_keystone_stale_verdict(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_conflict_resolution(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["passed"] is True
    scores = _all_scores(r, _OBS)
    assert sum(scores) / len(scores) == 1.0


def test_full_answer_singleline_scores_everything():
    r = _r(_FULL_SINGLELINE)
    scores = _all_scores(r, _OBS)
    assert min(scores) == 1.0


def test_first_source_only_fails_keystone_and_gates_secondaries():
    """The mechanism's target failure: the supplied page looks complete, the agent reports its
    figure as the answer and never verifies."""
    r = _r(
        "The elevation of Aoraki / Mount Cook is 3754 m, as published by "
        "https://www.mountain-forecast.com/peaks/Aoraki-Mount-Cook. This is the current official "
        "height of New Zealand's highest peak."
    )
    assert t.validate_keystone_stale_verdict(r, _OBS)["score"] == 0.0
    assert t.validate_conflict_resolution(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    # Un-gated breadth diagnostic still records what little was gathered.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0 / 6.0
    scores = _all_scores(r, _OBS)
    assert sum(scores) / len(scores) < 0.75


def test_current_value_without_verification_fails_keystone():
    """Reporting 3,724 m alone (the shape that satisfies tests 056/066) is NOT enough here: with
    no stale verdict on the supplied page's figure, the verification hop is unproven."""
    r = _r(
        "The current elevation of Aoraki / Mount Cook is 3,724 m (12,218 ft), per "
        "https://en.wikipedia.org/wiki/Aoraki_/_Mount_Cook."
    )
    assert t.validate_keystone_stale_verdict(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0 / 6.0


def test_inverted_roles_fails_keystone():
    r = _r(
        "Current elevation: 3,754 m (superseded value: 3,724 m). Sources: "
        "https://www.mountain-forecast.com/peaks/Aoraki-Mount-Cook and "
        "https://en.wikipedia.org/wiki/Aoraki_/_Mount_Cook"
    )
    assert t.validate_keystone_stale_verdict(r, _OBS)["score"] == 0.0
    assert t.validate_conflict_resolution(r, _OBS)["score"] == 0.0


def test_ungrounded_answer_gates_keystone_but_keeps_coverage():
    r = _r(_FULL_MULTILINE)
    assert t.validate_keystone_stale_verdict(r, _NO_VISITS)["score"] == 0.0
    assert t.validate_conflict_resolution(r, _NO_VISITS)["score"] == 0.0
    assert t.validate_citations(r, _NO_VISITS)["score"] == 0.0
    assert t.validate_visits(r, _NO_VISITS)["score"] == 0.0
    assert t.validate_visits(r, _NO_VISITS)["passed"] is False
    # Breadth diagnostic is deliberately un-gated.
    assert t.validate_coverage(r, _NO_VISITS)["score"] == 1.0
    scores = _all_scores(r, _NO_VISITS)
    assert sum(scores) / len(scores) < 0.75


def test_partial_coverage_is_an_exact_fraction():
    r = _r(
        "Supplied page lists 3754 m, which is superseded; the current accepted elevation is "
        "3,724 m. I did not establish the earlier figure or the survey details."
    )
    cov = t.validate_coverage(r, _OBS)
    assert cov["score"] == 2.0 / 6.0
    assert cov["passed"] is False
    # Keystone still earned - coverage and keystone are independent axes.
    assert t.validate_keystone_stale_verdict(r, _OBS)["score"] == 1.0


def test_visit_gate_thresholds():
    r = _r(_FULL_MULTILINE)
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True


def test_longer_number_does_not_match_keystone_figures():
    r = _r("Reported figures were 13,724 and 23,754 which are unrelated.")
    assert t.validate_coverage(r, _OBS)["score"] == 0.0


def test_compiled_plan_validates_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    blob = " ".join(str(leaf) for leaf in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("3724", "3,724", "3754", "3,754", "3764", "3,764", "2013", "1991", "otago", "sirguey", "12,218"):
        assert leak not in blob, f"plan leaks {leak!r}"


def test_task_statement_does_not_leak_the_answer():
    statement = t.get_task_statement().lower()
    for leak in ("3724", "3,724", "3754", "3,754", "3764", "3,764", "2013", "1991", "otago"):
        assert leak not in statement, f"task statement leaks {leak!r}"


def test_metadata_and_api_surface():
    meta = t.get_test_metadata()
    assert meta["test_id"] == "304"
    assert meta["level"] == "integration"
    assert meta["weight"] == "long"
    assert t.get_llm_validation_function() is None
    assert len(t.get_required_deliverables()) >= 5
    assert len(t.get_success_criteria()) >= 5
