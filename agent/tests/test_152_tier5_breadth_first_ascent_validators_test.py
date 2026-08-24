"""
Offline unit tests for the tier-5 7-way fan-out & aggregation (argmax) task (test 152) —
free, no LLM.

Adversarial cases: a grounded full answer in BOTH a single-line and a multi-line report
layout (both must reach 1.0 on every check), a wrong keystone (the argmax gate collapses to
0 and the citations secondary short-circuits with it, while the UN-gated breadth diagnostic
retains its full value), an ungrounded-but-correct parametric-memory answer (visit gate),
partial coverage at an exact fraction, a bare coverage row that must NOT be read as the
superlative verdict, and the compiled plan being well-formed, fully parallel and
leak-free.
"""
from agent.app.idea_tests import test_152_tier5_breadth_first_ascent_argmax as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 7}}

_ROWS = (
    "Mont Blanc -> 1786 (https://en.wikipedia.org/wiki/Mont_Blanc)\n"
    "Matterhorn -> 1865 (https://en.wikipedia.org/wiki/Matterhorn)\n"
    "Mount Kilimanjaro -> 1889 (https://en.wikipedia.org/wiki/Mount_Kilimanjaro)\n"
    "Aconcagua -> 1897 (https://en.wikipedia.org/wiki/Aconcagua)\n"
    "Mount Erebus -> 1908 (https://en.wikipedia.org/wiki/Mount_Erebus)\n"
    "Denali -> 1913 (https://en.wikipedia.org/wiki/Denali)\n"
    "Vinson Massif -> 1966 (https://en.wikipedia.org/wiki/Vinson_Massif)\n"
)

# Multi-line layout: the superlative cue and the peak name are separated by a NEWLINE, which
# the [^.] proximity window must tolerate.
_FULL_MULTILINE = "Most recent first ascent:\nVinson Massif, 1966.\n\n" + _ROWS

# Single-line layout: everything on one line.
_FULL_SINGLELINE = (
    "The peak first climbed most recently is Vinson Massif (1966). Rows: " +
    _ROWS.replace("\n", " ")
)


def test_grounded_correct_answer_scores_all_multiline():
    r = _r(_FULL_MULTILINE)
    assert t.validate_keystone_latest(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0
    assert all(f(r, _OBS)["passed"] for f in t.get_validation_functions())


def test_grounded_correct_answer_scores_all_singleline():
    r = _r(_FULL_SINGLELINE)
    assert t.validate_keystone_latest(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_wrong_keystone_gates_citations_but_keeps_coverage():
    # Names Denali (the runner-up, 1913) as the most recent -> keystone must be 0.
    r = _r("Most recent first ascent: Denali, 1913.\n\n" + _ROWS)
    assert t.validate_keystone_latest(r, _OBS)["score"] == 0.0
    assert t.validate_keystone_latest(r, _OBS)["passed"] is False
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["passed"] is False
    # UN-gated breadth diagnostic survives: all seven facts were still gathered.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_bare_coverage_row_is_not_the_verdict():
    # The rows alone contain "Vinson Massif -> 1966" but no superlative claim: only a TRUE
    # superlative trigger may satisfy the keystone.
    r = _r(_ROWS)
    assert t.validate_keystone_latest(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_value_gates_keystone_and_citations_but_keeps_coverage():
    r = _r(_FULL_MULTILINE)
    ungrounded = {"visit": {"count": 0}}
    assert t.validate_keystone_latest(r, ungrounded)["score"] == 0.0
    assert t.validate_citations(r, ungrounded)["score"] == 0.0
    assert t.validate_coverage(r, ungrounded)["score"] == 1.0
    all_scores = [f(r, ungrounded)["score"] for f in t.get_validation_functions()]
    assert sum(all_scores) / len(all_scores) < 0.75


def test_partial_coverage_exact_fraction():
    # Only the first four rows reported, with the correct verdict still stated.
    four = "".join(_ROWS.splitlines(keepends=True)[:4])
    r = _r("Most recent first ascent: Vinson Massif, 1966.\n\n" + four)
    cov = t.validate_coverage(r, _OBS)
    # Vinson's own row is missing, but the verdict line supplies name+year -> 5/7.
    assert cov["score"] == 5 / 7
    assert cov["passed"] is False
    # Keystone is independent of coverage and still holds.
    assert t.validate_keystone_latest(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_MULTILINE)
    assert t.validate_visits(r, {"visit": {"count": 7}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 5}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_compiled_plan_is_fully_parallel_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    assert len(plan["leaves"]) == 7
    # Genuinely independent arms: NO leaf may declare a dependency.
    assert all(not leaf.get("depends_on") for leaf in plan["leaves"])
    assert len({leaf["id"] for leaf in plan["leaves"]}) == 7
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("1966", "1913", "1786", "1908", "1897", "1889", "1865"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # Strongest leak test: the plan text itself must not satisfy the keystone regex, i.e. the
    # scaffold never asserts WHICH peak is the argmax.
    assert t._LATEST_NEAR_VINSON.search(blob) is None
    assert t._KEYSTONE_YEAR.search(blob) is None
