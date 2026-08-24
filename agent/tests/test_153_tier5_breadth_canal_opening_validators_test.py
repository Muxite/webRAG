"""
Offline unit tests for the tier-5 5-way fan-out & aggregation (argmin) task (test 153) —
free, no LLM.

Adversarial cases: a grounded full answer in BOTH a single-line and a multi-line report
layout (both must reach 1.0 on every check), a wrong keystone (the argmin gate collapses to 0
and the citations secondary short-circuits with it, while the UN-gated breadth diagnostic
retains its full value), an ungrounded-but-correct parametric-memory answer (visit gate),
partial coverage at an exact fraction, a bare coverage row that must NOT be read as the
superlative verdict, and the compiled plan being well-formed, fully parallel and leak-free.
"""
from agent.app.idea_tests import test_153_tier5_breadth_canal_opening_argmin as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 5}}

_ROWS = (
    "Erie Canal -> 1825 (https://en.wikipedia.org/wiki/Erie_Canal)\n"
    "Suez Canal -> 1869 (https://en.wikipedia.org/wiki/Suez_Canal)\n"
    "Corinth Canal -> 1893 (https://en.wikipedia.org/wiki/Corinth_Canal)\n"
    "Kiel Canal -> 1895 (https://en.wikipedia.org/wiki/Kiel_Canal)\n"
    "Panama Canal -> 1914 (https://en.wikipedia.org/wiki/Panama_Canal)\n"
)

# Multi-line layout: the superlative cue and the canal name are separated by a NEWLINE, which
# the [^.] proximity window must tolerate.
_FULL_MULTILINE = "Earliest-opened canal:\nErie Canal, 1825.\n\n" + _ROWS

# Single-line layout: everything on one line.
_FULL_SINGLELINE = (
    "The canal that opened earliest is the Erie Canal (1825). Rows: " +
    _ROWS.replace("\n", " ")
)


def test_grounded_correct_answer_scores_all_multiline():
    r = _r(_FULL_MULTILINE)
    assert t.validate_keystone_earliest(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0
    assert all(f(r, _OBS)["passed"] for f in t.get_validation_functions())


def test_grounded_correct_answer_scores_all_singleline():
    r = _r(_FULL_SINGLELINE)
    assert t.validate_keystone_earliest(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_wrong_keystone_gates_citations_but_keeps_coverage():
    # Names the Suez Canal (the runner-up, 1869) as the earliest -> keystone must be 0.
    r = _r("Earliest-opened canal: Suez Canal, 1869.\n\n" + _ROWS)
    assert t.validate_keystone_earliest(r, _OBS)["score"] == 0.0
    assert t.validate_keystone_earliest(r, _OBS)["passed"] is False
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["passed"] is False
    # UN-gated breadth diagnostic survives: all five facts were still gathered.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_bare_coverage_row_is_not_the_verdict():
    # The rows alone contain "Erie Canal -> 1825" but no superlative claim: only a TRUE
    # superlative trigger may satisfy the keystone.
    r = _r(_ROWS)
    assert t.validate_keystone_earliest(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_ungrounded_correct_value_gates_keystone_and_citations_but_keeps_coverage():
    r = _r(_FULL_MULTILINE)
    ungrounded = {"visit": {"count": 0}}
    assert t.validate_keystone_earliest(r, ungrounded)["score"] == 0.0
    assert t.validate_citations(r, ungrounded)["score"] == 0.0
    assert t.validate_coverage(r, ungrounded)["score"] == 1.0
    all_scores = [f(r, ungrounded)["score"] for f in t.get_validation_functions()]
    assert sum(all_scores) / len(all_scores) < 0.75


def test_partial_coverage_exact_fraction():
    # Only the last three rows reported, with the correct verdict still stated.
    three = "".join(_ROWS.splitlines(keepends=True)[2:])
    r = _r("Earliest-opened canal: Erie Canal, 1825.\n\n" + three)
    cov = t.validate_coverage(r, _OBS)
    # Suez's row is missing entirely; Erie's row is missing but the verdict line supplies
    # name+year -> 4/5.
    assert cov["score"] == 4 / 5
    assert cov["passed"] is False
    # Keystone is independent of coverage and still holds.
    assert t.validate_keystone_earliest(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_MULTILINE)
    assert t.validate_visits(r, {"visit": {"count": 5}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_compiled_plan_is_fully_parallel_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    assert len(plan["leaves"]) == 5
    # Genuinely independent arms: NO leaf may declare a dependency.
    assert all(not leaf.get("depends_on") for leaf in plan["leaves"])
    assert len({leaf["id"] for leaf in plan["leaves"]}) == 5
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    for leak in ("1825", "1869", "1893", "1895", "1914"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # Strongest leak test: the plan text itself must not satisfy the keystone regex, i.e. the
    # scaffold never asserts WHICH canal is the argmin.
    assert t._EARLIEST_NEAR_ERIE.search(blob) is None
    assert t._KEYSTONE_YEAR.search(blob) is None


def test_metadata_and_exports():
    md = t.get_test_metadata()
    assert md["test_id"] == "153"
    assert md["level"] == "graph"
    assert t.get_llm_validation_function() is None
    assert len(t.get_validation_functions()) == 4
    stmt = t.get_task_statement()
    # The statement gives the five entities but never the answer.
    for e in t.ENTRIES:
        assert e["canal"] in stmt
    assert "1825" not in stmt
