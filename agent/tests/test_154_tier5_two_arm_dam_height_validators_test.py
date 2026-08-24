"""
Offline unit tests for the tier-5 two-arm independent comparison task (test 154) — free, no
LLM.

Adversarial cases: a grounded full answer in BOTH a single-line and a multi-line report layout
(both must reach 1.0 on every check), a FLIPPED verdict (says Hoover is taller — the gate
collapses to 0 and the citations secondary short-circuits with it, while the UN-gated breadth
diagnostic keeps its full value), several flipped phrasings that must not fool the tempered
proximity windows, an ungrounded-but-correct parametric answer (visit gate), one-arm-only
coverage at an exact fraction, a bare coverage table that states no verdict, and the compiled
plan being well-formed, fully parallel and leak-free.
"""
import py_compile

from agent.app.idea_tests import test_154_tier5_two_arm_dam_height_comparison as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}

_ROWS = (
    "Grande Dixence Dam - 285 m (https://en.wikipedia.org/wiki/Grande_Dixence_Dam)\n"
    "Hoover Dam - 221.4 m (https://en.wikipedia.org/wiki/Hoover_Dam)\n"
)

# Multi-line layout: the comparative cue and the winner's name are separated by a NEWLINE,
# which the [^.] proximity windows must tolerate.
_FULL_MULTILINE = "Taller dam:\nGrande Dixence Dam, by 63.6 m.\n\n" + _ROWS

# Single-line layout: everything on one line.
_FULL_SINGLELINE = (
    "The Grande Dixence Dam is taller than the Hoover Dam, by 63.6 m. Heights: "
    + _ROWS.replace("\n", " ")
)


def test_grounded_correct_answer_scores_all_multiline():
    r = _r(_FULL_MULTILINE)
    assert t.validate_keystone_taller(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0
    assert all(f(r, _OBS)["passed"] for f in t.get_validation_functions())


def test_grounded_correct_answer_scores_all_singleline():
    r = _r(_FULL_SINGLELINE)
    assert t.validate_keystone_taller(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_inverted_phrasing_of_the_correct_verdict_also_counts():
    # "the loser is SHORTER than the winner" is the same verdict, stated the other way round.
    r = _r("The Hoover Dam is shorter than the Grande Dixence Dam.\n\n" + _ROWS)
    assert t.validate_keystone_taller(r, _OBS)["score"] == 1.0


def test_flipped_verdict_gates_citations_but_keeps_coverage():
    r = _r("Taller dam: Hoover Dam, by 63.6 m.\n\n" + _ROWS)
    assert t.validate_keystone_taller(r, _OBS)["score"] == 0.0
    assert t.validate_keystone_taller(r, _OBS)["passed"] is False
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["passed"] is False
    # UN-gated breadth diagnostic survives: both arms were still resolved.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_flipped_verdict_phrasings_do_not_fool_the_proximity_windows():
    flipped = [
        "The Hoover Dam is taller than the Grande Dixence Dam.",
        "Hoover Dam is the tallest of the two, ahead of Grande Dixence Dam.",
        "The Grande Dixence Dam is shorter than the Hoover Dam.",
        "Taller: Hoover Dam (221.4 m) vs Grande Dixence Dam (285 m).",
        "Verdict: Hoover. Grande Dixence Dam is 285 m.",
    ]
    for text in flipped:
        r = _r(text + "\n\n" + _ROWS)
        assert t.validate_keystone_taller(r, _OBS)["score"] == 0.0, text


def test_bare_coverage_table_is_not_a_verdict():
    # The rows alone carry both name+height pairs but assert no comparison.
    r = _r(_ROWS)
    assert t.validate_keystone_taller(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_verdict_without_the_winning_value_fails_the_keystone():
    r = _r("The Grande Dixence Dam is taller than the Hoover Dam (221.4 m).")
    assert t.validate_keystone_taller(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 0.5


def test_ungrounded_correct_answer_gates_keystone_and_citations_but_keeps_coverage():
    r = _r(_FULL_MULTILINE)
    ungrounded = {"visit": {"count": 0}}
    assert t.validate_keystone_taller(r, ungrounded)["score"] == 0.0
    assert t.validate_citations(r, ungrounded)["score"] == 0.0
    assert t.validate_coverage(r, ungrounded)["score"] == 1.0
    scores = [f(r, ungrounded)["score"] for f in t.get_validation_functions()]
    assert sum(scores) / len(scores) < 0.75


def test_one_arm_only_coverage_exact_fraction():
    r = _r("Grande Dixence Dam - 285 m (https://en.wikipedia.org/wiki/Grande_Dixence_Dam)")
    cov = t.validate_coverage(r, _OBS)
    assert cov["score"] == 0.5
    assert cov["passed"] is False
    assert t.validate_keystone_taller(r, _OBS)["score"] == 0.0


def test_imperial_units_still_count_for_coverage_and_keystone():
    r = _r("Taller dam:\nGrande Dixence Dam (935 ft) beats Hoover Dam (726.4 ft).")
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_keystone_taller(r, _OBS)["score"] == 1.0


def test_visit_gate():
    r = _r(_FULL_MULTILINE)
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 3}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 1}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 1}})["score"] == 0.5
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_partial_citations_exact_fraction():
    r = _r(
        "Taller dam:\nGrande Dixence Dam, 285 m, by 63.6 m over the Hoover Dam (221.4 m).\n"
        "Source: https://en.wikipedia.org/wiki/Grande_Dixence_Dam"
    )
    cit = t.validate_citations(r, _OBS)
    assert cit["score"] == 0.5
    assert cit["passed"] is False


def test_metadata_and_statement_are_shape_agnostic():
    md = t.get_test_metadata()
    assert md["test_id"] == "154"
    assert md["level"] == "graph"
    statement = t.get_task_statement().lower()
    # No sequential narrative may be implied by the wording.
    for banned in ("first find", "then look", "then find", "after you", "step 1", "next, "):
        assert banned not in statement, banned
    assert "independent" in statement
    assert len(t.get_required_deliverables()) == 3
    assert t.get_llm_validation_function() is None


def test_compiled_plan_is_fully_parallel_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    leaves = plan["leaves"]
    assert len(leaves) == 2
    # Genuinely independent arms: NEITHER leaf may declare a dependency, and neither leaf's
    # instruction may reference the other arm at all (the shape-fairness property).
    assert all(not leaf.get("depends_on") for leaf in leaves)
    assert len({leaf["id"] for leaf in leaves}) == 2
    assert "{" not in " ".join(leaf["instruction"] for leaf in leaves)
    by_id = {leaf["id"]: leaf["instruction"].lower() for leaf in leaves}
    assert "hoover" not in by_id["grande_dixence_dam"]
    assert "dixence" not in by_id["hoover_dam"]

    blob = " ".join(str(l) for l in leaves).lower() + " " + plan["aggregation"].lower()
    for leak in ("285", "935", "221", "726", "63.6"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # Strongest leak test: the plan text must not itself satisfy the keystone regex, i.e. the
    # scaffold never asserts WHICH dam is taller. The aggregation asks the question without
    # naming either dam.
    assert t._VERDICT_A_TALLER.search(blob) is None
    assert t._KEYSTONE_VALUE.search(blob) is None
    assert "dixence" not in plan["aggregation"].lower()
    assert "hoover" not in plan["aggregation"].lower()


def test_module_byte_compiles():
    py_compile.compile(t.__file__, doraise=True)
