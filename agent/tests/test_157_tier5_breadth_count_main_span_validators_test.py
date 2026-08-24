"""
Offline unit tests for the tier-5 breadth COUNT-WITH-CONDITION task (test 157) — free, no LLM.

Adversarial cases: a grounded full answer in BOTH a per-row table layout and a two-named-lists
layout (both must reach 1.0 on every check), an off-by-one count from a misclassified boundary
item (keystone collapses to 0 and both gated secondaries short-circuit with it, while the
UN-gated coverage/classification diagnostics keep their real values), the naive 'all seven'
answer, an ungrounded parametric-memory answer (visit gate), partial coverage at an exact
fraction, the visit gate itself, the threshold-margin fixture invariant, and the compiled plan
being well-formed, fully parallel and leak-free.
"""
from agent.app.idea_tests import test_157_tier5_breadth_count_main_span as t
from agent.app.testing import compiled_plan as cp


def _r(text, primary=None):
    out = {"output": {"final_deliverable": text}}
    if primary is not None:
        out["deliverables"] = [primary]
    return out


_OBS = {"visit": {"count": 7}}

# Per-row table layout: name — span — verdict — URL (the URL's dots sit AFTER the verdict).
_ROWS = (
    "Xihoumen Bridge (China): longest span 1,650 m - YES, exceeds 1,200 m - "
    "https://en.wikipedia.org/wiki/Xihoumen_Bridge\n"
    "Yi Sun-sin Bridge (South Korea): longest span 1,545 m - YES, exceeds 1,200 m - "
    "https://en.wikipedia.org/wiki/Yi_Sun-sin_Bridge\n"
    "Yavuz Sultan Selim Bridge (Turkey): longest span 1,408 m - YES, exceeds 1,200 m - "
    "https://en.wikipedia.org/wiki/Yavuz_Sultan_Selim_Bridge\n"
    "Jiangyin Yangtze River Bridge (China): longest span 1,385 m - YES, exceeds 1,200 m - "
    "https://en.wikipedia.org/wiki/Jiangyin_Yangtze_River_Bridge\n"
    "Onaruto Bridge (Japan): longest span 876 m - NO, does not exceed 1,200 m - "
    "https://en.wikipedia.org/wiki/%C5%8Cnaruto_Bridge\n"
    "Askoy Bridge (Norway): longest span 850 m - NO, does not exceed 1,200 m - "
    "https://en.wikipedia.org/wiki/Ask%C3%B8y_Bridge\n"
    "Angostura Bridge (Venezuela): longest span 712 m - NO, does not exceed 1,200 m - "
    "https://en.wikipedia.org/wiki/Angostura_Bridge\n"
)

_FULL_TABLE = "Count: 4 bridges have a longest span greater than 1,200 m\n\n" + _ROWS

# Two-named-lists layout: the verdict is a LIST HEADER preceding several names, and the header
# and its members are separated by newlines — the backward, segment-scoped verdict path.
_FULL_LISTS = (
    "Answer: 4\n\n"
    "Exceeding 1,200 m:\n"
    "Xihoumen Bridge 1,650 m (https://en_wikipedia_org/wiki/Xihoumen_Bridge)\n"
    "Yi Sun-sin Bridge 1,545 m (https://en_wikipedia_org/wiki/Yi_Sun-sin_Bridge)\n"
    "Yavuz Sultan Selim Bridge 1,408 m (https://en_wikipedia_org/wiki/Yavuz_Sultan_Selim_Bridge)\n"
    "Jiangyin Yangtze River Bridge 1,385 m "
    "(https://en_wikipedia_org/wiki/Jiangyin_Yangtze_River_Bridge)\n\n"
    "Below 1,200 m:\n"
    "Onaruto Bridge 876 m (https://en_wikipedia_org/wiki/%C5%8Cnaruto_Bridge)\n"
    "Askoy Bridge 850 m (https://en_wikipedia_org/wiki/Ask%C3%B8y_Bridge)\n"
    "Angostura Bridge 712 m (https://en_wikipedia_org/wiki/Angostura_Bridge)\n"
)


def test_grounded_correct_answer_scores_all_table_layout():
    r = _r(_FULL_TABLE)
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_classification(r, _OBS)["score"] == 1.0
    assert t.validate_passing_bridges(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0
    assert all(f(r, _OBS)["passed"] for f in t.get_validation_functions())


def test_grounded_correct_answer_scores_all_two_list_layout():
    r = _r(_FULL_LISTS)
    assert all(f(r, _OBS)["score"] == 1.0 for f in t.get_validation_functions())


def test_off_by_one_count_gates_secondaries_but_keeps_diagnostics():
    """Ōnaruto (876 m) wrongly counted as a passer -> count 5. The keystone and both gated
    secondaries collapse to 0; coverage stays full and classification loses exactly that item."""
    bad = _ROWS.replace(
        "Onaruto Bridge (Japan): longest span 876 m - NO, does not exceed 1,200 m",
        "Onaruto Bridge (Japan): longest span 876 m - YES, exceeds 1,200 m",
    )
    r = _r("Count: 5 bridges exceed 1,200 m\n\n" + bad)
    assert t.validate_keystone_count(r, _OBS)["score"] == 0.0
    assert t.validate_keystone_count(r, _OBS)["passed"] is False
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    assert t.validate_passing_bridges(r, _OBS)["score"] == 0.0
    # UN-gated diagnostics survive the wrong count.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    cls = t.validate_classification(r, _OBS)
    assert cls["score"] == 6 / 7
    assert "Ōnaruto Bridge" in cls["reason"]


def test_naive_all_seven_fails_keystone():
    r = _r("All 7 of the bridges exceed 1,200 m.\n\n" + _ROWS)
    assert t.validate_keystone_count(r, _OBS)["score"] == 0.0


def test_dropped_arm_count_three_fails_keystone():
    six = "".join(_ROWS.splitlines(keepends=True)[:6])
    r = _r("Count: 3\n\n" + six, primary="3")
    assert t.validate_keystone_count(r, _OBS)["score"] == 0.0


def test_ungrounded_correct_value_gates_keystone_and_all_credit():
    ungrounded = {"visit": {"count": 0}}
    r = _r(_FULL_TABLE)
    assert t.validate_keystone_count(r, ungrounded)["score"] == 0.0
    assert t.validate_citation(r, ungrounded)["score"] == 0.0
    assert t.validate_passing_bridges(r, ungrounded)["score"] == 0.0
    # Visit-capped diagnostics also bank nothing without a single page read.
    assert t.validate_coverage(r, ungrounded)["score"] == 0.0
    assert t.validate_classification(r, ungrounded)["score"] == 0.0
    assert t.validate_visits(r, ungrounded)["score"] == 0.0


def test_partial_coverage_exact_fraction():
    """Only four arms gathered, but the count still stated correctly: the keystone holds while
    the breadth diagnostics report exactly 4/7."""
    four = "".join(_ROWS.splitlines(keepends=True)[:4])
    r = _r("Count: 4\n\n" + four, primary="4")
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 4 / 7
    assert t.validate_coverage(r, _OBS)["passed"] is False
    assert t.validate_classification(r, _OBS)["score"] == 4 / 7


def test_coverage_capped_by_visit_count():
    r = _r(_FULL_TABLE)
    assert t.validate_coverage(r, {"visit": {"count": 3}})["score"] == 3 / 7
    assert t.validate_classification(r, {"visit": {"count": 3}})["score"] == 3 / 7


def test_visit_gate():
    r = _r(_FULL_TABLE)
    assert t.validate_visits(r, {"visit": {"count": 7}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 6}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 5}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0


def test_boundary_margins_are_safe_and_count_is_midrange():
    """Fixture invariant: no item sits near the threshold, and the answer is not 0, N or N-1."""
    assert t.KEYSTONE_COUNT == 4
    assert len(t.ENTITIES) == 7
    assert 1 < t.KEYSTONE_COUNT < len(t.ENTITIES) - 1
    spans = [e["span"] for e in t.ENTITIES]
    assert len(set(spans)) == len(spans), "spans must be distinct (no coverage cross-crediting)"
    for e in t.ENTITIES:
        margin = e["span"] - t.THRESHOLD
        assert (margin > 0) is e["passes"]
        assert abs(margin) >= 180, f"{e['name']} sits only {abs(margin)} m from the threshold"


def test_real_composer_render_scores_every_check():
    """Regression guard: the answer shape the REAL deterministic composer emits for this plan's
    ``count_threshold`` composition must satisfy every validator, including the classification
    diagnostic (which has to read the composer's '(>1,200 m? yes/no)' verdict rendering)."""
    from agent.app.testing.execution_compiled import _compose_count_threshold

    plan = t.get_compiled_plan()
    results = {
        f"{e['key']}_span": (
            f"The longest span is {e['span']:,} m "
            f"(https://en.wikipedia.org/wiki/{e['key']}_bridge)"
        )
        for e in t.ENTITIES
    }
    composed = _compose_count_threshold(plan["leaves"], results, plan["composition"])
    r = _r(composed, primary=composed)
    for fn in (t.validate_keystone_count, t.validate_coverage, t.validate_classification,
               t.validate_passing_bridges):
        out = fn(r, _OBS)
        assert out["score"] == 1.0, f"{out['check']} scored {out['score']}: {out['reason']}"


def test_compiled_plan_is_fully_parallel_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    assert len(plan["leaves"]) == 7
    # Genuinely independent arms: NO leaf may declare a dependency.
    assert all(not leaf.get("depends_on") for leaf in plan["leaves"])
    assert len({leaf["id"] for leaf in plan["leaves"]}) == 7
    assert plan["composition"]["op"] in cp.COMPOSITION_OPS
    assert plan["composition"]["threshold"] == t.THRESHOLD
    blob = " ".join(str(l) for l in plan["leaves"]) + " " + plan["aggregation"]
    for leak in ("1,650", "1650", "1,545", "1545", "1,408", "1408", "1,385", "1385",
                 "876", "850", "712"):
        assert leak not in blob, f"plan leaks {leak!r}"
    # Strongest leak test: the plan text itself must not satisfy the keystone gate, i.e. the
    # scaffold never asserts the count, and never states a per-bridge verdict.
    assert not t._keystone_ok(_r(blob, primary=blob), {"visit": {"count": 1}})
    for e in t.ENTITIES:
        assert t._verdict_for(blob, e) == "", f"plan leaks a verdict for {e['name']}"
