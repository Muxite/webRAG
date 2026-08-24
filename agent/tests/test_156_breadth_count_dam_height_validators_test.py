"""
Offline unit tests for the tier-5 independent 7-way count-with-condition task (test 156) —
free, no LLM.

Adversarial cases: a grounded full answer in BOTH a multi-line and a single-line report layout
(both must reach 1.0 on every check), a wrong keystone count in both directions plus the naive
"all seven" (the gate collapses to 0 and BOTH gated secondaries short-circuit with it, while
the two UN-gated breadth diagnostics retain their full value), an ungrounded parametric answer
(visit gate + breadth cap), partial coverage at an exact fraction, bare coverage rows that must
NOT be read as per-item verdicts, a near-threshold misclassification, the verified fixture
margins themselves, and the compiled plan being well-formed, fully parallel and leak-free.
"""
from agent.app.idea_tests import test_156_tier5_breadth_count_dam_height as t
from agent.app.testing import compiled_plan as cp


def _r(text, primary=None):
    """Result envelope: ``primary`` populates deliverables[0] (the keystone slot)."""
    out = {"output": {"final_deliverable": text}}
    if primary is not None:
        out["deliverables"] = [primary]
    return out


_OBS = {"visit": {"count": 7}}
_NO_VISITS = {"visit": {"count": 0}}

# Rows carrying BOTH the (dam, height) coverage pair AND the per-dam threshold verdict.
_VERDICT_ROWS = (
    "Nurek Dam (Tajikistan): 300 m — above 220 m (https://en.wikipedia.org/wiki/Nurek_Dam)\n"
    "Grande Dixence Dam (Switzerland): 285 m — above 220 m "
    "(https://en.wikipedia.org/wiki/Grande_Dixence_Dam)\n"
    "Enguri Dam (Georgia): 271.5 m — above 220 m (https://en.wikipedia.org/wiki/Enguri_Dam)\n"
    "Vajont Dam (Italy): 262 m — above 220 m (https://en.wikipedia.org/wiki/Vajont_Dam)\n"
    "Katse Dam (Lesotho): 185 m — below 220 m (https://en.wikipedia.org/wiki/Katse_Dam)\n"
    "Karakaya Dam (Turkey): 158 m — below 220 m "
    "(https://en.wikipedia.org/wiki/Karakaya_Dam)\n"
    "Gordon Dam (Australia): 140 m — below 220 m (https://en.wikipedia.org/wiki/Gordon_Dam)\n"
)

# Bare coverage rows: name + height only, no verdict cue anywhere. The '->' separator must NOT
# be mistaken for a '>' comparison verdict.
_BARE_ROWS = (
    "Nurek Dam -> 300 m\nGrande Dixence Dam -> 285 m\nEnguri Dam -> 271.5 m\n"
    "Vajont Dam -> 262 m\nKatse Dam -> 185 m\nKarakaya Dam -> 158 m\nGordon Dam -> 140 m\n"
)

_FULL_MULTILINE = "Count of dams taller than 220 m:\n4\n\n" + _VERDICT_ROWS
_FULL_SINGLELINE = (
    "4 of the seven dams are taller than 220 m. " + _VERDICT_ROWS.replace("\n", " ")
)


# ── happy paths ──────────────────────────────────────────────────────────────────────────

def test_grounded_correct_answer_scores_all_multiline():
    r = _r(_FULL_MULTILINE, primary="4")
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_item_classification(r, _OBS)["score"] == 1.0
    assert t.validate_passing_dams(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0
    assert all(f(r, _OBS)["passed"] for f in t.get_validation_functions())


def test_grounded_correct_answer_scores_all_singleline():
    r = _r(_FULL_SINGLELINE, primary="4")
    assert all(f(r, _OBS)["score"] == 1.0 for f in t.get_validation_functions())


def test_two_list_layout_binds_each_dam_to_its_own_heading():
    """No sentence periods, both headings in one blob: nearest-cue binding must still give the
    failing dams a 'below' verdict rather than inheriting the earlier 'Above' heading."""
    text = (
        "Answer: 4\n"
        "Above 220 m: Nurek Dam (300 m), Grande Dixence Dam (285 m), Enguri Dam (271.5 m), "
        "Vajont Dam (262 m)\n"
        "Below 220 m: Katse Dam (185 m), Karakaya Dam (158 m), Gordon Dam (140 m)\n"
    )
    r = _r(text, primary="4")
    assert t.validate_item_classification(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


# ── keystone gate ────────────────────────────────────────────────────────────────────────

def test_off_by_one_high_gates_secondaries_but_keeps_breadth():
    r = _r("Count of dams taller than 220 m:\n5\n\n" + _VERDICT_ROWS, primary="5")
    assert t.validate_keystone_count(r, _OBS)["score"] == 0.0
    assert t.validate_keystone_count(r, _OBS)["passed"] is False
    # Both GATED secondaries short-circuit to 0 with the keystone.
    assert t.validate_passing_dams(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    # Both UN-gated breadth diagnostics survive intact.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_item_classification(r, _OBS)["score"] == 1.0


def test_off_by_one_low_and_naive_all_seven_both_fail():
    for bad in ("3", "7", "0", "6"):
        r = _r(f"Count: {bad}", primary=bad)
        assert t.validate_keystone_count(r, _OBS)["passed"] is False, bad


def test_near_threshold_misclassification_costs_one_item_and_the_keystone():
    """Vajont (262 m) wrongly called below 220 m -> count 3: keystone 0, classification 6/7,
    coverage untouched at 7/7."""
    rows = _VERDICT_ROWS.replace(
        "Vajont Dam (Italy): 262 m — above 220 m",
        "Vajont Dam (Italy): 262 m — below 220 m",
    )
    r = _r("Count of dams taller than 220 m:\n3\n\n" + rows, primary="3")
    assert t.validate_keystone_count(r, _OBS)["score"] == 0.0
    assert t.validate_item_classification(r, _OBS)["score"] == 6 / 7
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


# ── grounding / breadth ──────────────────────────────────────────────────────────────────

def test_ungrounded_correct_count_earns_nothing():
    r = _r(_FULL_MULTILINE, primary="4")
    assert t.validate_keystone_count(r, _NO_VISITS)["score"] == 0.0
    assert t.validate_citations(r, _NO_VISITS)["score"] == 0.0
    assert t.validate_passing_dams(r, _NO_VISITS)["score"] == 0.0
    # Breadth credit is capped by visits, so a pure recall answer banks nothing.
    assert t.validate_coverage(r, _NO_VISITS)["score"] == 0.0
    assert t.validate_item_classification(r, _NO_VISITS)["score"] == 0.0
    assert t.validate_visits(r, _NO_VISITS)["score"] == 0.0


def test_partial_coverage_exact_fraction():
    four = "".join(_VERDICT_ROWS.splitlines(keepends=True)[:4])
    r = _r("Count of dams taller than 220 m:\n4\n\n" + four, primary="4")
    cov = t.validate_coverage(r, _OBS)
    assert cov["score"] == 4 / 7
    assert cov["passed"] is False
    assert t.validate_item_classification(r, _OBS)["score"] == 4 / 7
    # Keystone is independent of coverage and still holds.
    assert t.validate_keystone_count(r, _OBS)["score"] == 1.0


def test_bare_rows_give_coverage_but_no_classification():
    r = _r(_BARE_ROWS, primary="4")
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_item_classification(r, _OBS)["score"] == 0.0


def test_visit_gate():
    r = _r(_FULL_MULTILINE, primary="4")
    assert t.validate_visits(r, {"visit": {"count": 7}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 6}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 5}})["passed"] is False
    assert t.validate_visits(r, _NO_VISITS)["score"] == 0.0


def test_scores_are_bimodal_not_a_constant_trap():
    good = _r(_FULL_MULTILINE, primary="4")
    bad = _r("Count: 7\n\nAll seven dams are taller than 220 m.", primary="7")
    g = [f(good, _OBS)["score"] for f in t.get_validation_functions()]
    b = [f(bad, _OBS)["score"] for f in t.get_validation_functions()]
    assert sum(g) / len(g) == 1.0
    assert sum(b) / len(b) < 0.25


# ── fixtures / ground truth ──────────────────────────────────────────────────────────────

def test_fixture_margins_are_safe():
    assert len(t.ENTITIES) == 7
    assert t.KEYSTONE_COUNT == 4 and len(t.PASSING) == 4 and len(t.FAILING) == 3
    # Not trivially 0, N or N-1 -> a real discriminator.
    assert 0 < t.KEYSTONE_COUNT < len(t.ENTITIES) - 1
    for e in t.ENTITIES:
        assert (e["height"] > t.THRESHOLD) is e["passes"], e["name"]
        assert abs(e["height"] - t.THRESHOLD) >= 30, e["name"]
    # Heights are pairwise distinct -> coverage cannot cross-credit.
    assert len({e["height"] for e in t.ENTITIES}) == 7


def test_every_fixture_regex_matches_its_own_row_only():
    for e in t.ENTITIES:
        row = f"{e['name']} — {e['height']:g} m — https://en.wikipedia.org/wiki/{e['key']}"
        import re
        assert re.search(e["name_rx"], row, re.IGNORECASE), e["name"]
        assert re.search(e["value_rx"], row), e["name"]
        # No other entity's value regex may fire on this row.
        others = [o for o in t.ENTITIES if o is not e and re.search(o["value_rx"], row)]
        assert not others, (e["name"], [o["name"] for o in others])


# ── compiled plan ────────────────────────────────────────────────────────────────────────

def test_compiled_plan_is_fully_parallel_and_leaks_nothing():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)
    leaves = plan["leaves"]
    assert len(leaves) == 7
    # Genuinely independent arms: NO leaf may declare a dependency.
    assert all(not leaf.get("depends_on") for leaf in leaves)
    assert len({leaf["id"] for leaf in leaves}) == 7
    blob = " ".join(str(l) for l in leaves) + " " + plan["aggregation"]
    for leak in ("300", "285", "271", "262", "185", "158", "140"):
        assert leak not in blob, f"plan leaks height {leak!r}"
    # The count itself must appear nowhere as an asserted integer.
    assert t.KEYSTONE_COUNT not in t._int_values(blob)
    # The plan text must not satisfy the keystone gate on its own.
    assert not t._keystone_ok({"deliverables": [blob]}, _OBS)
    # Leaf ids are keyed on the GIVEN dams, never on a value or the answer.
    assert {leaf["id"] for leaf in leaves} == {f"{e['key']}_height" for e in t.ENTITIES}
