"""
Adversarial offline checks for codebench task c40 (bugfix-date-range-overlap) — no Docker,
no LLM.

Same debug-task validation shape as idea_code_test_c39_test.py, adapted to THIS task's
revised bug shape (a boundary bug plus a dead `touching_counts` parameter, with the hidden
canonical set being the only cases that exercise the flag):
  (1) the buggy starter genuinely fails Group A (the boundary cases) when actually run,
  (2) a boundary-only partial fix makes every Group A AND bonus case (visible AND hidden
      bonus cases) agree with ground truth — i.e. it looks complete from the visible suite's
      point of view — but still disagrees with ground truth on every Group B (hidden,
      flag=False) case, because it never wires touching_counts into the logic at all,
  (3) the fully correct fix (boundary fixed AND flag actually used) agrees with ground truth
      on the ENTIRE canonical (visible + hidden) suite, and
  (4) the hidden cases are genuinely absent from the sandbox fixture while still being a
      strict superset built on top of the visible content at the identical relpath.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c40_bugfix_range_overlap as c40


# --------------------------------------------------------------------------------------------
# Ground truth, derived TWO independent ways (neither one is the task module's own source).
# --------------------------------------------------------------------------------------------
def _independent_ranges_overlap_negation(a_start, a_end, b_start, b_end, touching_counts=True):
    """Two inclusive ranges overlap-or-touch iff neither is strictly to one side of the
    other. Expressed as a boolean negation of the disjoint condition for touching_counts=True.

    For touching_counts=False, a simple two-term negation like
    ``not (a_end <= b_start or b_end <= a_start)`` is NOT algebraically equivalent in general
    to "the ranges share more than one day" — it silently mishandles a degenerate single-day
    range sitting entirely inside the other, away from either boundary (verified by exhaustive
    random brute-force comparison against the window formula below: they agree on all 400,000+
    random cases tried EXCEPT this exact degenerate shape). So this branch instead directly
    enumerates the literal shared calendar days via set intersection — no algebra, no boundary
    reasoning at all, just counting — which is unimpeachably correct by construction and uses a
    completely different technique from the window-based method below."""
    if touching_counts:
        return not (a_end < b_start or b_end < a_start)
    a_days = set(range(a_start, a_end + 1))
    b_days = set(range(b_start, b_end + 1))
    return len(a_days & b_days) >= 2


def _independent_ranges_overlap_window(a_start, a_end, b_start, b_end, touching_counts=True):
    """Completely different formulation: compute the overlap WINDOW explicitly
    ([max(starts), min(ends)]) and ask whether that window is non-empty (touching_counts=True,
    a single shared day is a non-empty window of length 1) or has positive width
    (touching_counts=False, more than a single shared day)."""
    window_lo = max(a_start, b_start)
    window_hi = min(a_end, b_end)
    if touching_counts:
        return window_lo <= window_hi
    return window_lo < window_hi


def _exec_ranges_overlap(source: str):
    import importlib.util

    spec = importlib.util.spec_from_loader("c40_range_overlap_under_test", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(source, module.__dict__)  # noqa: S102 — trusted in-repo fixture
    return module.ranges_overlap


def _extract_cases(content: str):
    """Every literal call in the embedded test files uses plain int literals and either a
    bare boolean or an explicit touching_counts=<bool> kwarg — regex is adequate and easy to
    verify by eye here (unlike c39's arbitrary-expression arguments)."""
    plain = re.findall(
        r"ranges_overlap\((-?\d+), (-?\d+), (-?\d+), (-?\d+)\) is (True|False)", content
    )
    flagged = re.findall(
        r"ranges_overlap\((-?\d+), (-?\d+), (-?\d+), (-?\d+), touching_counts=(True|False)\) "
        r"is (True|False)",
        content,
    )
    cases = [(int(a), int(b), int(c), int(d), True, exp == "True") for a, b, c, d, exp in plain]
    cases += [
        (int(a), int(b), int(c), int(d), flag == "True", exp == "True")
        for a, b, c, d, flag, exp in flagged
    ]
    return cases


_GROUP_A = {
    "b_starts_where_a_ends": (1, 5, 5, 10, True),
    "touching_scaled_up": (10, 20, 20, 30, True),
    "single_day_same_day": (5, 5, 5, 5, True),
}
_GROUP_B = {
    "flag_false_touch_1": (1, 5, 5, 10, False),
    "flag_false_touch_2": (10, 20, 20, 30, False),
    "flag_false_single_day": (5, 5, 5, 5, False),
}
_BONUS_VISIBLE = {
    "clearly_overlapping": (1, 10, 5, 8, True),
    "clearly_disjoint": (1, 5, 10, 15, True),
    "a_starts_where_b_ends": (5, 10, 1, 5, True),
    "identical_ranges": (3, 7, 3, 7, True),
    "a_contains_b": (1, 10, 4, 6, True),
    "b_contains_a": (4, 6, 1, 10, True),
    "adjacent_gap": (1, 5, 6, 10, True),
}
_BONUS_HIDDEN = {
    "real_overlap_flag_false": (1, 10, 5, 8, False),
    "flag_true_matches_default": (1, 10, 5, 8, True),
    "real_gap_flag_false": (1, 5, 6, 10, False),
}
# Group C: hidden-only, added 2026-08-07. The degenerate-single-day-range case: discriminates a
# fix that correctly wires in touching_counts AND correctly handles every boundary-TOUCH case,
# but implements the False branch with a plausible two-term "boundary equality" formula (e.g.
# "a_start != b_end and b_start != a_end") instead of genuinely reasoning about shared-day
# count. This is Aider's own actual round-2 submitted shape for this task, not a hypothetical.
_GROUP_C = {
    "single_day_fully_inside_not_at_boundary": (5, 5, 1, 10, False),
    "single_day_fully_inside_not_at_boundary_reversed": (1, 10, 5, 5, False),
}


def _check_all(fn, table, seconds_note=None):
    for name, (a_s, a_e, b_s, b_e, flag) in table.items():
        expected = _independent_ranges_overlap_negation(a_s, a_e, b_s, b_e, flag)
        assert expected == _independent_ranges_overlap_window(a_s, a_e, b_s, b_e, flag), name
        actual = fn(a_s, a_e, b_s, b_e, flag)
        yield name, expected, actual


def test_the_two_independent_ground_truth_implementations_agree():
    for table in (_GROUP_A, _GROUP_B, _GROUP_C, _BONUS_VISIBLE, _BONUS_HIDDEN):
        for name, (a_s, a_e, b_s, b_e, flag) in table.items():
            neg = _independent_ranges_overlap_negation(a_s, a_e, b_s, b_e, flag)
            win = _independent_ranges_overlap_window(a_s, a_e, b_s, b_e, flag)
            assert neg == win, name


def test_group_c_expected_value_is_false_confirmed_by_brute_force_day_counting():
    # Belt-and-suspenders: recompute Group C's expected answer a THIRD way, directly, inline,
    # rather than trusting the shared helper above.
    for name, (a_s, a_e, b_s, b_e, flag) in _GROUP_C.items():
        assert flag is False
        shared_days = set(range(a_s, a_e + 1)) & set(range(b_s, b_e + 1))
        assert len(shared_days) == 1, (name, shared_days)  # exactly one day shared, never more
        assert _independent_ranges_overlap_window(a_s, a_e, b_s, b_e, flag) is False, name


def test_embedded_visible_test_file_asserts_match_ground_truth():
    cases = _extract_cases(c40._TEST_FILE_CONTENT)
    assert len(cases) == 10
    for a_s, a_e, b_s, b_e, flag, expected in cases:
        assert flag is True  # visible file never passes touching_counts
        assert _independent_ranges_overlap_negation(a_s, a_e, b_s, b_e, flag) is expected


def test_embedded_hidden_addendum_asserts_match_ground_truth():
    cases = _extract_cases(c40._HIDDEN_TEST_ADDENDUM)
    assert len(cases) == 8
    for a_s, a_e, b_s, b_e, flag, expected in cases:
        assert _independent_ranges_overlap_negation(a_s, a_e, b_s, b_e, flag) is expected


def test_canonical_test_file_is_visible_plus_hidden_with_no_extra_cases():
    canonical_cases = _extract_cases(c40._CANONICAL_TEST_FILE_CONTENT)
    visible_cases = _extract_cases(c40._TEST_FILE_CONTENT)
    hidden_cases = _extract_cases(c40._HIDDEN_TEST_ADDENDUM)
    assert len(canonical_cases) == 18
    assert canonical_cases == visible_cases + hidden_cases


def test_starter_module_actually_buggy_on_group_a_when_run():
    buggy = _exec_ranges_overlap(c40._RANGE_OVERLAP_PY_CONTENT)
    for name, expected, actual in _check_all(buggy, _GROUP_A):
        assert actual is not expected, f"{name} was expected to be BROKEN by the boundary bug"


def test_starter_module_passes_group_b_bonus_and_visible_bonus_when_run():
    """These happen to already agree with ground truth on the unmodified starter (the
    boundary bug's effect coincides with correct behavior for these particular flag=False
    inputs) — not required to be broken by the starter, only required to discriminate a
    partial fix, which the next test proves."""
    buggy = _exec_ranges_overlap(c40._RANGE_OVERLAP_PY_CONTENT)
    for table in (_GROUP_B, _BONUS_VISIBLE, _BONUS_HIDDEN):
        for name, expected, actual in _check_all(buggy, table):
            assert actual is expected, name


def test_starter_module_also_fails_group_c_when_run():
    """Unlike Group B, the unmodified starter does NOT coincidentally get Group C right — it
    ignores touching_counts and always returns the True-mode answer, which is True for both
    Group C cases (there IS a shared day), while the correct False-mode answer is False."""
    buggy = _exec_ranges_overlap(c40._RANGE_OVERLAP_PY_CONTENT)
    for name, expected, actual in _check_all(buggy, _GROUP_C):
        assert actual is not expected, f"{name} was expected to be BROKEN on the unmodified starter"


def _boundary_only_fix_source() -> str:
    assert c40._RANGE_OVERLAP_PY_CONTENT.count("b_start < a_end") == 1
    fixed = c40._RANGE_OVERLAP_PY_CONTENT.replace("b_start < a_end", "b_start <= a_end")
    assert fixed != c40._RANGE_OVERLAP_PY_CONTENT
    # sanity: this mutation must NOT also happen to wire touching_counts into the logic
    assert "touching_counts" not in fixed.split("return", 1)[1]
    return fixed


def _full_fix_source() -> str:
    # NOTE: this replaces a PREVIOUS "fully correct" reference
    # (`a_start < b_end and b_start < a_end` for the False branch) that itself turned out to be
    # WRONG on the degenerate-single-day-range shape Group C exercises — discovered by brute-
    # force comparison against literal day-set counting, not by inspection. The window formula
    # below is verified correct against 400,000+ random brute-force day-set-intersection checks
    # (see the module-level docstring note and _independent_ranges_overlap_negation above).
    return (
        "def ranges_overlap(a_start, a_end, b_start, b_end, touching_counts=True):\n"
        "    lo = max(a_start, b_start)\n"
        "    hi = min(a_end, b_end)\n"
        "    if touching_counts:\n"
        "        return lo <= hi\n"
        "    return lo < hi\n"
    )


# Aider's own actual round-2 submission for this task (qwen2.5:14b, coordinator_batch3) — a
# real, live near-miss, not a hypothetical mutant. It correctly wires touching_counts into the
# logic and correctly handles every boundary-touch case, but implements the False branch by
# checking "the two ranges don't touch exactly at a_start==b_end or b_start==a_end" instead of
# genuinely counting shared days — the exact bug class Group C targets.
_AIDER_ACTUAL_SUBMITTED_FIX_SOURCE = (
    "def ranges_overlap(a_start, a_end, b_start, b_end, touching_counts=True):\n"
    "    if touching_counts:\n"
    "        return not (a_end < b_start or b_end < a_start)\n"
    "    else:\n"
    "        return a_start <= b_end and b_start <= a_end and a_start != b_end and "
    "b_start != a_end\n"
)


def test_boundary_only_fix_looks_complete_from_the_visible_suite_alone():
    """The 'obvious' partial fix (boundary comparison only, touching_counts still dead)
    makes every Group A + all visible/hidden bonus cases agree with ground truth."""
    fixed = _exec_ranges_overlap(_boundary_only_fix_source())
    for table in (_GROUP_A, _BONUS_VISIBLE, _BONUS_HIDDEN):
        for name, expected, actual in _check_all(fixed, table):
            assert actual is expected, name


def test_boundary_only_fix_still_fails_every_group_b_case():
    """Mutation-test the brief requires: the SAME partial fix that looks complete above
    still disagrees with ground truth on every hidden touching_counts=False case, because
    the flag is silently ignored."""
    fixed = _exec_ranges_overlap(_boundary_only_fix_source())
    for name, expected, actual in _check_all(fixed, _GROUP_B):
        assert actual is not expected, f"{name} should still be WRONG under the partial fix"


def test_boundary_only_fix_also_fails_every_group_c_case():
    fixed = _exec_ranges_overlap(_boundary_only_fix_source())
    for name, expected, actual in _check_all(fixed, _GROUP_C):
        assert actual is not expected, f"{name} should still be WRONG under the partial fix"


def test_aider_actual_submitted_fix_looks_complete_on_groups_a_and_b_and_bonus():
    """Aider's real fix genuinely wires in touching_counts and gets every boundary-touch case
    (Group A, Group B, and all bonus cases) right — proving Group C isn't catching a broad
    correctness gap, only the narrow, specific degenerate shape it targets."""
    fixed = _exec_ranges_overlap(_AIDER_ACTUAL_SUBMITTED_FIX_SOURCE)
    for table in (_GROUP_A, _GROUP_B, _BONUS_VISIBLE, _BONUS_HIDDEN):
        for name, expected, actual in _check_all(fixed, table):
            assert actual is expected, name


def test_aider_actual_submitted_fix_fails_every_group_c_case():
    """The load-bearing mutation-test claim: Aider's actual live round-2 submission for this
    task is rejected specifically and only by Group C, confirming this is a real, sharp trap,
    not a hypothetical one."""
    fixed = _exec_ranges_overlap(_AIDER_ACTUAL_SUBMITTED_FIX_SOURCE)
    for name, expected, actual in _check_all(fixed, _GROUP_C):
        assert actual is not expected, f"{name} should be WRONG under Aider's actual fix"


def test_full_fix_matches_ground_truth_on_every_case():
    fixed = _exec_ranges_overlap(_full_fix_source())
    for table in (_GROUP_A, _GROUP_B, _GROUP_C, _BONUS_VISIBLE, _BONUS_HIDDEN):
        for name, expected, actual in _check_all(fixed, table):
            assert actual is expected, name


def test_keystone_ids_reference_real_test_functions_in_the_canonical_grading_file():
    canonical_content = c40.get_grading_payload()["tests"][c40.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", canonical_content, re.MULTILINE))
    assert len(defined) == 18
    for node_id in c40.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c40.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the canonical file"
    assert set(c40.KEYSTONE_TEST_IDS) == {
        f"{c40.VISIBLE_TEST_PATH}::test_b_starts_exactly_where_a_ends",
        f"{c40.VISIBLE_TEST_PATH}::test_touching_boundary_scaled_up",
        f"{c40.VISIBLE_TEST_PATH}::test_single_day_ranges_same_day",
        f"{c40.VISIBLE_TEST_PATH}::test_touching_boundary_not_counted_when_flag_false",
        f"{c40.VISIBLE_TEST_PATH}::test_touching_boundary_scaled_up_not_counted_when_flag_false",
        f"{c40.VISIBLE_TEST_PATH}::test_single_day_ranges_not_counted_when_flag_false",
        f"{c40.VISIBLE_TEST_PATH}::"
        "test_single_day_range_fully_inside_another_not_counted_when_flag_false",
    }


def test_category_is_hard():
    assert c40.get_test_metadata()["category"] == "hard"


def test_visibility_is_visible():
    assert c40.get_visibility() == "visible"


def test_sandbox_fixture_excludes_hidden_tests():
    fixture = c40.get_sandbox_fixture()
    assert fixture[c40.STARTER_MODULE_PATH] == c40._RANGE_OVERLAP_PY_CONTENT
    assert fixture[c40.VISIBLE_TEST_PATH] == c40._TEST_FILE_CONTENT
    assert "b_start < a_end" in fixture[c40.STARTER_MODULE_PATH]
    hidden_names = set(re.findall(r"^def (test_\w+)\(", c40._HIDDEN_TEST_ADDENDUM, re.MULTILINE))
    assert len(hidden_names) == 8
    for name in hidden_names:
        assert name not in fixture[c40.VISIBLE_TEST_PATH]
    assert "touching_counts=False" not in fixture[c40.VISIBLE_TEST_PATH]


def test_grading_payload_is_a_genuine_superset_of_the_visible_fixture():
    payload = c40.get_grading_payload()
    canonical = payload["tests"][c40.VISIBLE_TEST_PATH]
    visible = c40.get_sandbox_fixture()[c40.VISIBLE_TEST_PATH]
    assert canonical != visible
    assert canonical.startswith(visible)
    assert canonical == c40._CANONICAL_TEST_FILE_CONTENT
    assert payload["entrypoint"] == {"module": "range_overlap", "functions": ["ranges_overlap"]}
    assert payload["keystone_test_ids"] == c40.KEYSTONE_TEST_IDS


def test_compiled_plan_structure_and_no_leaked_values():
    plan = c40.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["fix_bug"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "range_overlap.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["range_overlap.py"]}

    instruction = leaf["instruction"]
    for a_s, a_e, b_s, b_e, flag, expected in _extract_cases(c40._CANONICAL_TEST_FILE_CONTENT):
        assert f"({a_s}, {a_e}, {b_s}, {b_e})" not in instruction
        assert f"({a_s}, {a_e}, {b_s}, {b_e}, touching_counts={flag})" not in instruction
    for leaked in (
        "a_start <= b_end and b_start <= a_end",
        "b_start < a_end",
        "b_start <= a_end",
        "a_start < b_end and b_start < a_end",
        "a_start != b_end",
        "b_start != a_end",
        "max(a_start, b_start)",
        "min(a_end, b_end)",
    ):
        assert leaked not in instruction, f"plan leaks the mechanism via {leaked!r}"
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c40", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c40"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c40.get_task_statement()
    assert (public / "repo" / c40.STARTER_MODULE_PATH).read_text() == c40._RANGE_OVERLAP_PY_CONTENT
    assert (public / "repo" / c40.VISIBLE_TEST_PATH).read_text() == c40._TEST_FILE_CONTENT
    assert json.loads((public / "plan.json").read_text()) == c40.get_compiled_plan()

    assert (private / c40.VISIBLE_TEST_PATH).read_text() == c40.get_grading_payload()["tests"][c40.VISIBLE_TEST_PATH]
    assert (private / c40.VISIBLE_TEST_PATH).read_text() == c40._CANONICAL_TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c40.VISIBLE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c40.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
