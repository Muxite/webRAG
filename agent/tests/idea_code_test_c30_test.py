"""
Adversarial offline checks for codebench task c30 (line-diff-patch-round-trip) — no Docker,
no LLM. Mirrors idea_code_test_c06_test.py's structural-checker pattern: c30's canonical test
file grades most non-trivial cases via an embedded `is_valid_diff`/`op_counts` checker rather
than one hardcoded op list (several distinct minimal edit scripts can exist), so this file
independently re-derives the LCS lengths/op counts, re-implements the checker separately, and
cross-checks Python's own difflib.SequenceMatcher match length as a THIRD independent source
of truth for every case, before anything reaches a live sandbox.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c30_line_diff_patch as c30


def _independent_lcs_length(old: list, new: list) -> int:
    """Bottom-up DP table, reimplemented independently of any op-emitting diff logic."""
    n, m = len(old), len(new)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if old[i] == new[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def _independent_is_valid_diff(old: list, new: list, ops: list) -> bool:
    """Reimplemented independently of the embedded test file's own is_valid_diff, to catch a
    mistake in either copy."""
    out = []
    i = 0
    for tag, line in ops:
        if tag == "equal":
            if i >= len(old) or old[i] != line:
                return False
            out.append(line)
            i += 1
        elif tag == "delete":
            if i >= len(old) or old[i] != line:
                return False
            i += 1
        elif tag == "insert":
            out.append(line)
        else:
            return False
    if i != len(old):
        return False
    return out == new


def _difflib_match_length(old: list, new: list) -> int:
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks())


_CASES = [
    ([], []),
    (["a", "b", "c"], ["a", "b", "c"]),
    (["a", "b", "c"], ["a", "b", "c", "d"]),
    (["b", "c", "d"], ["a", "b", "c", "d"]),
    ([], ["a", "b", "c"]),
    (["a", "b", "c"], []),
    (["a", "b", "c", "d", "e"], ["a", "x", "c", "d", "f"]),
    (["a", "b"], ["x", "y"]),
    (["a", "a", "b"], ["a", "b", "a"]),
    (["line1", "line2", "line3", "line4", "line5", "line6"],
     ["line1", "lineX", "line3", "lineY", "line5", "line6", "line7"]),
    (["alpha", "beta", "gamma"], ["gamma", "beta", "alpha"]),
]


def test_lcs_length_agrees_with_difflib_for_every_case():
    # Two structurally different sources of truth (a from-scratch DP table vs. Python's own
    # stdlib SequenceMatcher) must agree on every case before any expected value is trusted.
    for old, new in _CASES:
        assert _independent_lcs_length(old, new) == _difflib_match_length(old, new), (old, new)


def test_pinned_exact_op_lists_match_the_true_lcs_length():
    # For the five cases the canonical test file pins an exact op list, the equal-op count in
    # that pinned list must equal the true LCS length (a wrong pin would silently under- or
    # over-credit every submission).
    pinned = {
        "identical": (["a", "b", "c"], ["a", "b", "c"],
                      [["equal", "a"], ["equal", "b"], ["equal", "c"]]),
        "append": (["a", "b", "c"], ["a", "b", "c", "d"],
                   [["equal", "a"], ["equal", "b"], ["equal", "c"], ["insert", "d"]]),
        "prepend": (["b", "c", "d"], ["a", "b", "c", "d"],
                    [["insert", "a"], ["equal", "b"], ["equal", "c"], ["equal", "d"]]),
        "insert_into_empty": ([], ["a", "b", "c"],
                               [["insert", "a"], ["insert", "b"], ["insert", "c"]]),
        "delete_all": (["a", "b", "c"], [],
                        [["delete", "a"], ["delete", "b"], ["delete", "c"]]),
    }
    for name, (old, new, ops) in pinned.items():
        equal_count = sum(1 for tag, _ in ops if tag == "equal")
        assert equal_count == _independent_lcs_length(old, new), name
        assert _independent_is_valid_diff(old, new, ops), name


def test_pinned_op_counts_match_the_true_lcs_length():
    # For the three cases graded via op_counts() instead of an exact list, the (equal, delete,
    # insert) tuple embedded in the canonical test file must match independently-derived counts.
    expectations = {
        "middle_change": ((["a", "b", "c", "d", "e"], ["a", "x", "c", "d", "f"]), (3, 2, 2)),
        "complete_replacement": ((["a", "b"], ["x", "y"]), (0, 2, 2)),
        "duplicate_lines": ((["a", "a", "b"], ["a", "b", "a"]), (2, 1, 1)),
    }
    for name, ((old, new), expected_counts) in expectations.items():
        lcs_len = _independent_lcs_length(old, new)
        equal, delete, insert = expected_counts
        assert equal == lcs_len, name
        assert delete == len(old) - lcs_len, name
        assert insert == len(new) - lcs_len, name


def test_is_valid_diff_checker_rejects_a_wrong_reconstruction():
    # Sanity-check the checker itself (both the embedded copy's logic, re-derived here, and
    # this file's independent copy): ops that reconstruct the wrong output must be rejected,
    # and ops that claim 'equal'/'delete' against a mismatched old line must be rejected.
    old, new = ["a", "b", "c"], ["a", "z", "c"]
    right_ops = [["equal", "a"], ["delete", "b"], ["insert", "z"], ["equal", "c"]]
    assert _independent_is_valid_diff(old, new, right_ops)

    wrong_reconstruction = [["equal", "a"], ["equal", "b"], ["equal", "c"]]  # reconstructs old, not new
    assert not _independent_is_valid_diff(old, new, wrong_reconstruction)

    lying_equal = [["equal", "a"], ["equal", "z"], ["equal", "c"]]  # old[1] is "b", not "z"
    assert not _independent_is_valid_diff(old, new, lying_equal)

    incomplete = [["equal", "a"]]  # never consumes "b" or "c"
    assert not _independent_is_valid_diff(old, new, incomplete)


def test_apply_patch_direct_example_is_correct():
    old = ["a", "b", "c"]
    ops = [["equal", "a"], ["delete", "b"], ["insert", "x"], ["equal", "c"]]
    assert _independent_is_valid_diff(old, ["a", "x", "c"], ops)


def test_embedded_test_file_defines_and_uses_the_structural_checker():
    content = c30.get_grading_payload()["tests"][c30.CANONICAL_TEST_PATH]
    assert "def is_valid_diff(old: list, new: list, ops: list) -> bool:" in content
    assert "def op_counts(ops: list) -> tuple:" in content
    assert content.count("is_valid_diff(") >= 3  # used by at least 3 test functions, not just defined
    assert content.count("op_counts(") >= 3


def test_embedded_test_file_covers_the_two_apply_patch_error_contracts():
    content = c30.get_grading_payload()["tests"][c30.CANONICAL_TEST_PATH]
    assert content.count("pytest.raises(ValueError)") == 2


def test_keystone_ids_reference_real_test_functions():
    content = c30.get_grading_payload()["tests"][c30.CANONICAL_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c30.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c30.CANONICAL_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_the_degenerate_and_bonus_cases():
    non_keystone = [
        "test_round_trip_empty_to_empty",
        "test_insert_into_empty_old",
        "test_delete_all_from_old",
        "test_apply_patch_rejects_overrunning_patch",
        "test_round_trip_on_reordered_lines",
    ]
    for name in non_keystone:
        assert f"{c30.CANONICAL_TEST_PATH}::{name}" not in c30.KEYSTONE_TEST_IDS, name
    # and every keystone id really is present in the embedded file
    content = c30.get_grading_payload()["tests"][c30.CANONICAL_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert len(c30.KEYSTONE_TEST_IDS) == 9
    assert len(defined) == 14


def test_visibility_is_hidden():
    assert c30.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c30.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c30.get_grading_payload()
    assert payload["tests"] == {c30.CANONICAL_TEST_PATH: c30._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "line_diff", "functions": ["diff_lines", "apply_patch"],
    }
    assert payload["keystone_test_ids"] == c30.KEYSTONE_TEST_IDS


def test_task_statement_worked_examples_match_ground_truth():
    statement = c30.get_task_statement()
    assert _independent_lcs_length(["a", "b", "c"], ["a", "b", "c", "d"]) == 3
    assert '[["equal","a"],["equal","b"],["equal","c"],["insert","d"]]' in statement
    assert _independent_lcs_length(["b", "c", "d"], ["a", "b", "c", "d"]) == 3
    assert '[["insert","a"],["equal","b"],["equal","c"],["equal","d"]]' in statement
    assert _independent_lcs_length(["a", "b"], ["x", "y"]) == 0


def test_compiled_plan_structure():
    plan = c30.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "line_diff.py" in leaf["instruction"]
    assert "ValueError" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["line_diff.py"]}
    json.dumps(plan)


def test_compiled_plan_does_not_leak_canonical_test_content():
    # SECURITY: the plan is exposed to the agent sandbox verbatim (public/plan.json) — it must
    # never contain the canonical test file's literal source or any of the private grading
    # module's helper function bodies.
    plan = c30.get_compiled_plan()
    plan_text = json.dumps(plan)
    assert "def is_valid_diff" not in plan_text
    assert "def op_counts" not in plan_text
    assert "import pytest" not in plan_text


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c30", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c30"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c30.get_task_statement()
    assert list((public / "repo").rglob("*")) == []
    assert json.loads((public / "plan.json").read_text()) == c30.get_compiled_plan()

    assert (private / c30.CANONICAL_TEST_PATH).read_text() == c30.get_grading_payload()["tests"][c30.CANONICAL_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert manifest["test_file_globs"] == [c30.CANONICAL_TEST_PATH]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c30.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
