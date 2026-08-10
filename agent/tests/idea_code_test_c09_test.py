"""
Adversarial offline checks for codebench task c09 (interval-merge) — no Docker, no LLM.

Mirrors the spirit of execution_compiled_*_validators_test.py: prove the task module's own
claims are internally consistent (ground truth is actually correct, keystone ids reference
real tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox.
Also exercises codebench/materialize_task.py end-to-end against this task.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agent.app.idea_code_tests import test_c09_interval_merge as c09


def _independent_merge_intervals(intervals):
    """Reimplemented via a completely different algorithm (integer-point coverage sweep,
    not sort-then-sweep-intervals) so a sorting/off-by-one bug in the task's own reference
    couldn't accidentally survive into both. Only sane for the small integer ranges these
    test cases use."""
    if not intervals:
        return []
    lo = min(iv[0] for iv in intervals)
    hi = max(iv[1] for iv in intervals)
    covered = [False] * (hi - lo + 2)
    for start, end in intervals:
        for x in range(start, end + 1):
            covered[x - lo] = True
    result = []
    i, n = 0, len(covered)
    while i < n:
        if covered[i]:
            j = i
            while j < n and covered[j]:
                j += 1
            result.append([lo + i, lo + j - 1])
            i = j
        else:
            i += 1
    return result


_CASES = [
    ("no_overlap", [[1, 2], [4, 5], [7, 8]], [[1, 2], [4, 5], [7, 8]]),
    ("full_chain", [[1, 4], [2, 5], [3, 6]], [[1, 6]]),
    ("touching", [[1, 2], [2, 3]], [[1, 3]]),
    ("unsorted", [[5, 6], [1, 3], [8, 10]], [[1, 3], [5, 6], [8, 10]]),
    ("single", [[1, 1]], [[1, 1]]),
    ("empty", [], []),
]


def test_ground_truth_values_are_internally_correct():
    for name, intervals, expected in _CASES:
        assert _independent_merge_intervals(intervals) == expected, name


def test_embedded_test_file_asserts_match_ground_truth():
    content = c09.get_sandbox_fixture()[c09.VISIBLE_TEST_PATH]
    namespace = {"merge_intervals": _independent_merge_intervals}
    code = content.replace("from interval_merge import merge_intervals\n", "")
    exec(compile(code, "<c09 embedded test>", "exec"), namespace)
    test_fns = [v for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    assert len(test_fns) == 6, "expected all 6 embedded test_ functions to be present"
    for fn in test_fns:
        fn()


def test_keystone_ids_reference_real_test_functions():
    content = c09.get_sandbox_fixture()[c09.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c09.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c09.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_degenerate_cases():
    # single-interval and empty-list are degenerate (any correct passthrough handles them
    # trivially) and should not gate the score.
    assert f"{c09.VISIBLE_TEST_PATH}::test_single_interval" not in c09.KEYSTONE_TEST_IDS
    assert f"{c09.VISIBLE_TEST_PATH}::test_empty_list" not in c09.KEYSTONE_TEST_IDS


def test_visibility_is_visible():
    assert c09.get_visibility() == "visible"


def test_grading_payload_shape():
    payload = c09.get_grading_payload()
    assert payload["tests"][c09.VISIBLE_TEST_PATH] == c09.get_sandbox_fixture()[c09.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {"module": "interval_merge", "functions": ["merge_intervals"]}
    assert payload["keystone_test_ids"] == c09.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c09.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "interval_merge.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["interval_merge.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c09", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c09"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c09.get_task_statement()
    assert (public / "repo" / c09.VISIBLE_TEST_PATH).read_text() == c09.get_sandbox_fixture()[c09.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c09.get_compiled_plan()

    assert (private / c09.VISIBLE_TEST_PATH).read_text() == c09.get_grading_payload()["tests"][c09.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c09.VISIBLE_TEST_PATH in manifest["test_file_globs"], (
        "visible task's test path must still be manifest-dropped from the agent's own "
        "submission — grading always re-injects the canonical private/tests/ copy"
    )

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c09.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
