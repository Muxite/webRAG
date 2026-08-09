"""
Adversarial offline checks for codebench task c01 (nth-even-fib-sum) — no Docker, no LLM.

Mirrors the spirit of execution_compiled_*_validators_test.py: prove the task module's own
claims are internally consistent (ground truth is actually correct, keystone ids reference
real tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox.
Also exercises badmodel-lab/codebench/materialize_task.py end-to-end against this task,
since c01 is the first task that script has ever had to render — a schema mismatch here
would silently break every downstream harness script.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c01_nth_even_fib_sum as c01


def _independent_nth_even_fib_sum(n: int) -> int:
    """Reimplemented independently of task's own prose spec, to catch a hand-arithmetic
    error in the task module's embedded expected values."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b, total = 0, 1, 0
    for _ in range(n + 1):
        if a % 2 == 0:
            total += a
        a, b = b, a + b
    return total


def test_ground_truth_values_are_internally_correct():
    for n, expected in [(0, 0), (3, 2), (6, 10), (9, 44), (12, 188), (15, 798)]:
        assert _independent_nth_even_fib_sum(n) == expected, n


def test_embedded_test_file_asserts_match_ground_truth():
    content = c01.get_sandbox_fixture()[c01.VISIBLE_TEST_PATH]
    pairs = re.findall(r"nth_even_fib_sum\((-?\d+)\) == (\d+)", content)
    assert pairs, "expected at least one literal assertion in the embedded test file"
    for n_str, expected_str in pairs:
        assert _independent_nth_even_fib_sum(int(n_str)) == int(expected_str)


def test_negative_input_contract_present():
    content = c01.get_sandbox_fixture()[c01.VISIBLE_TEST_PATH]
    assert "pytest.raises(ValueError)" in content
    assert "nth_even_fib_sum(-1)" in content


def test_keystone_ids_reference_real_test_functions():
    content = c01.get_sandbox_fixture()[c01.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c01.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c01.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_degenerate_and_contract_cases():
    # n=0 is degenerate (every branch returns 0 trivially) and the ValueError contract
    # tests input handling, not the sequence logic — neither should gate the score.
    assert f"{c01.VISIBLE_TEST_PATH}::test_n0" not in c01.KEYSTONE_TEST_IDS
    assert f"{c01.VISIBLE_TEST_PATH}::test_negative_raises" not in c01.KEYSTONE_TEST_IDS


def test_visibility_is_visible():
    assert c01.get_visibility() == "visible"


def test_grading_payload_shape():
    payload = c01.get_grading_payload()
    assert payload["tests"][c01.VISIBLE_TEST_PATH] == c01.get_sandbox_fixture()[c01.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {"module": "fib_mod", "functions": ["nth_even_fib_sum"]}
    assert payload["keystone_test_ids"] == c01.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c01.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "fib_mod.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["fib_mod.py"]}
    # Must be JSON-serializable as-is: materialize_task.py writes it verbatim to
    # public/plan.json with no transformation, and the agent image never imports this
    # module at all (see agents/badmodel/Dockerfile's SECURITY comment) — a plan that
    # isn't plain JSON-safe data would silently break at materialize time.
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c01", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c01"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c01.get_task_statement()
    assert (public / "repo" / c01.VISIBLE_TEST_PATH).read_text() == c01.get_sandbox_fixture()[c01.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c01.get_compiled_plan()

    assert (private / c01.VISIBLE_TEST_PATH).read_text() == c01.get_grading_payload()["tests"][c01.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c01.VISIBLE_TEST_PATH in manifest["test_file_globs"], (
        "visible task's test path must still be manifest-dropped from the agent's own "
        "submission — grading always re-injects the canonical private/tests/ copy"
    )

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c01.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
