"""
Adversarial offline checks for codebench task c11 (nth-prime-sieve) — no Docker, no LLM.

Mirrors the spirit of execution_compiled_*_validators_test.py: prove the task module's own
claims are internally consistent (ground truth is actually correct, keystone ids reference
real tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox.
Also exercises badmodel-lab/codebench/materialize_task.py end-to-end against this task.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agent.app.idea_code_tests import test_c11_prime_sieve as c11


def _independent_nth_prime(n: int) -> int:
    """Reimplemented via a sieve of Eratosthenes (not trial division, which is what the task
    itself expects/permits agents to use) up to a fixed generous bound, so a shared algorithmic
    mistake couldn't slip through both implementations."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    limit = 600  # comfortably covers every n this task tests (max tested n=25 -> prime 97)
    is_composite = [False, False] + [True] * (limit - 1)
    for i in range(2, int(limit**0.5) + 1):
        if is_composite[i]:
            for j in range(i * i, limit + 1, i):
                is_composite[j] = False
    primes = [i for i, is_p in enumerate(is_composite) if is_p]
    if n > len(primes):
        raise AssertionError(f"sieve bound too small for n={n}")
    return primes[n - 1]


def test_ground_truth_values_are_internally_correct():
    for n, expected in [(1, 2), (2, 3), (6, 13), (25, 97)]:
        assert _independent_nth_prime(n) == expected, n


def test_negative_and_zero_raise():
    for n in (0, -3):
        try:
            _independent_nth_prime(n)
            raise AssertionError(f"expected ValueError for n={n}")
        except ValueError:
            pass


def test_embedded_test_file_asserts_match_ground_truth():
    content = c11.get_sandbox_fixture()[c11.VISIBLE_TEST_PATH]
    namespace = {"nth_prime": _independent_nth_prime, "pytest": pytest}
    code = content.replace("import pytest\n", "").replace(
        "from prime_sieve import nth_prime\n", ""
    )
    exec(compile(code, "<c11 embedded test>", "exec"), namespace)
    test_fns = [v for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    assert len(test_fns) == 6, "expected all 6 embedded test_ functions to be present"
    for fn in test_fns:
        fn()


def test_keystone_ids_reference_real_test_functions():
    content = c11.get_sandbox_fixture()[c11.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c11.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c11.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_contract_cases():
    # The ValueError contract tests input validation, not the prime-finding logic.
    assert f"{c11.VISIBLE_TEST_PATH}::test_zero_raises" not in c11.KEYSTONE_TEST_IDS
    assert f"{c11.VISIBLE_TEST_PATH}::test_negative_raises" not in c11.KEYSTONE_TEST_IDS


def test_visibility_is_visible():
    assert c11.get_visibility() == "visible"


def test_grading_payload_shape():
    payload = c11.get_grading_payload()
    assert payload["tests"][c11.VISIBLE_TEST_PATH] == c11.get_sandbox_fixture()[c11.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {"module": "prime_sieve", "functions": ["nth_prime"]}
    assert payload["keystone_test_ids"] == c11.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c11.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "prime_sieve.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["prime_sieve.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c11", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c11"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c11.get_task_statement()
    assert (public / "repo" / c11.VISIBLE_TEST_PATH).read_text() == c11.get_sandbox_fixture()[c11.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c11.get_compiled_plan()

    assert (private / c11.VISIBLE_TEST_PATH).read_text() == c11.get_grading_payload()["tests"][c11.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c11.VISIBLE_TEST_PATH in manifest["test_file_globs"], (
        "visible task's test path must still be manifest-dropped from the agent's own "
        "submission — grading always re-injects the canonical private/tests/ copy"
    )

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c11.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
