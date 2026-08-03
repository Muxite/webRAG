"""
Adversarial offline checks for codebench task c03 (balanced-brackets-validator) — no
Docker, no LLM. Mirrors idea_code_test_c01_test.py's structure: prove the task module's
own claims are internally consistent (ground truth is actually correct, keystone ids
reference real tests, the compiled plan is well-formed) BEFORE anything ever reaches a
live sandbox, and exercise materialize_task.py end-to-end against this task. c03 is
hidden, so get_sandbox_fixture() is empty — the canonical test content lives only in
get_grading_payload()["tests"].
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c03_balanced_brackets as c03


def _independent_is_balanced(s: str) -> bool:
    """Reimplemented independently of task's own prose spec (index-based loop mapping
    openers directly to their expected closer), to catch a hand-reasoning error in the
    task module's embedded expected values."""
    open_to_close = {"(": ")", "[": "]", "{": "}"}
    close_set = {")", "]", "}"}
    stack = []
    for ch in s:
        if ch in open_to_close:
            stack.append(open_to_close[ch])
        elif ch in close_set:
            if not stack:
                return False
            expected = stack.pop()
            if expected != ch:
                return False
    return len(stack) == 0


def test_ground_truth_values_are_internally_correct():
    cases = [
        ("", True),
        ("()[]{}", True),
        ("())", False),
        ("((", False),
        ("([)]", False),
        ("a(b[c]d)e", True),
        ("[{(})]", False),
        ("(((((((())))))))", True),
    ]
    for s, expected in cases:
        assert _independent_is_balanced(s) is expected, s


def test_embedded_test_file_asserts_match_ground_truth():
    content = c03.get_grading_payload()["tests"][c03.CANONICAL_TEST_PATH]
    pairs = re.findall(r'is_balanced\("([^"]*)"\) is (True|False)', content)
    assert pairs, "expected at least one literal is_balanced() assertion"
    for s, expected_str in pairs:
        expected = expected_str == "True"
        assert _independent_is_balanced(s) is expected, (s, expected)


def test_spec_worked_examples_match_ground_truth():
    # get_task_statement() gives the agent worked examples inline since this task is
    # hidden — those examples must themselves be correct, or the agent is misled.
    statement = c03.get_task_statement()
    assert _independent_is_balanced("a(b)c") is True
    assert '"a(b)c")` -> `True`' in statement
    assert _independent_is_balanced("([)]") is False
    assert '"([)]")` -> `False`' in statement
    assert _independent_is_balanced("{[()()]}") is True
    assert '"{[()()]}")` -> `True`' in statement


def test_keystone_ids_reference_real_test_functions():
    content = c03.get_grading_payload()["tests"][c03.CANONICAL_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c03.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c03.CANONICAL_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_degenerate_empty_case():
    # The empty string is trivially True under any reasonable implementation (including
    # a completely broken one that never pushes/pops) so it shouldn't gate the score.
    assert f"{c03.CANONICAL_TEST_PATH}::test_empty_string" not in c03.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c03.get_visibility() == "hidden"


def test_sandbox_fixture_is_empty_for_hidden_task():
    assert c03.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c03.get_grading_payload()
    assert payload["tests"][c03.CANONICAL_TEST_PATH] == c03._TEST_FILE_CONTENT
    assert payload["entrypoint"] == {"module": "brackets", "functions": ["is_balanced"]}
    assert payload["keystone_test_ids"] == c03.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c03.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "brackets.py" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["brackets.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c03", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c03"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c03.get_task_statement()
    # hidden task: no test file (or any other starter file) leaked into public/repo
    assert list((public / "repo").rglob("*")) == []
    assert json.loads((public / "plan.json").read_text()) == c03.get_compiled_plan()

    assert (private / c03.CANONICAL_TEST_PATH).read_text() == c03.get_grading_payload()["tests"][c03.CANONICAL_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert manifest["test_file_globs"] == [c03.CANONICAL_TEST_PATH]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c03.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
