"""
Adversarial offline checks for codebench task c12 (json-schema-diff-tool) — no Docker, no LLM.

Mirrors the spirit of execution_compiled_*_validators_test.py: prove the task module's own
claims are internally consistent (ground truth is actually correct, keystone ids reference
real tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox.
Also exercises badmodel-lab/codebench/materialize_task.py end-to-end against this task. c12 is
hidden, so (unlike c01/c09/c11) the canonical test content lives ONLY in
get_grading_payload(), never in get_sandbox_fixture().
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c12_json_diff as c12


def _independent_diff_keys(old: dict, new: dict) -> dict:
    """Reimplemented via set algebra instead of list-comprehension membership checks, to
    reduce the odds of a copy-correlated bug against the task's own reference."""
    old_keys, new_keys = set(old), set(new)
    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    changed = sorted(k for k in (old_keys & new_keys) if old[k] != new[k])
    return {"added": added, "removed": removed, "changed": changed}


_CASES = [
    ("worked_example", {"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 20, "d": 4},
     {"added": ["d"], "removed": ["c"], "changed": ["b"]}),
    ("pure_addition", {"a": 1}, {"a": 1, "b": 2}, {"added": ["b"], "removed": [], "changed": []}),
    ("pure_removal", {"a": 1, "b": 2}, {"a": 1}, {"added": [], "removed": ["b"], "changed": []}),
    ("pure_change", {"a": 1}, {"a": 2}, {"added": [], "removed": [], "changed": ["a"]}),
    ("no_difference", {"a": 1, "b": 2}, {"a": 1, "b": 2}, {"added": [], "removed": [], "changed": []}),
    ("list_value_change", {"a": [1, 2]}, {"a": [1, 3]}, {"added": [], "removed": [], "changed": ["a"]}),
    ("int_float_equal", {"a": 1}, {"a": 1.0}, {"added": [], "removed": [], "changed": []}),
]


def test_ground_truth_values_are_internally_correct():
    for name, old, new, expected in _CASES:
        assert _independent_diff_keys(old, new) == expected, name


def test_task_statement_worked_example_matches_ground_truth():
    statement = c12.get_task_statement()
    assert '{"added": ["d"], "removed": ["c"], "changed": ["b"]}' in statement
    _, old, new, expected = _CASES[0]
    assert _independent_diff_keys(old, new) == expected


def test_embedded_test_file_asserts_match_ground_truth():
    content = c12.get_grading_payload()["tests"][c12.VISIBLE_TEST_PATH]
    namespace = {"diff_keys": _independent_diff_keys}
    code = content.replace("from json_diff import diff_keys\n", "")
    exec(compile(code, "<c12 embedded test>", "exec"), namespace)
    test_fns = [v for k, v in namespace.items() if k.startswith("test_") and callable(v)]
    assert len(test_fns) == 7, "expected all 7 embedded test_ functions to be present"
    for fn in test_fns:
        fn()


def test_keystone_ids_reference_real_test_functions():
    content = c12.get_grading_payload()["tests"][c12.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c12.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c12.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_int_float_edge_case():
    # int-vs-float equality is a genuinely defensible edge case to get "wrong" (e.g. via a
    # type(...) != type(...) check) — bonus credit only, not keystone.
    assert (
        f"{c12.VISIBLE_TEST_PATH}::test_int_and_float_equal_value_omitted"
        not in c12.KEYSTONE_TEST_IDS
    )


def test_visibility_is_hidden():
    assert c12.get_visibility() == "hidden"


def test_sandbox_fixture_has_no_test_file_for_hidden_task():
    assert c12.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c12.get_grading_payload()
    assert payload["entrypoint"] == {"module": "json_diff", "functions": ["diff_keys"]}
    assert payload["keystone_test_ids"] == c12.KEYSTONE_TEST_IDS
    assert c12.VISIBLE_TEST_PATH in payload["tests"]


def test_compiled_plan_structure():
    plan = c12.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "json_diff.py" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["json_diff.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c12", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c12"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c12.get_task_statement()
    assert not (public / "repo" / c12.VISIBLE_TEST_PATH).exists()
    assert json.loads((public / "plan.json").read_text()) == c12.get_compiled_plan()

    assert (private / c12.VISIBLE_TEST_PATH).read_text() == c12.get_grading_payload()["tests"][c12.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c12.VISIBLE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c12.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
