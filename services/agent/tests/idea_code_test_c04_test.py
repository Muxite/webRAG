"""
Adversarial offline checks for codebench task c04 (csv-to-json-mold) — no Docker, no
LLM. Mirrors idea_code_test_c01_test.py's structure: prove the task module's own claims
are internally consistent (ground truth is actually correct, keystone ids reference real
tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox, and
exercise materialize_task.py end-to-end against this task.

Because c04's assertions compare structurally-parsed JSON against dict literals (not
simple string literals), "does the embedded test file's ground truth match an
independent reimplementation" is checked by actually EXECUTING the embedded test file's
test functions against a reference `csv_to_json` (imported as a fake `csv_mold` module)
rather than regex-scraping literals — every `assert` in the canonical test file must
pass for real when run against the independently-written reference implementation.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

from agent.app.idea_code_tests import test_c04_csv_mold as c04


def _independent_csv_to_json(csv_text: str) -> str:
    """Reimplemented independently of task's own prose spec (csv.DictReader-based), to
    catch a hand-typed error in the task module's embedded expected values."""
    reader = csv.DictReader(io.StringIO(csv_text))
    records = [dict(row) for row in reader]
    return json.dumps(records, sort_keys=False)


def test_ground_truth_values_are_internally_correct():
    cases = [
        (
            "name,age,city\nAlice,30,NYC\nBob,25,LA\n",
            [
                {"name": "Alice", "age": "30", "city": "NYC"},
                {"name": "Bob", "age": "25", "city": "LA"},
            ],
        ),
        (
            'name,quote,age\n"Doe, Jane","She said ""hi""",41\nBob,plain,25\n',
            [
                {"name": "Doe, Jane", "quote": 'She said "hi"', "age": "41"},
                {"name": "Bob", "quote": "plain", "age": "25"},
            ],
        ),
        ("a,b\n1,2\n", [{"a": "1", "b": "2"}]),
    ]
    for csv_text, expected in cases:
        assert json.loads(_independent_csv_to_json(csv_text)) == expected, csv_text


def test_embedded_test_file_asserts_match_ground_truth():
    content = c04.get_sandbox_fixture()[c04.VISIBLE_TEST_PATH]

    fake_module = types.ModuleType("csv_mold")
    fake_module.csv_to_json = _independent_csv_to_json
    original = sys.modules.get("csv_mold")
    sys.modules["csv_mold"] = fake_module
    try:
        namespace: dict = {}
        exec(compile(content, "<embedded c04 test file>", "exec"), namespace)
        test_functions = [
            v for k, v in namespace.items() if k.startswith("test_") and callable(v)
        ]
        assert test_functions, "expected at least one test_ function in the embedded file"
        for fn in test_functions:
            fn()  # raises AssertionError if an embedded expected value is wrong
    finally:
        if original is not None:
            sys.modules["csv_mold"] = original
        else:
            sys.modules.pop("csv_mold", None)


def test_keystone_ids_reference_real_test_functions():
    content = c04.get_sandbox_fixture()[c04.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c04.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c04.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_bonus_format_check():
    # test_returns_valid_json_string only checks the output TYPE (str -> JSON list), not
    # any real transformation logic, so it shouldn't gate the score.
    assert f"{c04.VISIBLE_TEST_PATH}::test_returns_valid_json_string" not in c04.KEYSTONE_TEST_IDS


def test_visibility_is_visible():
    assert c04.get_visibility() == "visible"


def test_grading_payload_shape():
    payload = c04.get_grading_payload()
    assert payload["tests"][c04.VISIBLE_TEST_PATH] == c04.get_sandbox_fixture()[c04.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {"module": "csv_mold", "functions": ["csv_to_json"]}
    assert payload["keystone_test_ids"] == c04.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c04.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "csv_mold.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["csv_mold.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c04", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c04"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c04.get_task_statement()
    assert (public / "repo" / c04.VISIBLE_TEST_PATH).read_text() == c04.get_sandbox_fixture()[c04.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c04.get_compiled_plan()

    assert (private / c04.VISIBLE_TEST_PATH).read_text() == c04.get_grading_payload()["tests"][c04.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c04.VISIBLE_TEST_PATH in manifest["test_file_globs"], (
        "visible task's test path must still be manifest-dropped from the agent's own "
        "submission — grading always re-injects the canonical private/tests/ copy"
    )

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c04.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
