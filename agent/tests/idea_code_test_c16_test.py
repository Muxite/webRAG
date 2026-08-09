"""
Adversarial offline checks for codebench task c16 (build-a-todo-api) — no Docker, no LLM.

Same shape as idea_code_test_c15_test.py: soft/hidden task, so these checks only verify the
module's own internal consistency and exercise materialize_task.py end-to-end (including
rubric.json). The one c16-specific wrinkle worth checking: the spec text must actually contain
the concrete fallback API names (add/list_todos/mark_done/remove) that the smoke test hardcodes
an import of — otherwise a hidden-task agent that skips the fallback shape would have no way to
discover it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c16_todo_api as c16


def test_metadata_is_soft():
    meta = c16.get_test_metadata()
    assert meta["test_id"] == "c16"
    assert meta["category"] == "soft"


def test_visibility_is_hidden():
    assert c16.get_visibility() == "hidden"


def test_sandbox_fixture_is_empty_for_hidden_task():
    assert c16.get_sandbox_fixture() == {}


def test_task_statement_covers_key_requirements():
    statement = c16.get_task_statement()
    assert "todo_api.py" in statement
    assert "in-memory" in statement.lower() or "in memory" in statement.lower()
    # the concrete fallback API the smoke test targets must be spelled out in the spec
    for name in ("add(text)", "list_todos()", "mark_done(id)", "remove(id)"):
        assert name in statement, f"fallback API name {name!r} missing from task statement"


def test_smoke_test_file_imports_the_fallback_api():
    content = c16.get_grading_payload()["tests"][c16.SMOKE_TEST_PATH]
    assert "from todo_api import add, list_todos, mark_done, remove" in content
    assert "def test_add_returns_id_like_value" in content
    assert "def test_list_reflects_addition" in content
    assert "def test_mark_done_does_not_crash" in content
    assert "def test_remove_removes_it" in content


def test_keystone_ids_reference_real_test_functions():
    content = c16.get_grading_payload()["tests"][c16.SMOKE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert c16.KEYSTONE_TEST_IDS, "c16 defines a non-empty keystone list (all smoke checks gate)"
    for node_id in c16.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c16.SMOKE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_grading_payload_shape():
    payload = c16.get_grading_payload()
    assert payload["tests"][c16.SMOKE_TEST_PATH] == c16._TEST_FILE_CONTENT
    assert payload["entrypoint"] == {
        "module": "todo_api",
        "functions": ["add", "list_todos", "mark_done", "remove"],
    }
    assert payload["keystone_test_ids"] == c16.KEYSTONE_TEST_IDS
    assert len(payload["tests"]) == 1


def test_judge_rubric_is_well_formed():
    rubric = c16.get_judge_rubric()
    assert isinstance(rubric, dict)
    assert isinstance(rubric["criteria"], list)
    assert len(rubric["criteria"]) >= 3
    assert all(isinstance(c, str) and c.strip() for c in rubric["criteria"])
    assert isinstance(rubric.get("notes", ""), str) and rubric["notes"].strip()
    json.dumps(rubric)


def test_compiled_plan_structure():
    plan = c16.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "todo_api.py" in leaf["instruction"]
    assert "run_python" in leaf["instruction"] or "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["todo_api.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c16", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c16"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c16.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c16.get_compiled_plan()

    assert (private / c16.SMOKE_TEST_PATH).read_text() == c16.get_grading_payload()["tests"][c16.SMOKE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c16.SMOKE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "soft"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c16.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is True

    rubric_path = task_dir / "rubric.json"
    assert rubric_path.exists()
    assert json.loads(rubric_path.read_text()) == c16.get_judge_rubric()
