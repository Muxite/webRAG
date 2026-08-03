"""
Adversarial offline checks for codebench task c20 (write-a-readme-for-given-codebase) —
no Docker, no LLM.

This is the softest of the three soft tasks authored in this pass: the deliverable is
natural-language documentation, not code, so there's no ground truth to independently
re-derive the way c18's regression values or a hard task's function outputs allow. What
CAN be verified offline: the starter "given codebase" files are themselves valid,
importable Python (so a real agent could actually read and run them), the smoke test
correctly targets wherever run_grade.sh actually stages the submission, get_judge_rubric()
is well-formed, and materialize_task.py renders the whole task end-to-end.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c20_readme_for_codebase as c20


def _exec_module(content: str, name: str):
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(content, module.__dict__)  # noqa: S102 — trusted in-repo fixture
    return module


def test_visibility_is_hidden():
    assert c20.get_visibility() == "hidden"


def test_category_is_soft():
    assert c20.get_test_metadata()["category"] == "soft"


def test_starter_files_are_real_working_python():
    """The 'given mini-codebase' the agent has to document must actually run, or the
    task is unfair from the start — exercise every function/method named in the task
    statement / rubric."""
    utils = _exec_module(c20._UTILS_PY_CONTENT, "c20_utils_under_test")
    assert utils.slugify("Hello World") == "hello-world"
    assert utils.clamp(15, 0, 10) == 10
    assert utils.clamp(-5, 0, 10) == 0
    assert utils.clamp(5, 0, 10) == 5

    stack_mod = _exec_module(c20._STACK_PY_CONTENT, "c20_stack_under_test")
    s = stack_mod.Stack()
    assert s.is_empty() is True
    s.push(1)
    s.push(2)
    assert s.is_empty() is False
    assert s.pop() == 2
    assert s.pop() == 1
    assert s.is_empty() is True


def test_starter_files_are_small_as_specced():
    total_lines = (
        c20._UTILS_PY_CONTENT.count("\n") + c20._STACK_PY_CONTENT.count("\n")
    )
    assert total_lines <= 35, "starter mini-codebase was supposed to stay small (~20-30 lines)"


def test_sandbox_fixture_contains_both_starter_files():
    fixture = c20.get_sandbox_fixture()
    assert fixture["utils.py"] == c20._UTILS_PY_CONTENT
    assert fixture["stack.py"] == c20._STACK_PY_CONTENT
    assert c20.README_PATH not in fixture, "README.md is the agent's deliverable, not a starter file"


def test_smoke_test_checks_existence_and_size_not_content():
    """This task's grading test is deliberately a weak structural check (see the
    module-level docstring's note that it exercises the boundary of what pytest-based
    grading can check) — confirm it only checks existence/size, and targets the grade
    root one level up from tests/, matching run_grade.sh's layout."""
    content = c20._TEST_FILE_CONTENT
    assert 'os.path.join(os.path.dirname(__file__), "..", "README.md")' in content
    assert "os.path.exists(readme_path)" in content
    assert "os.path.getsize(readme_path) > 100" in content


def test_entrypoint_documents_its_own_shape_deviation():
    payload = c20.get_grading_payload()
    assert payload["entrypoint"] == {"deliverable": "README.md"}
    # not the module/functions convention every other task uses — deliberate, and
    # documented in the task module's own comment above get_grading_payload().
    assert "module" not in payload["entrypoint"]
    assert "functions" not in payload["entrypoint"]


def test_keystone_test_ids_reference_the_one_smoke_test():
    payload = c20.get_grading_payload()
    assert payload["keystone_test_ids"] == [f"{c20.SMOKE_TEST_PATH}::test_readme_exists_and_is_nontrivial"]


def test_judge_rubric_is_well_formed():
    rubric = c20.get_judge_rubric()
    assert isinstance(rubric, dict)
    assert isinstance(rubric["criteria"], list)
    assert len(rubric["criteria"]) >= 3
    assert all(isinstance(c, str) and c.strip() for c in rubric["criteria"])
    assert isinstance(rubric.get("notes", ""), str)
    assert "smoke test" in rubric["notes"].lower() or "grading" in rubric["notes"].lower(), (
        "rubric notes should acknowledge the smoke test's weak signal for this task"
    )
    json.dumps(rubric)


def test_compiled_plan_structure():
    plan = c20.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["document"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "README.md" in leaf["instruction"]
    assert "read_file" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["README.md"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c20", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c20"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c20.get_task_statement()
    assert (public / "repo" / "utils.py").read_text() == c20._UTILS_PY_CONTENT
    assert (public / "repo" / "stack.py").read_text() == c20._STACK_PY_CONTENT
    assert json.loads((public / "plan.json").read_text()) == c20.get_compiled_plan()

    assert (private / c20.SMOKE_TEST_PATH).read_text() == c20._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c20.SMOKE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "soft"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c20.get_grading_payload()["keystone_test_ids"]
    assert meta["has_rubric"] is True

    rubric_path = task_dir / "rubric.json"
    assert rubric_path.exists()
    assert json.loads(rubric_path.read_text()) == c20.get_judge_rubric()
