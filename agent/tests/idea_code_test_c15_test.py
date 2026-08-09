"""
Adversarial offline checks for codebench task c15 (build-a-calculator-cli) — no Docker, no LLM.

Soft/hidden task: unlike c01 (hard/visible), there's no single canonical implementation to
grade byte-for-byte, so these checks only verify the module's OWN internal consistency (light
smoke tests are well-formed, visibility is hidden so the fixture is empty, the judge rubric is
present and well-formed) and exercise materialize_task.py end-to-end — including its rubric.json
output path, which c01's test never touches (c01 has no get_judge_rubric()).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c15_calculator_cli as c15


def test_metadata_is_soft():
    meta = c15.get_test_metadata()
    assert meta["test_id"] == "c15"
    assert meta["category"] == "soft"


def test_visibility_is_hidden():
    assert c15.get_visibility() == "hidden"


def test_sandbox_fixture_is_empty_for_hidden_task():
    assert c15.get_sandbox_fixture() == {}


def test_task_statement_covers_key_requirements():
    statement = c15.get_task_statement()
    assert "calculator.py" in statement
    assert "evaluate" in statement
    # explicit instruction NOT to use eval() must actually be present in the spec text
    assert "eval()" in statement
    assert "division by zero" in statement.lower()
    assert "parenthes" in statement.lower()


def test_smoke_test_file_contains_the_three_described_cases():
    content = c15.get_grading_payload()["tests"][c15.SMOKE_TEST_PATH]
    assert '"2 + 2"' in content and "== 4" in content
    assert '"2 * (3 + 4)"' in content and "== 14" in content
    assert "pytest.raises(Exception)" in content
    assert '"2 / 0"' in content


def test_keystone_ids_reference_real_test_functions():
    content = c15.get_grading_payload()["tests"][c15.SMOKE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert c15.KEYSTONE_TEST_IDS, "c15 defines a non-empty keystone list (all smoke checks gate)"
    for node_id in c15.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c15.SMOKE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_grading_payload_shape():
    payload = c15.get_grading_payload()
    assert payload["tests"][c15.SMOKE_TEST_PATH] == c15._TEST_FILE_CONTENT
    assert payload["entrypoint"] == {"module": "calculator", "functions": ["evaluate"]}
    assert payload["keystone_test_ids"] == c15.KEYSTONE_TEST_IDS
    # soft-task suites are light — a handful of smoke checks, not an exhaustive spec
    assert len(payload["tests"]) == 1
    assert len(payload["keystone_test_ids"]) <= 5


def test_judge_rubric_is_well_formed():
    rubric = c15.get_judge_rubric()
    assert isinstance(rubric, dict)
    assert isinstance(rubric["criteria"], list)
    assert len(rubric["criteria"]) >= 3
    assert all(isinstance(c, str) and c.strip() for c in rubric["criteria"])
    assert isinstance(rubric.get("notes", ""), str) and rubric["notes"].strip()
    # must be plain JSON-safe data — materialize_task.py writes it verbatim to rubric.json
    json.dumps(rubric)


def test_compiled_plan_structure():
    plan = c15.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "calculator.py" in leaf["instruction"]
    assert "run_python" in leaf["instruction"] or "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["calculator.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c15", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c15"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c15.get_task_statement()
    # hidden task: no fixture files placed under public/repo at all
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c15.get_compiled_plan()

    assert (private / c15.SMOKE_TEST_PATH).read_text() == c15.get_grading_payload()["tests"][c15.SMOKE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c15.SMOKE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "soft"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c15.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is True

    rubric_path = task_dir / "rubric.json"
    assert rubric_path.exists()
    assert json.loads(rubric_path.read_text()) == c15.get_judge_rubric()
