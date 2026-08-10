"""
Adversarial offline checks for codebench task c17 (markdown-to-html-renderer) — no Docker,
no LLM.

Same shape as idea_code_test_c15_test.py/idea_code_test_c16_test.py: soft/hidden task, so
these checks only verify the module's own internal consistency and exercise
materialize_task.py end-to-end (including rubric.json). c17-specific: the spec text must
name exactly the three MUST-support features (h1-h3 headings, bold, paragraphs) since that's
the explicit scope boundary distinguishing this from a full CommonMark renderer.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c17_markdown_html_renderer as c17


def test_metadata_is_soft():
    meta = c17.get_test_metadata()
    assert meta["test_id"] == "c17"
    assert meta["category"] == "soft"


def test_visibility_is_hidden():
    assert c17.get_visibility() == "hidden"


def test_sandbox_fixture_is_empty_for_hidden_task():
    assert c17.get_sandbox_fixture() == {}


def test_task_statement_covers_key_requirements():
    statement = c17.get_task_statement()
    assert "md_render.py" in statement
    assert "render" in statement
    assert "<h1>" in statement and "<h2>" in statement and "<h3>" in statement
    assert "**bold**" in statement
    assert "<p>" in statement
    assert "CommonMark" in statement  # explicit small-subset scope boundary


def test_smoke_test_file_contains_the_three_described_cases():
    content = c17.get_grading_payload()["tests"][c17.SMOKE_TEST_PATH]
    assert '"# Hello"' in content and "<h1>" in content
    assert '"**bold**"' in content
    assert "<b>" in content and "<strong>" in content  # the loose either/or check
    assert "<p>" in content


def test_keystone_ids_reference_real_test_functions():
    content = c17.get_grading_payload()["tests"][c17.SMOKE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    assert c17.KEYSTONE_TEST_IDS, "c17 defines a non-empty keystone list (all smoke checks gate)"
    for node_id in c17.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c17.SMOKE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_grading_payload_shape():
    payload = c17.get_grading_payload()
    assert payload["tests"][c17.SMOKE_TEST_PATH] == c17._TEST_FILE_CONTENT
    assert payload["entrypoint"] == {"module": "md_render", "functions": ["render"]}
    assert payload["keystone_test_ids"] == c17.KEYSTONE_TEST_IDS
    assert len(payload["tests"]) == 1
    assert len(payload["keystone_test_ids"]) <= 5


def test_judge_rubric_is_well_formed():
    rubric = c17.get_judge_rubric()
    assert isinstance(rubric, dict)
    assert isinstance(rubric["criteria"], list)
    assert len(rubric["criteria"]) >= 3
    assert all(isinstance(c, str) and c.strip() for c in rubric["criteria"])
    assert isinstance(rubric.get("notes", ""), str) and rubric["notes"].strip()
    json.dumps(rubric)


def test_compiled_plan_structure():
    plan = c17.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "md_render.py" in leaf["instruction"]
    assert "run_python" in leaf["instruction"] or "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["md_render.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c17", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c17"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c17.get_task_statement()
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c17.get_compiled_plan()

    assert (private / c17.SMOKE_TEST_PATH).read_text() == c17.get_grading_payload()["tests"][c17.SMOKE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c17.SMOKE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "soft"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c17.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is True

    rubric_path = task_dir / "rubric.json"
    assert rubric_path.exists()
    assert json.loads(rubric_path.read_text()) == c17.get_judge_rubric()
