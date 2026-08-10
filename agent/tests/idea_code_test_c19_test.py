"""
Adversarial offline checks for codebench task c19 (key-value-store-with-persistence) —
no Docker, no LLM.

Soft-task variant of idea_code_test_c01_test.py's pattern: since there is no single
correct implementation, these checks verify the task module's OWN claims are internally
consistent — the smoke test file is well-formed and actually exercises real file I/O (not
just an in-memory check), get_judge_rubric() is well-formed, the compiled plan is
structurally sound, and materialize_task.py renders this task end-to-end.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c19_kv_store_persistence as c19


def test_visibility_is_hidden():
    assert c19.get_visibility() == "hidden"


def test_category_is_soft():
    assert c19.get_test_metadata()["category"] == "soft"


def test_sandbox_fixture_is_empty():
    # Hidden, implement-from-scratch task — nothing to give the agent up front, same
    # convention as c03's hidden-task get_sandbox_fixture().
    assert c19.get_sandbox_fixture() == {}


def test_smoke_test_uses_real_tempfile_not_pytest_fixtures():
    content = c19._TEST_FILE_CONTENT
    assert "import tempfile" in content
    assert "tmp_path" not in content, "must not depend on a pytest fixture the grading harness may not provide"
    assert "tempfile.mkstemp" in content or "tempfile.mkdtemp" in content


def test_smoke_test_round_trip_is_a_real_disk_check_not_a_tautology():
    """The round-trip test must mutate the in-memory value between save() and load() —
    otherwise a load() that's a total no-op would still make the assertion pass."""
    content = c19._TEST_FILE_CONTENT
    assert "save(path)" in content
    assert "load(path)" in content
    # the mutating re-set between save and load is what makes this non-trivial
    assert content.count('set_value("smoke_beta"') >= 2, (
        "expected set_value(smoke_beta) called twice (before save, and again after, to "
        "a DIFFERENT value) so a passing load() can only be explained by real disk I/O"
    )


def test_smoke_test_cleans_up_its_tempfile():
    assert "os.unlink(path)" in c19._TEST_FILE_CONTENT
    assert "finally:" in c19._TEST_FILE_CONTENT


def test_keystone_ids_reference_real_test_functions():
    import re

    content = c19._TEST_FILE_CONTENT
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c19.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c19.CANONICAL_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"
    # every smoke test is keystone — there is no larger suite to draw a subset from
    assert set(c19.KEYSTONE_TEST_IDS) == {f"{c19.CANONICAL_TEST_PATH}::{f}" for f in defined}


def test_grading_payload_shape():
    payload = c19.get_grading_payload()
    assert payload["tests"][c19.CANONICAL_TEST_PATH] == c19._TEST_FILE_CONTENT
    assert payload["entrypoint"] == {
        "module": "kv_store",
        "functions": ["set_value", "get_value", "save", "load"],
    }
    assert payload["keystone_test_ids"] == c19.KEYSTONE_TEST_IDS


def test_judge_rubric_is_well_formed():
    rubric = c19.get_judge_rubric()
    assert isinstance(rubric, dict)
    assert isinstance(rubric["criteria"], list)
    assert len(rubric["criteria"]) >= 3
    assert all(isinstance(c, str) and c.strip() for c in rubric["criteria"])
    assert isinstance(rubric.get("notes", ""), str)
    json.dumps(rubric)


def test_compiled_plan_structure():
    plan = c19.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "kv_store.py" in leaf["instruction"]
    assert "save" in leaf["instruction"] and "load" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["kv_store.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c19", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c19"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c19.get_task_statement()
    assert not (public / "repo").exists() or list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c19.get_compiled_plan()

    assert (private / c19.CANONICAL_TEST_PATH).read_text() == c19._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c19.CANONICAL_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "soft"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c19.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is True

    rubric_path = task_dir / "rubric.json"
    assert rubric_path.exists()
    assert json.loads(rubric_path.read_text()) == c19.get_judge_rubric()
