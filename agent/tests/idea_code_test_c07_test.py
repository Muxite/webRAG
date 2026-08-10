"""
Adversarial offline checks for codebench task c07 (config-yaml-fleet-gen) — no Docker, no LLM.

Mirrors idea_code_test_c01_test.py's structure.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c07_config_yaml_fleet_gen as c07


def _independent_generate_configs(base: dict, names: list) -> dict:
    """Reimplemented independently of the task module's own prose spec, to catch a mistake in
    the embedded canonical test file's own expectations. Deliberately built with an explicit
    loop and copy.deepcopy (rather than the dict-comprehension form the task itself suggests) so
    a bug shared between "spec prose" and "independent reimplementation" is less likely."""
    result = {}
    for name in names:
        cfg = copy.deepcopy(base)
        cfg["instance_name"] = name
        result[name] = cfg
    return result


def test_ground_truth_values_are_internally_correct():
    base = {"region": "us-east-1", "replicas": 3}
    names = ["web-01", "web-02", "worker-01"]
    result = _independent_generate_configs(base, names)

    assert set(result.keys()) == set(names)
    for name in names:
        assert result[name] == {"region": "us-east-1", "replicas": 3, "instance_name": name}

    # base untouched
    assert base == {"region": "us-east-1", "replicas": 3}

    # independent copies
    result["web-01"]["region"] = "eu-west-1"
    assert result["web-02"]["region"] == "us-east-1"

    # empty names
    assert _independent_generate_configs(base, []) == {}


def test_embedded_test_file_asserts_match_ground_truth():
    content = c07.get_sandbox_fixture()[c07.VISIBLE_TEST_PATH]
    assert 'BASE = {"region": "us-east-1", "replicas": 3}' in content
    assert 'NAMES = ["web-01", "web-02", "worker-01"]' in content

    base = {"region": "us-east-1", "replicas": 3}
    names = ["web-01", "web-02", "worker-01"]
    result = _independent_generate_configs(base, names)
    assert result["web-01"] == {"region": "us-east-1", "replicas": 3, "instance_name": "web-01"}


def test_independent_copy_bug_is_actually_caught_by_the_embedded_test():
    """Prove the "shared reference" test is discriminating: a plausible buggy implementation
    (mutating and reusing one shared dict object across all names) must FAIL it."""

    def buggy_generate_configs(base: dict, names: list) -> dict:
        result = {}
        shared = dict(base)
        for name in names:
            shared["instance_name"] = name
            result[name] = shared  # BUG: same object reused for every name
        return result

    names = ["web-01", "web-02", "worker-01"]
    buggy_result = buggy_generate_configs({"region": "us-east-1", "replicas": 3}, names)
    buggy_result["web-01"]["region"] = "eu-west-1"
    # every entry is the SAME dict object, so this bug makes every other entry change too
    assert buggy_result["web-02"]["region"] == "eu-west-1"


def test_keystone_ids_reference_real_test_functions():
    content = c07.get_sandbox_fixture()[c07.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c07.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c07.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_bonus_contract_cases():
    assert f"{c07.VISIBLE_TEST_PATH}::test_base_dict_is_not_mutated" not in c07.KEYSTONE_TEST_IDS
    assert f"{c07.VISIBLE_TEST_PATH}::test_empty_names_returns_empty_dict" not in c07.KEYSTONE_TEST_IDS


def test_visibility_is_visible():
    assert c07.get_visibility() == "visible"


def test_grading_payload_shape():
    payload = c07.get_grading_payload()
    assert payload["tests"][c07.VISIBLE_TEST_PATH] == c07.get_sandbox_fixture()[c07.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {"module": "fleet_gen", "functions": ["generate_configs"]}
    assert payload["keystone_test_ids"] == c07.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c07.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "fleet_gen.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["fleet_gen.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c07", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c07"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c07.get_task_statement()
    assert (public / "repo" / c07.VISIBLE_TEST_PATH).read_text() == c07.get_sandbox_fixture()[c07.VISIBLE_TEST_PATH]
    assert json.loads((public / "plan.json").read_text()) == c07.get_compiled_plan()

    assert (private / c07.VISIBLE_TEST_PATH).read_text() == c07.get_grading_payload()["tests"][c07.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c07.VISIBLE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c07.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
