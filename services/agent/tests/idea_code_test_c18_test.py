"""
Adversarial offline checks for codebench task c18 (refactor-messy-module) — no Docker, no LLM.

Soft-task variant of idea_code_test_c01_test.py's pattern: instead of pinning down a single
correct output (there isn't one — this is open-ended), verify (1) the regression suite's
embedded expected values actually match running the REAL starter messy.py (same
ground-truth-independently-verified principle hard tasks use), (2) get_judge_rubric() is
well-formed, (3) the compiled plan is structurally sound and JSON-serializable, and (4)
materialize_task.py renders this task end-to-end including writing rubric.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c18_refactor_messy_module as c18


def _load_messy_calc():
    """Import the embedded messy.py content as a real module and return its calc(), so
    the regression suite's literal assertions can be checked against ACTUALLY RUNNING it
    rather than trusted at face value."""
    spec = importlib.util.spec_from_loader("c18_messy_under_test", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(c18._MESSY_PY_CONTENT, module.__dict__)  # noqa: S102 — trusted in-repo fixture
    return module.calc


def test_starter_module_is_genuinely_messy():
    content = c18._MESSY_PY_CONTENT
    assert "x1" in content and "tmp2" in content, "expected the unclear-name pattern"
    # deeply nested: at least 3 levels of indentation inside one branch
    assert "            result = x1 - tmp2" in content or "        if x1 > tmp2" in content
    assert content.count("if ") + content.count("elif ") >= 6, "expected a real if/elif chain"


def test_regression_values_match_running_the_real_starter_module():
    calc = _load_messy_calc()
    content = c18.get_sandbox_fixture()[c18.VISIBLE_TEST_PATH]
    pairs = re.findall(
        r'calc\((-?\d+), (-?\d+), "(\w+)"\) == (-?\d+(?:\.\d+)?)', content
    )
    assert pairs, "expected at least one literal numeric assertion in the regression file"
    for a_str, b_str, op, expected_str in pairs:
        expected = float(expected_str) if "." in expected_str else int(expected_str)
        assert calc(int(a_str), int(b_str), op) == expected, (a_str, b_str, op, expected_str)


def test_regression_values_include_none_and_unknown_op_cases():
    calc = _load_messy_calc()
    assert calc(10, 0, "div") is None, "division by zero must be a handled None, not a crash"
    assert calc(3, 5, "unknown") is None, "an unrecognized op must be a handled None"
    content = c18.get_sandbox_fixture()[c18.VISIBLE_TEST_PATH]
    assert "is None" in content, "expected at least one `is None` assertion in the regression file"


def test_dead_branch_is_actually_unreachable():
    """The final `else: result = -999` under the sub op's >/</== chain can never execute
    for real numbers (exactly one of >, <, == always holds) — confirm no embedded
    assertion depends on it, and that direct construction can't reach it either."""
    calc = _load_messy_calc()
    for a in range(-3, 4):
        for b in range(-3, 4):
            assert calc(a, b, "sub") != -999
    assert "-999" not in c18.get_sandbox_fixture()[c18.VISIBLE_TEST_PATH]


def test_keystone_ids_reference_real_test_functions():
    content = c18.get_sandbox_fixture()[c18.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c18.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c18.VISIBLE_TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_covers_every_test_in_the_regression_file():
    # Unlike a hard task's degenerate-case exclusions, this task's whole point is "don't
    # change behavior" — every regression assertion is load-bearing for that constraint.
    content = c18.get_sandbox_fixture()[c18.VISIBLE_TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    keystone_funcs = {nid.partition("::")[2] for nid in c18.KEYSTONE_TEST_IDS}
    assert keystone_funcs == defined


def test_visibility_is_visible():
    assert c18.get_visibility() == "visible"


def test_category_is_soft():
    assert c18.get_test_metadata()["category"] == "soft"


def test_sandbox_fixture_includes_starter_module_and_test_file():
    fixture = c18.get_sandbox_fixture()
    assert fixture[c18.STARTER_MODULE_PATH] == c18._MESSY_PY_CONTENT
    assert fixture[c18.VISIBLE_TEST_PATH] == c18._TEST_FILE_CONTENT


def test_grading_payload_shape():
    payload = c18.get_grading_payload()
    assert payload["tests"][c18.VISIBLE_TEST_PATH] == c18.get_sandbox_fixture()[c18.VISIBLE_TEST_PATH]
    assert payload["entrypoint"] == {"module": "messy", "functions": ["calc"]}
    assert payload["keystone_test_ids"] == c18.KEYSTONE_TEST_IDS


def test_judge_rubric_is_well_formed():
    rubric = c18.get_judge_rubric()
    assert isinstance(rubric, dict)
    assert isinstance(rubric["criteria"], list)
    assert len(rubric["criteria"]) >= 3
    assert all(isinstance(c, str) and c.strip() for c in rubric["criteria"])
    assert isinstance(rubric.get("notes", ""), str)
    json.dumps(rubric)  # written verbatim to rubric.json by materialize_task.py


def test_compiled_plan_structure():
    plan = c18.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["refactor"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "messy.py" in leaf["instruction"]
    assert "run_pytest" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["messy.py"]}
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c18", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c18"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c18.get_task_statement()
    assert (public / "repo" / c18.STARTER_MODULE_PATH).read_text() == c18._MESSY_PY_CONTENT
    assert (public / "repo" / c18.VISIBLE_TEST_PATH).read_text() == c18._TEST_FILE_CONTENT
    assert json.loads((public / "plan.json").read_text()) == c18.get_compiled_plan()

    assert (private / c18.VISIBLE_TEST_PATH).read_text() == c18.get_grading_payload()["tests"][c18.VISIBLE_TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c18.VISIBLE_TEST_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "soft"
    assert meta["visibility"] == "visible"
    assert meta["keystone_test_ids"] == c18.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is True

    rubric_path = task_dir / "rubric.json"
    assert rubric_path.exists()
    assert json.loads(rubric_path.read_text()) == c18.get_judge_rubric()
