"""
Adversarial offline checks for codebench task c14 (log-line-aggregator) — no Docker, no LLM.

Mirrors idea_code_test_c01_test.py's spirit, adapted for a HIDDEN task: prove the task
module's own claims are internally consistent (ground truth is actually correct, keystone ids
reference real tests, the compiled plan is well-formed, and — the hidden-specific check — no
test content leaks into the agent-visible sandbox fixture) BEFORE anything ever reaches a live
sandbox, plus an end-to-end exercise of materialize_task.py against this task.

The reimplementation below matches each line against a compiled regex anchored at line start
(`^(INFO|WARN|ERROR): `) — structurally different from a manual startswith-prefix-per-level
loop an implementation might use — used purely to independently re-derive the literal count
dicts embedded in the canonical hidden test fixture, to catch a hand-transcription error.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c14_log_line_aggregator as c14

_LEVEL_RE = re.compile(r"^(INFO|WARN|ERROR): ")


def _independent_aggregate_by_level(log_lines: list) -> dict:
    """Reimplemented independently of the task's own prose spec, via a single anchored
    regex match rather than a per-level startswith loop, to catch a hand-transcription
    error in the task module's embedded expected values."""
    counts = {"INFO": 0, "WARN": 0, "ERROR": 0}
    for line in log_lines:
        m = _LEVEL_RE.match(line)
        if m:
            counts[m.group(1)] += 1
    return counts


def _extract_embedded_cases(content: str) -> dict:
    """Pull each test_*'s `lines = [...]` list and the dict it's asserted equal to, via AST
    (not regex) since the fixtures are multi-line list/dict literals — robust to formatting,
    unlike a hand-rolled multi-line regex would be."""
    tree = ast.parse(content)
    cases = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        lines_val = None
        expected_val = None
        for stmt in ast.walk(node):
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "lines"
            ):
                lines_val = ast.literal_eval(stmt.value)
            if isinstance(stmt, ast.Assert) and isinstance(stmt.test, ast.Compare):
                comparators = stmt.test.comparators
                if comparators:
                    expected_val = ast.literal_eval(comparators[0])
        if lines_val is not None and expected_val is not None:
            cases[node.name] = (lines_val, expected_val)
    return cases


def test_ground_truth_values_are_internally_correct():
    cases = {
        "mixed": (
            [
                "INFO: server started", "WARN: low disk space", "ERROR: connection refused",
                "INFO: request handled", "INFO: request handled again", "ERROR: timeout",
            ],
            {"INFO": 3, "WARN": 1, "ERROR": 2},
        ),
        "zero_occurrence": (
            ["INFO: booted", "INFO: ready", "ERROR: crash"],
            {"INFO": 2, "WARN": 0, "ERROR": 1},
        ),
        "malformed": (
            [
                "INFO: ok", "this line has no level prefix at all", "WARN: careful",
                "DEBUG: not a recognized level", "ERROR: bad",
                "INFO:missing the space after colon",
            ],
            {"INFO": 1, "WARN": 1, "ERROR": 1},
        ),
        "case_sensitivity": (
            [
                "INFO: real one", "info: lowercase should not count",
                "Info: mixed case should not count", "WARN: real warn",
            ],
            {"INFO": 1, "WARN": 1, "ERROR": 0},
        ),
    }
    for name, (lines, expected) in cases.items():
        assert _independent_aggregate_by_level(lines) == expected, name


def test_embedded_test_cases_match_ground_truth():
    content = c14.get_grading_payload()["tests"][c14.TEST_PATH]
    cases = _extract_embedded_cases(content)
    assert len(cases) == 4, f"expected 4 embedded test cases, found {sorted(cases)}"
    for name, (lines, expected) in cases.items():
        assert _independent_aggregate_by_level(lines) == expected, name
        # Sanity: canonical dict always carries exactly the three fixed keys.
        assert set(expected.keys()) == {"INFO", "WARN", "ERROR"}, name


def test_malformed_and_case_sensitivity_cases_present():
    content = c14.get_grading_payload()["tests"][c14.TEST_PATH]
    assert "DEBUG: not a recognized level" in content
    assert "INFO:missing the space after colon" in content
    assert "info: lowercase should not count" in content
    assert "Info: mixed case should not count" in content


def test_no_test_content_leaks_into_agent_visible_fixture():
    # Hidden task contract: the sandbox fixture the agent's /work gets seeded with must be
    # empty — the canonical test only lives in get_grading_payload(), which materialize_task
    # writes to private/ (never mounted into the agent sandbox).
    assert c14.get_sandbox_fixture() == {}


def test_keystone_ids_reference_real_test_functions():
    content = c14.get_grading_payload()["tests"][c14.TEST_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c14.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c14.TEST_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_all_four_cases_are_keystone():
    # Unlike c01/c13, none of c14's four cases are degenerate or contract-only bonus credit —
    # each one confirms a distinct part of the hidden spec (happy path, always-present keys,
    # silent-skip, exact-case matching) — so all four gate the score.
    assert len(c14.KEYSTONE_TEST_IDS) == 4


def test_visibility_is_hidden():
    assert c14.get_visibility() == "hidden"


def test_grading_payload_shape():
    payload = c14.get_grading_payload()
    assert payload["entrypoint"] == {"module": "log_agg", "functions": ["aggregate_by_level"]}
    assert payload["keystone_test_ids"] == c14.KEYSTONE_TEST_IDS
    assert c14.TEST_PATH in payload["tests"]


def test_compiled_plan_structure():
    plan = c14.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "log_agg.py" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["log_agg.py"]}
    # Must be JSON-serializable as-is: materialize_task.py writes it verbatim to
    # public/plan.json with no transformation, and the agent image never imports this
    # module at all (see agents/badmodel/Dockerfile's SECURITY comment) — a plan that
    # isn't plain JSON-safe data would silently break at materialize time.
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path, codebench_materialize_script):
    repo_root = Path(__file__).resolve().parents[2]
    script = codebench_materialize_script
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c14", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c14"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c14.get_task_statement()
    assert json.loads((public / "plan.json").read_text()) == c14.get_compiled_plan()
    # Hidden task: nothing lands under public/repo — the agent gets the prompt only.
    assert not (public / "repo" / c14.TEST_PATH).exists()
    assert sorted((public / "repo").iterdir()) == []

    assert (private / c14.TEST_PATH).read_text() == c14.get_grading_payload()["tests"][c14.TEST_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert manifest["test_file_globs"] == [c14.TEST_PATH]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c14.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
