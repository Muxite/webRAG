"""
Adversarial offline checks for codebench task c34 (directed-cycle-detection) -- no Docker, no
LLM.

Mirrors idea_code_test_c01_test.py / idea_code_test_c06_test.py: prove the task module's own
claims are internally consistent (ground truth is actually correct, keystone ids reference
real tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox.
Also specifically proves the false-positive trap the task is built around: an independent
BUGGY implementation that uses a flat "visited" set (no gray/black distinction) actually does
get the diamond case wrong, confirming the task genuinely discriminates the intended bug.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c34_directed_cycle_detection as c34


def _all_nodes(graph: dict) -> set:
    nodes = set(graph.keys())
    for succs in graph.values():
        nodes.update(succs)
    return nodes


def _independent_has_cycle(graph: dict) -> bool:
    """Reimplemented independently of the task's own prose spec (three-color DFS, written
    from scratch here rather than copied), to catch a mistake in the embedded canonical test
    file's own expected literals."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in _all_nodes(graph)}

    def dfs(u):
        color[u] = GRAY
        for v in graph.get(u, []):
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(color[n] == WHITE and dfs(n) for n in list(color))


def _buggy_flat_visited_has_cycle(graph: dict) -> bool:
    """Deliberately the WRONG implementation this task is designed to catch: a single flat
    'visited' set with no distinction between 'on the current DFS path' and 'already fully
    explored via another path'. Any node revisited at all (even via a completely different,
    unrelated path) is wrongly reported as a cycle."""
    visited = set()

    def dfs(u):
        if u in visited:
            return True  # BUG: treats any revisit as a cycle
        visited.add(u)
        return any(dfs(v) for v in graph.get(u, []))

    return any(dfs(n) for n in list(_all_nodes(graph)) if n not in visited)


def test_ground_truth_values_are_internally_correct():
    cases = [
        ({"a": ["b"], "b": ["c"], "c": []}, False),
        ({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}, False),
        ({"a": ["b"], "b": ["c"], "c": ["a"]}, True),
        ({"a": ["b"], "b": ["a"]}, True),
        ({"a": ["a"], "b": []}, True),
        ({"a": ["b"], "b": ["c"], "c": ["d"], "d": ["b"]}, True),
        ({"a": ["b"], "b": ["a"], "c": ["d"], "d": []}, True),
        ({"a": ["b"], "c": ["d"]}, False),
        ({"a": ["b"]}, False),
        ({"a": ["z"], "b": ["z"], "c": ["z"], "z": []}, False),
    ]
    for graph, expected in cases:
        assert _independent_has_cycle(graph) is expected, graph


def test_buggy_flat_visited_implementation_genuinely_fails_the_diamond_case():
    # Proves the task's central discriminator is real: a plausible-looking but wrong
    # implementation (flat visited set, no gray/black distinction) gets the diamond case
    # wrong (reports a cycle where there is none), and would therefore fail the keystone
    # test built around it.
    diamond = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    assert _independent_has_cycle(diamond) is False
    assert _buggy_flat_visited_has_cycle(diamond) is True, (
        "expected the buggy reference to false-positive on the diamond -- if it doesn't, "
        "the diamond case no longer discriminates the intended bug"
    )


def test_embedded_test_file_literals_match_ground_truth():
    content = c34.get_grading_payload()["tests"][c34._TEST_FILE_PATH]
    checks = [
        ('{"a": ["b"], "b": ["c"], "c": []}', {"a": ["b"], "b": ["c"], "c": []}, False),
        (
            '{"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}',
            {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []},
            False,
        ),
        ('{"a": ["b"], "b": ["c"], "c": ["a"]}', {"a": ["b"], "b": ["c"], "c": ["a"]}, True),
        ('{"a": ["b"], "b": ["a"]}', {"a": ["b"], "b": ["a"]}, True),
        ('{"a": ["a"], "b": []}', {"a": ["a"], "b": []}, True),
        (
            '{"a": ["b"], "b": ["c"], "c": ["d"], "d": ["b"]}',
            {"a": ["b"], "b": ["c"], "c": ["d"], "d": ["b"]},
            True,
        ),
        (
            '{"a": ["b"], "b": ["a"], "c": ["d"], "d": []}',
            {"a": ["b"], "b": ["a"], "c": ["d"], "d": []},
            True,
        ),
        ('{"a": ["b"], "c": ["d"]}', {"a": ["b"], "c": ["d"]}, False),
        ('{"a": ["b"]}', {"a": ["b"]}, False),
        (
            '{"a": ["z"], "b": ["z"], "c": ["z"], "z": []}',
            {"a": ["z"], "b": ["z"], "c": ["z"], "z": []},
            False,
        ),
    ]
    for literal, graph, expected in checks:
        assert literal in content, literal
        assert _independent_has_cycle(graph) is expected


def test_keystone_ids_reference_real_test_functions():
    content = c34.get_grading_payload()["tests"][c34._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c34.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c34._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_is_balanced_across_both_answers():
    # Neither an "always True" nor an "always False" implementation should be able to pass
    # every keystone test -- confirm both outcomes are represented among the keystone ids by
    # cross-referencing each keystone test name back to its expected boolean above.
    expected_by_name = {
        "test_diamond_shared_descendant_is_not_a_cycle": False,
        "test_simple_three_node_cycle": True,
        "test_two_node_cycle": True,
        "test_self_loop_is_a_cycle": True,
        "test_dag_except_one_back_edge": True,
        "test_disconnected_component_with_a_cycle_is_detected": True,
        "test_disconnected_components_with_no_cycle": False,
    }
    keystone_names = {nid.partition("::")[2] for nid in c34.KEYSTONE_TEST_IDS}
    assert keystone_names == set(expected_by_name)
    outcomes = set(expected_by_name.values())
    assert outcomes == {True, False}, "keystone set must include both True and False cases"


def test_keystone_excludes_the_easier_edge_cases():
    easy = [
        "test_linear_chain_has_no_cycle",
        "test_node_referenced_only_as_a_value_has_no_cycle",
        "test_wide_fanin_dag_has_no_cycle",
    ]
    for name in easy:
        assert f"{c34._TEST_FILE_PATH}::{name}" not in c34.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c34.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c34.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c34.get_grading_payload()
    assert payload["tests"] == {c34._TEST_FILE_PATH: c34._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {"module": "cycle_detect", "functions": ["has_cycle"]}
    assert payload["keystone_test_ids"] == c34.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c34.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "cycle_detect.py" in leaf["instruction"]
    assert "shared-descendant" in leaf["instruction"] or "converging" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["cycle_detect.py"]}
    # Must be JSON-serializable as-is -- see c01's identical check for why.
    json.dumps(plan)


def test_compiled_plan_leaks_no_canonical_literal_values():
    # The plan is public; it must describe the ALGORITHM, not embed private test literals.
    plan = c34.get_compiled_plan()
    instruction = plan["leaves"][0]["instruction"]
    assert '["b"], "b": ["c"], "c": ["a"]' not in instruction
    assert '["b"], "b": ["a"], "c": ["d"]' not in instruction


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c34", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c34"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c34.get_task_statement()
    # Hidden task: no starter files under public/repo/ at all.
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c34.get_compiled_plan()

    assert (private / c34._TEST_FILE_PATH).read_text() == c34._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c34._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c34.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
