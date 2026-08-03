"""
Adversarial offline checks for codebench task c06 (topo-sort-build-deps) — no Docker, no LLM.

Mirrors idea_code_test_c01_test.py: prove the task module's own claims are internally consistent
(ground truth is actually correct, keystone ids reference real tests, the compiled plan is
well-formed) BEFORE anything ever reaches a live sandbox. c06 is the first genuinely MULTI-LEAF
codebench task, so this file additionally checks the leaf_a/leaf_b dependency wiring: exactly two
leaves, leaf_b's depends_on names leaf_a, and leaf_b's instruction actually references leaf_a's id
via a ``{leaf_a}`` placeholder (the mechanism execution_compiled_code.py's ``substitute_deps``/
``_dep_text`` fill in at runtime).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c06_topo_sort_build_deps as c06


def _independent_topo_sort(deps: dict) -> list:
    """Reimplemented independently of the task module's own topo_sort/graph_types (plain Kahn's
    algorithm over dicts, no Graph class at all), to catch a mistake in the embedded canonical
    test file's own expectations."""
    all_nodes = list(deps.keys())
    for dlist in deps.values():
        for d in dlist:
            if d not in all_nodes:
                all_nodes.append(d)
    indegree = {n: 0 for n in all_nodes}
    successors = {n: [] for n in all_nodes}
    for target, dlist in deps.items():
        for d in dlist:
            successors[d].append(target)
            indegree[target] += 1
    queue = [n for n in all_nodes if indegree[n] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for succ in successors[node]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                queue.append(succ)
    if len(order) != len(all_nodes):
        raise ValueError("cycle detected in dependency graph")
    return order


def _independent_is_valid_topo_order(deps: dict, order: list) -> bool:
    all_nodes = set(deps.keys())
    for dlist in deps.values():
        all_nodes.update(dlist)
    if set(order) != all_nodes or len(order) != len(all_nodes):
        return False
    position = {node: i for i, node in enumerate(order)}
    for target, dlist in deps.items():
        for dep in dlist:
            if position[dep] >= position[target]:
                return False
    return True


def test_ground_truth_values_are_internally_correct():
    # Cases with exactly one valid order: assert the exact list.
    assert _independent_topo_sort({"solo": []}) == ["solo"]
    assert _independent_topo_sort({"a": [], "b": ["a"], "c": ["b"]}) == ["a", "b", "c"]

    # Cases with multiple valid orders: assert structural validity instead of one hardcoded list.
    multi_order_cases = [
        {"app": ["lib_a", "lib_b"], "lib_a": ["core"], "lib_b": ["core"], "core": []},
        {"a": [], "b": ["a"], "x": [], "y": ["x"]},
        {"app": ["utillib"]},
        {"root": [], "a": ["root"], "b": ["root"], "c": ["root"]},
    ]
    for deps in multi_order_cases:
        order = _independent_topo_sort(deps)
        assert _independent_is_valid_topo_order(deps, order), (deps, order)

    # Cycles must raise.
    import pytest
    for deps in [{"a": ["b"], "b": ["c"], "c": ["a"]}, {"a": ["b"], "b": ["a"]}]:
        with pytest.raises(ValueError):
            _independent_topo_sort(deps)


def test_is_valid_topo_order_checker_rejects_a_wrong_order():
    # Sanity-check the checker itself: a deliberately wrong order must be rejected.
    deps = {"app": ["lib_a", "lib_b"], "lib_a": ["core"], "lib_b": ["core"], "core": []}
    bad_order = ["app", "lib_a", "core", "lib_b"]  # core placed after lib_a -- invalid
    assert not _independent_is_valid_topo_order(deps, bad_order)
    good_order = ["core", "lib_a", "lib_b", "app"]
    assert _independent_is_valid_topo_order(deps, good_order)


def test_embedded_test_file_defines_and_uses_a_structural_checker():
    content = c06.get_grading_payload()["tests"][c06._TEST_FILE_PATH]
    assert "def is_valid_topo_order(deps: dict, order: list) -> bool:" in content
    # Multi-valid-order cases must use the checker, not a single hardcoded expected list.
    assert "assert is_valid_topo_order(deps, order)" in content


def test_embedded_test_file_pins_the_two_single_order_cases_exactly():
    content = c06.get_grading_payload()["tests"][c06._TEST_FILE_PATH]
    assert 'topo_sort({"solo": []}) == ["solo"]' in content
    assert 'topo_sort({"a": [], "b": ["a"], "c": ["b"]}) == ["a", "b", "c"]' in content


def test_embedded_test_file_covers_both_cycle_shapes():
    content = c06.get_grading_payload()["tests"][c06._TEST_FILE_PATH]
    assert content.count("pytest.raises(ValueError)") == 2


def test_keystone_ids_reference_real_test_functions():
    content = c06.get_grading_payload()["tests"][c06._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c06.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c06._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_the_two_non_discriminating_cases():
    # single-node and "dependency only referenced as a value" don't exercise real traversal.
    assert f"{c06._TEST_FILE_PATH}::test_single_node_no_deps" not in c06.KEYSTONE_TEST_IDS
    assert f"{c06._TEST_FILE_PATH}::test_dependency_referenced_only_as_value" not in c06.KEYSTONE_TEST_IDS


def test_visibility_is_hidden():
    assert c06.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c06.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c06.get_grading_payload()
    assert payload["tests"] == {c06._TEST_FILE_PATH: c06._TEST_FILE_CONTENT}
    assert payload["keystone_test_ids"] == c06.KEYSTONE_TEST_IDS
    modules = payload["entrypoint"]["modules"]
    assert {m["module"] for m in modules} == {"graph_types", "topo_sort"}
    graph_mod = next(m for m in modules if m["module"] == "graph_types")
    topo_mod = next(m for m in modules if m["module"] == "topo_sort")
    assert graph_mod["class"] == "Graph"
    assert set(graph_mod["methods"]) == {"add_node", "add_edge", "nodes", "successors", "in_degree"}
    assert topo_mod["functions"] == ["topo_sort"]


def test_compiled_plan_has_exactly_two_leaves_with_a_real_dependency():
    plan = c06.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["leaf_a", "leaf_b"]
    leaf_a, leaf_b = plan["leaves"]
    assert leaf_a["depends_on"] == []
    assert leaf_b["depends_on"] == ["leaf_a"]


def test_leaf_b_instruction_references_leaf_a_via_placeholder_and_restates_the_api():
    plan = c06.get_compiled_plan()
    leaf_b = plan["leaves"][1]
    assert "{leaf_a}" in leaf_b["instruction"], (
        "leaf_b must template leaf_a's actual outcome via {leaf_a} -- substitute_deps() only "
        "replaces placeholders that literally match a dependency id"
    )
    # Fallback: leaf_b independently restates the API contract rather than trusting leaf_a's
    # self-report alone (a weak model's own summary of what it built is not fully reliable).
    for name in ("add_node", "add_edge", "nodes", "successors", "in_degree", "Graph"):
        assert name in leaf_b["instruction"], name
    assert "from graph_types import Graph" in leaf_b["instruction"]


def test_leaf_a_instruction_names_the_exact_api_leaf_b_will_rely_on():
    plan = c06.get_compiled_plan()
    leaf_a = plan["leaves"][0]
    for name in ("Graph", "add_node", "add_edge", "nodes", "successors", "in_degree"):
        assert name in leaf_a["instruction"], name
    assert "graph_types.py" in leaf_a["instruction"]


def test_compiled_plan_structure():
    plan = c06.get_compiled_plan()
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["graph_types.py", "topo_sort.py"]}
    # Must be JSON-serializable as-is -- see c01's identical check for why.
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root / "services")}
    result = subprocess.run(
        [sys.executable, str(script), "c06", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c06"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c06.get_task_statement()
    # Hidden task: no starter files under public/repo/ at all.
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c06.get_compiled_plan()

    assert (private / c06._TEST_FILE_PATH).read_text() == c06._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c06._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c06.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
