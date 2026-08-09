"""
Adversarial offline checks for codebench task c33 (dijkstra-shortest-paths) -- no Docker, no LLM.

Mirrors idea_code_test_c01_test.py / idea_code_test_c06_test.py: prove the task module's own
claims are internally consistent (ground truth is actually correct, keystone ids reference
real tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox.
"""
from __future__ import annotations

import heapq
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from agent.app.idea_code_tests import test_c33_dijkstra_shortest_paths as c33


def _independent_shortest_paths(graph: dict, start: str) -> dict:
    """Reimplemented independently of the task's own prose spec (plain heapq-based Dijkstra,
    written from scratch here rather than copied), to catch a mistake in the embedded
    canonical test file's own expected literals."""
    for node, edges in graph.items():
        for _, weight in edges:
            if weight < 0:
                raise ValueError(f"negative edge weight from {node!r}")

    dist = {start: 0}
    finalized = set()
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if u in finalized:
            continue
        finalized.add(u)
        for v, w in graph.get(u, []):
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def test_ground_truth_values_are_internally_correct():
    cases = [
        ({"a": [("b", 1), ("c", 10)], "b": [("c", 2)], "c": []}, "a", {"a": 0, "b": 1, "c": 3}),
        (
            {"a": [("b", 1), ("c", 4)], "b": [("c", 1), ("d", 5)], "c": [("d", 1)], "d": []},
            "a",
            {"a": 0, "b": 1, "c": 2, "d": 3},
        ),
        ({"a": [("b", 2)], "b": [], "c": [("d", 1)], "d": []}, "a", {"a": 0, "b": 2}),
        (
            {"a": [("b", 1)], "b": [("c", 1)], "c": [("a", 1), ("d", 5)], "d": []},
            "a",
            {"a": 0, "b": 1, "c": 2, "d": 7},
        ),
        (
            {
                "a": [("b", 5), ("c", 1)],
                "b": [("d", 1)],
                "c": [("b", 1), ("d", 4)],
                "d": [],
            },
            "a",
            {"a": 0, "b": 2, "c": 1, "d": 3},
        ),
        ({"a": [("a", 5), ("b", 2)], "b": []}, "a", {"a": 0, "b": 2}),
        ({"a": [("b", 1)]}, "b", {"b": 0}),
    ]
    for graph, start, expected in cases:
        assert _independent_shortest_paths(graph, start) == expected, (graph, start)

    with pytest.raises(ValueError):
        _independent_shortest_paths({"a": [("b", -3)], "b": []}, "a")


def test_embedded_test_file_literals_match_ground_truth():
    # Each (graph-literal-substring, expected-literal-substring) pair below must appear
    # verbatim in the embedded fixture, AND the independent reimplementation must agree with
    # the expected value -- tying the fixture's literal text directly to ground truth rather
    # than trusting the fixture's own prose.
    content = c33.get_grading_payload()["tests"][c33._TEST_FILE_PATH]
    checks = [
        (
            {"a": [("b", 1), ("c", 10)], "b": [("c", 2)], "c": []},
            "a",
            '{"a": 0, "b": 1, "c": 3}',
        ),
        (
            {"a": [("b", 1), ("c", 4)], "b": [("c", 1), ("d", 5)], "c": [("d", 1)], "d": []},
            "a",
            '{"a": 0, "b": 1, "c": 2, "d": 3}',
        ),
        ({"a": [("b", 2)], "b": [], "c": [("d", 1)], "d": []}, "a", '{"a": 0, "b": 2}'),
        (
            {"a": [("b", 1)], "b": [("c", 1)], "c": [("a", 1), ("d", 5)], "d": []},
            "a",
            '{"a": 0, "b": 1, "c": 2, "d": 7}',
        ),
        (
            {
                "a": [("b", 5), ("c", 1)],
                "b": [("d", 1)],
                "c": [("b", 1), ("d", 4)],
                "d": [],
            },
            "a",
            '{"a": 0, "b": 2, "c": 1, "d": 3}',
        ),
        ({"a": [("a", 5), ("b", 2)], "b": []}, "a", '{"a": 0, "b": 2}'),
        ({"a": [("b", 1)]}, "b", '{"b": 0}'),
    ]
    for graph, start, expected_literal in checks:
        assert expected_literal in content, expected_literal
        # `expected_literal` is a valid Python dict literal (double-quoted keys are legal
        # Python syntax too), so it can be evaluated directly -- no quote-juggling needed.
        assert _independent_shortest_paths(graph, start) == eval(expected_literal)  # noqa: S307


def test_negative_weight_case_present_and_independently_raises():
    content = c33.get_grading_payload()["tests"][c33._TEST_FILE_PATH]
    assert "pytest.raises(ValueError)" in content
    assert '[("b", -3)]' in content
    with pytest.raises(ValueError):
        _independent_shortest_paths({"a": [("b", -3)], "b": []}, "a")


def test_keystone_ids_reference_real_test_functions():
    content = c33.get_grading_payload()["tests"][c33._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c33.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c33._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_the_easier_edge_cases():
    easy = [
        "test_two_hop_path_beats_a_pricier_direct_edge",
        "test_self_loop_does_not_worsen_the_start_distance",
        "test_start_node_referenced_only_as_a_neighbor_value",
    ]
    for name in easy:
        assert f"{c33._TEST_FILE_PATH}::{name}" not in c33.KEYSTONE_TEST_IDS


def test_keystone_includes_the_sharp_relaxation_case():
    # This is the single most discriminating case (a naive "first distance wins"
    # implementation gets it wrong) so it must gate the score.
    assert (
        f"{c33._TEST_FILE_PATH}::test_relaxation_beats_a_cheaper_looking_direct_edge"
        in c33.KEYSTONE_TEST_IDS
    )


def test_visibility_is_hidden():
    assert c33.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c33.get_sandbox_fixture() == {}


def test_grading_payload_shape():
    payload = c33.get_grading_payload()
    assert payload["tests"] == {c33._TEST_FILE_PATH: c33._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {"module": "shortest_paths", "functions": ["shortest_paths"]}
    assert payload["keystone_test_ids"] == c33.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c33.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "shortest_paths.py" in leaf["instruction"]
    assert "heapq" in leaf["instruction"] or "priority" in leaf["instruction"]
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["shortest_paths.py"]}
    # Must be JSON-serializable as-is -- see c01's identical check for why.
    json.dumps(plan)


def test_compiled_plan_leaks_no_canonical_literal_values():
    # The plan is public; it must describe the ALGORITHM, not embed the private test's
    # expected numeric answers.
    plan = c33.get_compiled_plan()
    instruction = plan["leaves"][0]["instruction"]
    for leaking_literal in ["{'a': 0, 'b': 1, 'c': 3}", '"a": 0, "b": 1, "c": 3']:
        assert leaking_literal not in instruction


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c33", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c33"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c33.get_task_statement()
    # Hidden task: no starter files under public/repo/ at all.
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c33.get_compiled_plan()

    assert (private / c33._TEST_FILE_PATH).read_text() == c33._TEST_FILE_CONTENT
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c33._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c33.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
