"""
Adversarial offline checks for codebench task c35 (connected-components-union-find) -- no
Docker, no LLM.

Mirrors idea_code_test_c01_test.py / idea_code_test_c09_test.py: prove the task module's own
claims are internally consistent (ground truth is actually correct, keystone ids reference
real tests, the compiled plan is well-formed) BEFORE anything ever reaches a live sandbox.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from agent.app.idea_code_tests import test_c35_connected_components_union_find as c35


def _independent_connected_components(nodes: list, edges: list) -> list:
    """Reimplemented independently of the task's own prose spec (plain union-find with path
    compression, written from scratch here rather than copied), to catch a mistake in the
    embedded canonical test file's own expected literals. Deliberately dict-keyed by node
    NAME (not by list position), so a duplicate name in `nodes` naturally collapses to one
    union-find element -- iterating `parent`'s own keys for the final grouping (rather than
    the raw, possibly-duplicated `nodes` list) is what makes the dedup explicit here."""
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for u, v in edges:
        union(u, v)

    groups: dict = {}
    for n in parent:  # iterate distinct names only, not the raw (possibly duplicated) nodes list
        groups.setdefault(find(n), []).append(n)
    components = [sorted(g) for g in groups.values()]
    components.sort(key=lambda c: c[0])
    return components


def test_ground_truth_values_are_internally_correct():
    cases = [
        (["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d")], [["a", "b", "c", "d"]]),
        (
            ["a", "b", "c", "d", "e"],
            [("a", "b"), ("c", "d")],
            [["a", "b"], ["c", "d"], ["e"]],
        ),
        (["a", "b", "c"], [("a", "b"), ("a", "b"), ("b", "a")], [["a", "b"], ["c"]]),
        (["a", "b"], [("a", "a")], [["a"], ["b"]]),
        (
            ["a", "b", "c", "d", "e", "f"],
            [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f")],
            [["a", "b", "c"], ["d", "e", "f"]],
        ),
        (["a", "b", "c"], [], [["a"], ["b"], ["c"]]),
        (
            ["a", "b", "c", "d", "e"],
            [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e")],
            [["a", "b", "c", "d", "e"]],
        ),
        (["z", "a", "m"], [], [["a"], ["m"], ["z"]]),
        (["d", "c", "b", "a"], [("a", "b"), ("c", "d")], [["a", "b"], ["c", "d"]]),
        (["c", "a", "b", "a"], [("a", "c")], [["a", "c"], ["b"]]),
    ]
    for nodes, edges, expected in cases:
        assert _independent_connected_components(nodes, edges) == expected, (nodes, edges)


def _independent_connected_components_deduping_nodes(nodes: list, edges: list) -> list:
    """Same reimplementation, but explicit about deduping `nodes` first -- proves the
    dedup behavior isn't an accident of dict key collision in the plain reimplementation
    above (which happens to already dedupe because it keys `parent` by name)."""
    seen = []
    for n in nodes:
        if n not in seen:
            seen.append(n)
    return _independent_connected_components(seen, edges)


def test_duplicate_node_ground_truth_matches_explicit_dedup_variant():
    nodes = ["c", "a", "b", "a"]
    edges = [("a", "c")]
    expected = [["a", "c"], ["b"]]
    assert _independent_connected_components(nodes, edges) == expected
    assert _independent_connected_components_deduping_nodes(nodes, edges) == expected


def test_a_positional_index_based_union_find_gets_the_duplicate_case_wrong():
    """Prove the duplicate-node keystone is discriminating: a plausible-looking union-find
    that assigns one array slot per `nodes` LIST INDEX (rather than one slot per distinct
    node name) treats the two occurrences of "a" as different elements."""

    def buggy_positional(nodes, edges):
        index_of_name = {name: i for i, name in enumerate(nodes)}  # last occurrence wins
        parent = list(range(len(nodes)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for u, v in edges:
            union(index_of_name[u], index_of_name[v])

        groups: dict = {}
        for i, name in enumerate(nodes):
            groups.setdefault(find(i), []).append(name)
        components = [sorted(g) for g in groups.values()]
        components.sort(key=lambda c: c[0])
        return components

    nodes = ["c", "a", "b", "a"]
    edges = [("a", "c")]
    correct = [["a", "c"], ["b"]]
    buggy_result = buggy_positional(nodes, edges)
    assert buggy_result != correct, (
        "expected the positional-index bug to diverge from the correct answer -- if this "
        "fails, the duplicate-node case no longer discriminates that bug class"
    )


def test_long_chain_ground_truth_is_one_sorted_component():
    n = 1300
    nodes = [f"n{i:04d}" for i in range(n)]
    edges = [(f"n{i:04d}", f"n{i + 1:04d}") for i in range(n - 1)]
    result = _independent_connected_components(nodes, edges)
    assert len(result) == 1
    assert result[0] == sorted(nodes)


def test_a_naive_recursive_find_overflows_on_the_long_chain():
    """Prove the long-chain keystone is discriminating: a plain recursive find() with no
    explicit stack -- a common, individually-reasonable-looking way to write path
    compression -- blows Python's default recursion limit on this exact chain length."""

    def buggy_recursive(nodes, edges):
        parent = {n: n for n in nodes}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for u, v in edges:
            union(u, v)

        groups: dict = {}
        for n in nodes:
            groups.setdefault(find(n), []).append(n)
        components = [sorted(g) for g in groups.values()]
        components.sort(key=lambda c: c[0])
        return components

    n = 1300
    nodes = [f"n{i:04d}" for i in range(n)]
    edges = [(f"n{i:04d}", f"n{i + 1:04d}") for i in range(n - 1)]
    raised = False
    try:
        buggy_recursive(nodes, edges)
    except RecursionError:
        raised = True
    assert raised, (
        "expected the naive recursive find() to overflow on a 1300-node chain -- if this no "
        "longer raises, the long-chain keystone no longer discriminates that bug class"
    )


def test_reference_checker_itself_rejects_an_unsorted_or_wrong_partition():
    # Sanity-check that the independent reference actually enforces sort order (not just
    # partition correctness) -- a component list in insertion order rather than sorted-by-
    # smallest-element order must differ from what the reference produces.
    got = _independent_connected_components(["z", "a", "m"], [])
    assert got == [["a"], ["m"], ["z"]]
    assert got != [["z"], ["a"], ["m"]]


def test_embedded_test_file_contains_every_expected_literal():
    content = c35.get_grading_payload()["tests"][c35._TEST_FILE_PATH]
    literals = [
        '[["a", "b", "c", "d"]]',
        '[["a", "b"], ["c", "d"], ["e"]]',
        '[["a", "b"], ["c"]]',
        '[["a"], ["b"]]',
        '[["a", "b", "c"], ["d", "e", "f"]]',
        '[["a"], ["b"], ["c"]]',
        '[["a", "b", "c", "d", "e"]]',
        '[["a"], ["m"], ["z"]]',
        '[["a", "b"], ["c", "d"]]',
        '[["a", "c"], ["b"]]',
    ]
    for literal in literals:
        assert literal in content, literal


def test_embedded_test_file_contains_the_long_chain_stress_case():
    content = c35.get_grading_payload()["tests"][c35._TEST_FILE_PATH]
    assert "n = 1300" in content
    assert "sorted(nodes)" in content
    assert "test_long_chain_of_many_nodes_merges_without_error" in content


def test_keystone_ids_reference_real_test_functions():
    content = c35.get_grading_payload()["tests"][c35._TEST_FILE_PATH]
    defined = set(re.findall(r"^def (test_\w+)\(", content, re.MULTILINE))
    for node_id in c35.KEYSTONE_TEST_IDS:
        path, _, func = node_id.partition("::")
        assert path == c35._TEST_FILE_PATH, node_id
        assert func in defined, f"keystone id {node_id!r} has no matching def in the fixture"


def test_keystone_excludes_the_easier_edge_cases():
    easy = [
        "test_self_loop_edge_does_not_misbehave",
        "test_no_edges_all_singletons",
        "test_star_shape_merges_every_spoke",
    ]
    for name in easy:
        assert f"{c35._TEST_FILE_PATH}::{name}" not in c35.KEYSTONE_TEST_IDS


def test_keystone_includes_the_hardened_2026_08_06_cases():
    # The duplicate-node-identity case and the long-chain stress case are the two cases added
    # after live calibration showed a strong coding agent acing the previous, easier version
    # of this task -- both must gate the score, not sit as bonus credit.
    assert (
        f"{c35._TEST_FILE_PATH}::test_duplicate_node_names_are_deduplicated_in_output"
        in c35.KEYSTONE_TEST_IDS
    )
    assert (
        f"{c35._TEST_FILE_PATH}::test_long_chain_of_many_nodes_merges_without_error"
        in c35.KEYSTONE_TEST_IDS
    )


def test_task_statement_does_not_leak_the_exact_hidden_chain_length():
    # The task statement may (and does) warn generically about "thousands of nodes" and
    # "1000+" nodes so the requirement is fair, but the exact hidden chain length (1300) used
    # by the canonical stress test must not appear verbatim in agent-visible prose.
    statement = c35.get_task_statement()
    assert "1300" not in statement


def test_keystone_includes_both_ordering_contract_cases():
    # The output-ordering contract (sorted inner lists, outer list sorted by first element)
    # is the sharpest, most implementation-specific part of the spec -- both cases that
    # exercise it (unsorted `nodes` input; reverse-ordered `nodes` input across multiple
    # components) must gate the score.
    assert (
        f"{c35._TEST_FILE_PATH}::test_output_is_sorted_even_when_input_nodes_are_not"
        in c35.KEYSTONE_TEST_IDS
    )
    assert (
        f"{c35._TEST_FILE_PATH}::test_components_are_ordered_by_their_smallest_element"
        in c35.KEYSTONE_TEST_IDS
    )


def test_visibility_is_hidden():
    assert c35.get_visibility() == "hidden"


def test_hidden_task_ships_no_starter_files():
    assert c35.get_sandbox_fixture() == {}


def test_task_statement_worked_example_does_not_leak_a_keystone_literal():
    # The task statement gives a worked example of the ordering contract so the spec is
    # fair, but it must use DIFFERENT node names than any embedded keystone assertion --
    # otherwise the model gets one keystone's exact answer for free.
    statement = c35.get_task_statement()
    assert '["p", "k"]' in statement
    assert '[["k"], ["p"]]' in statement
    content = c35.get_grading_payload()["tests"][c35._TEST_FILE_PATH]
    assert '"p"' not in content and '"k"' not in content


def test_grading_payload_shape():
    payload = c35.get_grading_payload()
    assert payload["tests"] == {c35._TEST_FILE_PATH: c35._TEST_FILE_CONTENT}
    assert payload["entrypoint"] == {
        "module": "connected_components",
        "functions": ["connected_components"],
    }
    assert payload["keystone_test_ids"] == c35.KEYSTONE_TEST_IDS


def test_compiled_plan_structure():
    plan = c35.get_compiled_plan()
    assert [leaf["id"] for leaf in plan["leaves"]] == ["implement"]
    leaf = plan["leaves"][0]
    assert leaf["depends_on"] == []
    assert "connected_components.py" in leaf["instruction"]
    assert "union" in leaf["instruction"].lower()
    assert plan["agg_mode"] == "sandbox_submit"
    assert plan["composition"] == {"op": "submit_files", "files": ["connected_components.py"]}
    # Must be JSON-serializable as-is -- see c01's identical check for why.
    json.dumps(plan)


def test_materialize_task_end_to_end(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "badmodel-lab" / "codebench" / "materialize_task.py"
    assert script.exists(), script

    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(script), "c35", "--out", str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr

    task_dir = tmp_path / "c35"
    public, private = task_dir / "public", task_dir / "private"

    assert (public / "prompt.md").read_text() == c35.get_task_statement()
    # Hidden task: no starter files under public/repo/ at all.
    assert list((public / "repo").iterdir()) == []
    assert json.loads((public / "plan.json").read_text()) == c35.get_compiled_plan()

    assert (
        private / c35._TEST_FILE_PATH
    ).read_text() == c35.get_grading_payload()["tests"][c35._TEST_FILE_PATH]
    manifest = json.loads((private / "test_manifest.json").read_text())
    assert c35._TEST_FILE_PATH in manifest["test_file_globs"]

    meta = json.loads((task_dir / "meta.json").read_text())
    assert meta["category"] == "hard"
    assert meta["visibility"] == "hidden"
    assert meta["keystone_test_ids"] == c35.KEYSTONE_TEST_IDS
    assert meta["has_rubric"] is False
    assert not (task_dir / "rubric.json").exists()
