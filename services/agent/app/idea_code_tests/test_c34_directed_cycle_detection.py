"""
codebench task c34 — hard/hidden, directed-graph cycle detection.

Graph-algorithms coverage beyond c06 (topo-sort, which only implicitly detects a cycle by
raising when Kahn's algorithm stalls) and c33 (weighted shortest paths): this task is a
direct, standalone boolean cycle check on an UNWEIGHTED directed graph, and is deliberately
built to punish the single most common real bug in a hand-rolled DFS cycle detector -- using
one flat "visited" set instead of proper three-color (white/gray/black) state, which makes a
DAG with a shared descendant (a diamond shape) look like it has a cycle even though it
doesn't. Ground truth for ``has_cycle`` verified with an independent throwaway white/gray/black
DFS reference script (run against every case below and cross-checked) before being embedded
here -- do not hand-edit these literals without re-deriving them; see
idea_code_test_c34_test.py for the reimplementation that re-verifies this at test time.

Graph representation: ``graph`` maps each node name (str) to a list of successor names (str)
describing its OUTGOING directed edges. A successor name may or may not also appear as its own
key in ``graph`` (mirrors c06/c33's "referenced only as a value" convention) -- if it doesn't,
treat it as having zero outgoing edges of its own.

Verified cases (graph -> expected ``has_cycle`` return value):
    {"a": ["b"], "b": ["c"], "c": []}                          -> False (linear chain)
    {"a": ["b","c"], "b": ["d"], "c": ["d"], "d": []}           -> False (diamond -- shared
        descendant "d" reached via two paths; NOT a cycle, this is the sharpest false-positive
        trap for a naive single-visited-set implementation)
    {"a": ["b"], "b": ["c"], "c": []} with "a","z" disconnected -> see disconnected cases below
    {"a": ["b"], "b": ["c"], "c": ["a"]}                        -> True (simple 3-cycle)
    {"a": ["b"], "b": ["a"]}                                    -> True (2-cycle)
    {"a": ["a"], "b": []}                                       -> True (self-loop)
    {"a": ["b"], "b": ["c"], "c": ["d"], "d": ["b"]}            -> True (DAG-except-one-back-
        edge: a->b->c->d is a clean chain, but d->b closes a cycle among b/c/d)
    {"a": ["b"], "b": ["a"], "c": ["d"], "d": []}               -> True (disconnected: one
        component (a/b) is cyclic, the other (c/d) is not -- must check every component, not
        just the first DFS root)
    {"a": ["b"], "c": ["d"]}                                    -> False (disconnected, neither
        component has a cycle)
    {"a": ["b"]}                                                -> False ("b" never appears as
        its own key -- zero outgoing edges, no cycle)
    {"a": ["z"], "b": ["z"], "c": ["z"], "z": []}               -> False (wide fan-in DAG)
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_cycle_detect.py"

_TEST_FILE_CONTENT = '''\
from cycle_detect import has_cycle


def test_linear_chain_has_no_cycle():
    assert has_cycle({"a": ["b"], "b": ["c"], "c": []}) is False


def test_diamond_shared_descendant_is_not_a_cycle():
    graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    assert has_cycle(graph) is False


def test_simple_three_node_cycle():
    assert has_cycle({"a": ["b"], "b": ["c"], "c": ["a"]}) is True


def test_two_node_cycle():
    assert has_cycle({"a": ["b"], "b": ["a"]}) is True


def test_self_loop_is_a_cycle():
    assert has_cycle({"a": ["a"], "b": []}) is True


def test_dag_except_one_back_edge():
    graph = {"a": ["b"], "b": ["c"], "c": ["d"], "d": ["b"]}
    assert has_cycle(graph) is True


def test_disconnected_component_with_a_cycle_is_detected():
    graph = {"a": ["b"], "b": ["a"], "c": ["d"], "d": []}
    assert has_cycle(graph) is True


def test_disconnected_components_with_no_cycle():
    assert has_cycle({"a": ["b"], "c": ["d"]}) is False


def test_node_referenced_only_as_a_value_has_no_cycle():
    assert has_cycle({"a": ["b"]}) is False


def test_wide_fanin_dag_has_no_cycle():
    graph = {"a": ["z"], "b": ["z"], "c": ["z"], "z": []}
    assert has_cycle(graph) is False
'''

# Balanced across both answers on purpose, so neither an "always return True" nor an "always
# return False" implementation can coast: the diamond and disconnected-no-cycle cases are the
# False keystones (the diamond specifically punishes a flat-visited-set false positive), while
# the three-cycle, two-cycle, self-loop, back-edge, and disconnected-with-a-cycle cases are the
# True keystones (self-loop and the DAG-except-one-back-edge case are the ones a lazy or
# incomplete DFS is most likely to miss). The plain linear chain, the value-only-referenced
# node, and the wide fan-in DAG are real but comparatively easy edge cases and are bonus
# credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_diamond_shared_descendant_is_not_a_cycle",
    f"{_TEST_FILE_PATH}::test_disconnected_components_with_no_cycle",
    f"{_TEST_FILE_PATH}::test_simple_three_node_cycle",
    f"{_TEST_FILE_PATH}::test_two_node_cycle",
    f"{_TEST_FILE_PATH}::test_self_loop_is_a_cycle",
    f"{_TEST_FILE_PATH}::test_dag_except_one_back_edge",
    f"{_TEST_FILE_PATH}::test_disconnected_component_with_a_cycle_is_detected",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c34",
        "title": "directed-cycle-detection",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `cycle_detect.py` that defines a function "
        "`has_cycle(graph: dict) -> bool`.\n\n"
        "`graph` maps each node name (str) to a list of successor names (str) describing "
        "that node's OUTGOING directed edges, e.g. `{\"a\": [\"b\", \"c\"], \"b\": [\"d\"], "
        "\"c\": [\"d\"], \"d\": []}`. A successor name may or may not also appear as its own "
        "key in `graph` -- if it doesn't, treat it as having zero outgoing edges of its "
        "own.\n\n"
        "`has_cycle(graph)` must return `True` if the directed graph contains AT LEAST ONE "
        "cycle (a path of one or more edges that starts and ends at the same node -- this "
        "includes a self-loop, a node with an edge directly back to itself), and `False` "
        "otherwise. The graph may be disconnected (several separate components); a cycle "
        "anywhere in ANY component must be detected, not just in whichever part of the "
        "graph a simple check happens to reach first.\n\n"
        "A node being reachable via two DIFFERENT paths from a common ancestor is NOT, by "
        "itself, a cycle. For example, `has_cycle({\"a\": [\"b\", \"c\"], \"b\": [\"d\"], "
        "\"c\": [\"d\"], \"d\": []})` must return `False`: both `b` and `c` lead to `d`, but "
        "no directed path ever returns to a node it already passed through. Get this case "
        "exactly right -- it is the sharpest test of whether your cycle check is actually "
        "sound or just happens to look plausible on simpler inputs.\n\n"
        "No test file is visible for this task. Write cycle_detect.py implementing has_cycle "
        "exactly as specified above, then use run_python to sanity-check it yourself against "
        "the shared-descendant example above (must be False), a simple 3-node cycle and a "
        "self-loop (both must be True), and a disconnected graph where only one of several "
        "components is cyclic (must be True), before finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test -- the agent works from the spec alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "cycle_detect", "functions": ["has_cycle"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Hand-authored offline plan -- a single leaf is enough for a one-function task."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write cycle_detect.py implementing has_cycle(graph: dict) -> bool: "
                    "`graph` maps node name -> list of successor names for that node's "
                    "outgoing directed edges (a successor need not be its own key -- treat it "
                    "as having no outgoing edges if it isn't). `has_cycle` must return True iff "
                    "the directed graph contains at least one real cycle (a path of edges that "
                    "loops back to a node still on that same path, including a self-loop), and "
                    "False otherwise -- across EVERY component if the graph is disconnected, "
                    "not just the first one you happen to explore. A node reached twice via two "
                    "DIFFERENT, unrelated paths (e.g. two branches converging on a shared "
                    "descendant, like a<-b<-shared and a<-c<-shared) is NOT a cycle by itself; "
                    "only actually looping back to a node still on your CURRENT traversal path "
                    "counts. Get this distinction exactly right -- an implementation that just "
                    "tracks 'have I ever seen this node before, period' (one flat set, no notion "
                    "of 'currently on this path' vs 'already fully explored elsewhere') will "
                    "wrongly flag ordinary converging paths as cycles. Use write_file to create "
                    "it, then use run_python to sanity-check it against a shared-descendant "
                    "shape (must be False), a simple cycle and a self-loop (both True), and a "
                    "disconnected graph where only one component is cyclic (must be True), "
                    "before finishing."
                ),
                "expect": "cycle_detect.py written implementing has_cycle that correctly rejects "
                          "converging-path false positives and checks every component of a "
                          "disconnected graph",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm cycle_detect.py exists and defines has_cycle.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["cycle_detect.py"]},
    }
