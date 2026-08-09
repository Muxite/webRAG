"""
codebench task c33 — hard/hidden, weighted-graph shortest paths (Dijkstra).

Graph-algorithms coverage beyond c06 (topo-sort, unweighted directed DAG traversal): this
task is a WEIGHTED directed graph and the correct algorithm requires priority-based
relaxation, not just a traversal order. Ground truth for ``shortest_paths`` verified with an
independent throwaway heapq-based Dijkstra reference script (run against every case below and
cross-checked) before being embedded here — do not hand-edit these literals without
re-deriving them; see idea_code_test_c33_test.py for the reimplementation that re-verifies
this at test time.

Graph representation: ``graph`` maps each node name (str) to a list of
``(neighbor, weight)`` tuples describing its OUTGOING directed edges. A neighbor name may or
may not also appear as its own key in ``graph`` (mirrors c06's "dependency referenced only as
a value" convention) — if it doesn't, it is treated as having zero outgoing edges.

Verified cases (graph, start -> expected ``shortest_paths`` return value):
    {"a": [("b",1), ("c",10)], "b": [("c",2)], "c": []}, "a"
        -> {"a": 0, "b": 1, "c": 3}   (a->b->c=3 beats the direct a->c=10 edge)
    {"a": [("b",1),("c",4)], "b": [("c",1),("d",5)], "c": [("d",1)], "d": []}, "a"
        -> {"a": 0, "b": 1, "c": 2, "d": 3}   (diamond; c relaxed via b, d relaxed via c)
    {"a": [("b",2)], "b": [], "c": [("d",1)], "d": []}, "a"
        -> {"a": 0, "b": 2}   (c and d are unreachable from a and must NOT appear in the result)
    {"a": [("b",1)], "b": [("c",1)], "c": [("a",1),("d",5)], "d": []}, "a"
        -> {"a": 0, "b": 1, "c": 2, "d": 7}   (cycle a->b->c->a: the c->a edge (cost 3) never
           beats a's own distance of 0, so it must not corrupt the result)
    {"a": [("b",5),("c",1)], "b": [("d",1)], "c": [("b",1),("d",4)], "d": []}, "a"
        -> {"a": 0, "b": 2, "c": 1, "d": 3}   (KEY relaxation case: a naive "first distance
           assigned wins" implementation would wrongly freeze b=5 and d=6 instead of
           discovering the cheaper a->c->b=2 and a->c->b->d=3 paths)
    {"a": [("a",5), ("b",2)], "b": []}, "a"
        -> {"a": 0, "b": 2}   (a self-loop must never make a node's own distance worse)
    {"a": [("b",1)]}, "b"
        -> {"b": 0}   (start node never appears as its own key -> zero outgoing edges; "a" is
           NOT reachable from "b" since the only edge is directed a->b)
    {"a": [("b",-3)], "b": []}, "a" -> raises ValueError (Dijkstra is undefined for negative
        edge weights; the implementation must detect and reject this rather than silently
        returning a wrong answer)
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_shortest_paths.py"

_TEST_FILE_CONTENT = '''\
import pytest
from shortest_paths import shortest_paths


def test_two_hop_path_beats_a_pricier_direct_edge():
    graph = {"a": [("b", 1), ("c", 10)], "b": [("c", 2)], "c": []}
    assert shortest_paths(graph, "a") == {"a": 0, "b": 1, "c": 3}


def test_diamond_shape_relaxes_through_the_shorter_side():
    graph = {
        "a": [("b", 1), ("c", 4)],
        "b": [("c", 1), ("d", 5)],
        "c": [("d", 1)],
        "d": [],
    }
    assert shortest_paths(graph, "a") == {"a": 0, "b": 1, "c": 2, "d": 3}


def test_unreachable_nodes_are_excluded_from_the_result():
    graph = {"a": [("b", 2)], "b": [], "c": [("d", 1)], "d": []}
    assert shortest_paths(graph, "a") == {"a": 0, "b": 2}


def test_cycle_back_to_start_does_not_corrupt_the_result():
    graph = {"a": [("b", 1)], "b": [("c", 1)], "c": [("a", 1), ("d", 5)], "d": []}
    assert shortest_paths(graph, "a") == {"a": 0, "b": 1, "c": 2, "d": 7}


def test_relaxation_beats_a_cheaper_looking_direct_edge():
    graph = {
        "a": [("b", 5), ("c", 1)],
        "b": [("d", 1)],
        "c": [("b", 1), ("d", 4)],
        "d": [],
    }
    assert shortest_paths(graph, "a") == {"a": 0, "b": 2, "c": 1, "d": 3}


def test_self_loop_does_not_worsen_the_start_distance():
    graph = {"a": [("a", 5), ("b", 2)], "b": []}
    assert shortest_paths(graph, "a") == {"a": 0, "b": 2}


def test_start_node_referenced_only_as_a_neighbor_value():
    graph = {"a": [("b", 1)]}
    assert shortest_paths(graph, "b") == {"b": 0}


def test_negative_weight_raises_value_error():
    graph = {"a": [("b", -3)], "b": []}
    with pytest.raises(ValueError):
        shortest_paths(graph, "a")
'''

# The diamond (real relaxation across two candidate paths), the unreachable-exclusion case
# (catches a "return every node" bug), the cycle case (catches an infinite loop or a
# cycle-corrupted distance), the sharp relaxation case (catches "first distance assigned
# wins" instead of true priority-based relaxation -- the single most discriminating case),
# and the negative-weight contract (catches missing input validation) gate the score. The
# plain two-hop case, the self-loop, and the start-referenced-only-as-a-value case are real
# but comparatively easy edge cases and are bonus credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_diamond_shape_relaxes_through_the_shorter_side",
    f"{_TEST_FILE_PATH}::test_unreachable_nodes_are_excluded_from_the_result",
    f"{_TEST_FILE_PATH}::test_cycle_back_to_start_does_not_corrupt_the_result",
    f"{_TEST_FILE_PATH}::test_relaxation_beats_a_cheaper_looking_direct_edge",
    f"{_TEST_FILE_PATH}::test_negative_weight_raises_value_error",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c33",
        "title": "dijkstra-shortest-paths",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `shortest_paths.py` that defines a function "
        "`shortest_paths(graph: dict, start: str) -> dict`.\n\n"
        "`graph` maps each node name (str) to a list of `(neighbor, weight)` tuples "
        "describing that node's OUTGOING directed edges, e.g. "
        "`{\"a\": [(\"b\", 1), (\"c\", 10)], \"b\": [(\"c\", 2)], \"c\": []}`. Every edge "
        "weight is a non-negative number. A neighbor name may or may not also appear as its "
        "own key in `graph` — if it doesn't, treat it as having zero outgoing edges of its "
        "own (it's a leaf/sink).\n\n"
        "`shortest_paths(graph, start)` must run Dijkstra's algorithm and return a dict "
        "mapping every node REACHABLE from `start` (including `start` itself, at distance "
        "0) to the total weight of the shortest path from `start` to that node. Nodes that "
        "are NOT reachable from `start` must be entirely absent from the returned dict — do "
        "not include them with a distance of `None` or `float('inf')`.\n\n"
        "The graph may contain cycles; your implementation must handle that correctly "
        "(never loop forever, never let a cycle edge worsen an already-finalized distance). "
        "It may also contain self-loop edges (a node with an edge to itself); these must "
        "never make that node's own distance worse than 0.\n\n"
        "If ANY edge weight in `graph` is negative, raise `ValueError` instead of computing "
        "anything — Dijkstra's algorithm is not valid on graphs with negative edge weights, "
        "so this must be detected and rejected rather than silently returning a wrong "
        "answer.\n\n"
        "Worked example showing why real priority-based relaxation matters (not just 'first "
        "distance found wins'): for "
        "`{\"a\": [(\"b\", 5), (\"c\", 1)], \"b\": [(\"d\", 1)], \"c\": [(\"b\", 1), "
        "(\"d\", 4)], \"d\": []}`, `shortest_paths(graph, \"a\")` must return "
        "`{\"a\": 0, \"b\": 2, \"c\": 1, \"d\": 3}` — even though the direct edge a->b costs "
        "5, the path a->c->b (cost 1+1=2) is cheaper and must win; similarly a->c->b->d "
        "(cost 3) beats both a->c->d (cost 5) and a->b->d (cost 6).\n\n"
        "No test file is visible for this task. Write shortest_paths.py implementing "
        "shortest_paths exactly as specified above, then use run_python to sanity-check it "
        "yourself against the worked example above and a couple of your own small graphs "
        "(including one with a cycle and one where you deliberately try a negative weight to "
        "confirm ValueError is raised) before finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test -- the agent works from the spec alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "shortest_paths", "functions": ["shortest_paths"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Hand-authored offline plan -- a single leaf is enough for a one-function task."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write shortest_paths.py implementing shortest_paths(graph, start): run "
                    "Dijkstra's algorithm over `graph`, a dict mapping node name -> list of "
                    "(neighbor, weight) tuples for that node's outgoing directed edges (a "
                    "neighbor need not be its own key -- treat it as having no outgoing edges "
                    "if it isn't). Use a min-priority-queue-based relaxation loop (e.g. "
                    "heapq): repeatedly pop the not-yet-finalized node with the smallest "
                    "tentative distance, finalize it, and relax each of its outgoing edges "
                    "(update a neighbor's tentative distance if this path is cheaper than any "
                    "found so far). Return a dict of node -> shortest distance for every node "
                    "reachable from `start` (start itself included at distance 0); nodes not "
                    "reachable from start must be entirely absent from the returned dict. "
                    "Handle cycles and self-loop edges correctly (they must never worsen an "
                    "already-finalized distance or cause an infinite loop). If any edge weight "
                    "in `graph` is negative, raise ValueError instead of computing a result. "
                    "Use write_file to create it, then use run_python to sanity-check it "
                    "against a few small hand-built graphs (including one with a cycle, and "
                    "one where a naive 'first distance wins' approach would give the wrong "
                    "answer but true relaxation gets it right) before finishing."
                ),
                "expect": "shortest_paths.py written implementing shortest_paths via Dijkstra's "
                          "algorithm with correct relaxation, cycle-safety, unreachable-node "
                          "exclusion, and a ValueError on negative weights",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm shortest_paths.py exists and defines shortest_paths.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["shortest_paths.py"]},
    }
