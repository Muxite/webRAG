"""
codebench task c35 — hard/hidden, undirected-graph connected components (union-find).

NOTE on visibility: originally authored as visible (test file shipped as a starter fixture).
Live calibration on 2026-08-06 showed qwen2.5:14b scoring 1.0/1.0 on both agent harnesses --
with the exact expected output sitting in a runnable test file, both harnesses just iterated
against run_pytest until it passed, regardless of whether they understood the sort-order
contract. Flipped to hidden (no starter test file; task statement describes the contract in
prose, using a worked example with DIFFERENT node names than any embedded keystone assertion)
so the model must actually derive and hold onto the ordering rule itself.

Graph-algorithms coverage beyond c06 (directed DAG topo-sort), c33 (directed weighted shortest
paths), and c34 (directed cycle detection): this task is UNDIRECTED partitioning -- given a
set of nodes and an edge list, group nodes into their connected components. To make the
output a single, unambiguous, exactly-comparable value (unlike c06's topo_sort, which needed a
structural checker because many valid orders exist), the required output contract is fully
deterministic: each component is returned as a list of its nodes SORTED ascending, and the
outer list of components is itself sorted by each component's first (smallest) element. Ground
truth for ``connected_components`` verified with an independent throwaway union-find (path
compression + union) reference script (run against every case below and cross-checked) before
being embedded here -- do not hand-edit these literals without re-deriving them; see
idea_code_test_c35_test.py for the reimplementation that re-verifies this at test time.

Verified cases (nodes, edges -> expected ``connected_components`` return value):
    ["a","b","c","d"], [("a","b"),("b","c"),("c","d")]
        -> [["a","b","c","d"]]                       (chain merges across 3 edges into one)
    ["a","b","c","d","e"], [("a","b"),("c","d")]
        -> [["a","b"], ["c","d"], ["e"]]              ("e" isolated, no edges at all)
    ["a","b","c"], [("a","b"),("a","b"),("b","a")]
        -> [["a","b"], ["c"]]                         (duplicate edge + reversed duplicate must
           not be double-counted or break anything)
    ["a","b"], [("a","a")]
        -> [["a"], ["b"]]                             (a self-loop edge must not merge "a" with
           anything or otherwise misbehave)
    ["a","b","c","d","e","f"], [("a","b"),("b","c"),("c","a"),("d","e"),("e","f")]
        -> [["a","b","c"], ["d","e","f"]]             (two separate components, each internally
           forms a cycle -- undirected edges, order within each triangle irrelevant to the
           final partition)
    ["a","b","c"], []
        -> [["a"], ["b"], ["c"]]                      (no edges at all -- every node its own
           singleton component)
    ["a","b","c","d","e"], [("a","b"),("a","c"),("a","d"),("a","e")]
        -> [["a","b","c","d","e"]]                    (star/fan-out -- one hub union-ed with
           four separate partners must merge all five into one component)
    ["z","a","m"], []
        -> [["a"], ["m"], ["z"]]                      (input `nodes` given in non-alphabetical
           order; output must still be sorted by each singleton's own name -- catches an
           "output components in input/insertion order" bug)
    ["d","c","b","a"], [("a","b"),("c","d")]
        -> [["a","b"], ["c","d"]]                     (input `nodes` given in reverse order;
           output must still be [["a","b"], ["c","d"]], sorted by each component's smallest
           element -- catches the same insertion-order bug at the component level)

Hardened 2026-08-06 after live calibration showed a strong coding agent (Aider) acing the
hidden-test version of this task on the first try -- the spec was clear enough that a careful,
correct implementation sailed through. Two genuinely subtler cases were added, verified with
the same independent throwaway reference (re-run and cross-checked, not hand-traced):
    ["c","a","b","a"], [("a","c")]
        -> [["a","c"], ["b"]]                        ("a" is listed TWICE in `nodes` (a
           duplicate/aliased entry) -- it is still exactly one graph node, and must appear
           exactly once in the output, not once per occurrence in the input list. A common
           implementation shape -- building a positional union-find array with one slot per
           `nodes` list index rather than one slot per DISTINCT node name -- treats the two
           occurrences of "a" as two different elements and either duplicates "a" in the
           output or leaves one occurrence un-unioned.)
    a chain of 1300 distinct nodes n0000..n1299 linked edge-by-edge (n0000-n0001, n0001-n0002,
    ..., n1298-n1299) -> one single component containing all 1300 nodes, sorted ascending
        (this is a correctness/robustness stress case, not a hand-traceable one -- verified
        programmatically, not by literal embedding. A `find` implemented as plain recursion
        with no explicit stack -- a very common, individually-reasonable-looking way to write
        path compression -- recurses one Python call per hop while walking an as-yet-
        uncompressed chain; with 1300 hops that exceeds Python's default recursion limit
        (1000) and raises RecursionError on the very first `find` call down the chain, even
        though the exact same code works fine on every small hand-picked example above.)
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_connected_components.py"

_TEST_FILE_CONTENT = '''\
from connected_components import connected_components


def test_chain_of_edges_merges_into_one_component():
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    assert connected_components(nodes, edges) == [["a", "b", "c", "d"]]


def test_two_pairs_plus_an_isolated_node():
    nodes = ["a", "b", "c", "d", "e"]
    edges = [("a", "b"), ("c", "d")]
    assert connected_components(nodes, edges) == [["a", "b"], ["c", "d"], ["e"]]


def test_duplicate_and_reversed_edges_are_not_double_counted():
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("a", "b"), ("b", "a")]
    assert connected_components(nodes, edges) == [["a", "b"], ["c"]]


def test_self_loop_edge_does_not_misbehave():
    nodes = ["a", "b"]
    edges = [("a", "a")]
    assert connected_components(nodes, edges) == [["a"], ["b"]]


def test_two_disjoint_triangles():
    nodes = ["a", "b", "c", "d", "e", "f"]
    edges = [("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f")]
    assert connected_components(nodes, edges) == [["a", "b", "c"], ["d", "e", "f"]]


def test_no_edges_all_singletons():
    assert connected_components(["a", "b", "c"], []) == [["a"], ["b"], ["c"]]


def test_star_shape_merges_every_spoke():
    nodes = ["a", "b", "c", "d", "e"]
    edges = [("a", "b"), ("a", "c"), ("a", "d"), ("a", "e")]
    assert connected_components(nodes, edges) == [["a", "b", "c", "d", "e"]]


def test_output_is_sorted_even_when_input_nodes_are_not():
    assert connected_components(["z", "a", "m"], []) == [["a"], ["m"], ["z"]]


def test_components_are_ordered_by_their_smallest_element():
    nodes = ["d", "c", "b", "a"]
    edges = [("a", "b"), ("c", "d")]
    assert connected_components(nodes, edges) == [["a", "b"], ["c", "d"]]


def test_duplicate_node_names_are_deduplicated_in_output():
    # "a" is listed twice in `nodes` -- it is still exactly one graph node and must appear
    # exactly once in the returned partition, not once per occurrence in the input list.
    nodes = ["c", "a", "b", "a"]
    edges = [("a", "c")]
    assert connected_components(nodes, edges) == [["a", "c"], ["b"]]


def test_long_chain_of_many_nodes_merges_without_error():
    # A stress case a small hand-picked example cannot exercise: 1300 nodes linked in a single
    # long chain must all merge into one component, and the implementation must not error out
    # (e.g. a purely recursive find() with no explicit stack can exceed Python's default
    # recursion limit on a long, as-yet-uncompressed chain).
    n = 1300
    nodes = [f"n{i:04d}" for i in range(n)]
    edges = [(f"n{i:04d}", f"n{i + 1:04d}") for i in range(n - 1)]
    result = connected_components(nodes, edges)
    assert len(result) == 1
    assert result[0] == sorted(nodes)
'''

# The chain merge (real multi-edge union, not a trivial passthrough), the two-pairs-plus-
# isolated partition, the duplicate/reversed-edge dedup case, the two-disjoint-triangles case
# (proves components are kept separate even when each one internally cycles), both
# insertion-order-independence cases (unsorted `nodes` input, and reverse-ordered `nodes`
# input across multiple components), the duplicate-node-identity case, and the long-chain
# stress case gate the score -- together they are exactly the cases a naive or
# partially-correct union-find would get wrong. The self-loop edge case, the all-isolated
# degenerate case, and the star/fan-out case are real but comparatively easy and are bonus
# credit only, not keystone.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_chain_of_edges_merges_into_one_component",
    f"{_TEST_FILE_PATH}::test_two_pairs_plus_an_isolated_node",
    f"{_TEST_FILE_PATH}::test_duplicate_and_reversed_edges_are_not_double_counted",
    f"{_TEST_FILE_PATH}::test_two_disjoint_triangles",
    f"{_TEST_FILE_PATH}::test_output_is_sorted_even_when_input_nodes_are_not",
    f"{_TEST_FILE_PATH}::test_components_are_ordered_by_their_smallest_element",
    f"{_TEST_FILE_PATH}::test_duplicate_node_names_are_deduplicated_in_output",
    f"{_TEST_FILE_PATH}::test_long_chain_of_many_nodes_merges_without_error",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c35",
        "title": "connected-components-union-find",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `connected_components.py` that defines a function "
        "`connected_components(nodes: list, edges: list) -> list`.\n\n"
        "`nodes` is a list of node names (str). `edges` is a list of 2-element "
        "tuples/lists `(u, v)`, each describing one UNDIRECTED edge between two nodes "
        "already present in `nodes` (an edge connects `u` and `v` in both directions; there "
        "is no 'direction' to an edge in this task). `edges` may contain duplicate edges, "
        "the same edge listed with its two endpoints reversed, and self-loop edges (`u == "
        "v`) -- none of these should cause an error or change the result beyond what a "
        "single copy of that edge would do.\n\n"
        "`connected_components(nodes, edges)` must partition `nodes` into its connected "
        "components (maximal groups of nodes that are all reachable from each other via some "
        "path of edges; a node with no edges at all is its own component of size 1) and "
        "return them as a list, with this EXACT, fully deterministic output format:\n"
        "- Each component is a list of its member node names, sorted ascending "
        "(alphabetically).\n"
        "- The outer list of components is itself sorted by each component's first "
        "(smallest, after the inner sort) element.\n\n"
        "This output format is deterministic regardless of the order `nodes` or `edges` are "
        "given in -- e.g. `connected_components([\"p\", \"k\"], [])` must return "
        "`[[\"k\"], [\"p\"]]` even though `nodes` was given in a completely different "
        "order.\n\n"
        "`nodes` may also contain the same node name listed more than once (a duplicate "
        "entry). All occurrences of the same name refer to the SAME graph node -- the "
        "returned partition must list each distinct node name exactly once, never repeated, "
        "even though it may have appeared multiple times in `nodes`.\n\n"
        "Your implementation must also work correctly on much larger inputs than the small "
        "examples above -- e.g. a graph with thousands of nodes chained together edge by "
        "edge. Avoid any design whose call-stack depth grows linearly with the number of "
        "nodes: for example, a `find`/root-lookup implemented as plain recursion (recursing "
        "one Python call per hop while walking an as-yet-uncompressed chain) can exceed "
        "Python's default recursion limit on a long chain, even though the exact same code "
        "works fine on small examples. Prefer an iterative walk, or make sure compression "
        "keeps any single lookup's call depth bounded.\n\n"
        "No test file is visible for this task. Write connected_components.py implementing "
        "connected_components exactly as specified above, then use run_python to "
        "sanity-check it yourself -- in particular, construct a small `nodes` list given in "
        "a DELIBERATELY non-alphabetical order (with no edges, or with edges spanning "
        "several components) and confirm the returned components are sorted internally and "
        "ordered by their smallest element regardless of the input order you used; also try "
        "a `nodes` list with a repeated name and confirm it is not duplicated in the output; "
        "and try a much longer generated chain (at least 1000+ nodes linked edge by edge) to "
        "confirm your implementation does not error out on larger inputs -- before "
        "finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test -- the agent works from the spec alone.
    (Hardened 2026-08-06 after live calibration: with the test file visible, qwen2.5:14b via
    both harnesses scored 1.0/1.0 by simply iterating against run_pytest's literal expected
    output until it passed. Hiding the test forces the model to actually derive and hold onto
    the sort-order contract from prose, rather than converge on it mechanically.)"""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "connected_components", "functions": ["connected_components"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Hand-authored offline plan -- a single leaf is enough for a one-function task."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write connected_components.py implementing "
                    "connected_components(nodes: list, edges: list) -> list: build a "
                    "union-find (disjoint-set) structure over `nodes`, then union() the two "
                    "endpoints of every edge in `edges` (duplicate edges, reversed-endpoint "
                    "duplicates, and self-loop edges must all be handled safely -- unioning a "
                    "node with itself, or unioning the same pair twice, must be a no-op, not "
                    "an error). Then group `nodes` by their final root/representative into "
                    "components. The RETURN VALUE must be fully deterministic regardless of "
                    "input order: sort the members of each component ascending, then sort the "
                    "outer list of components by each component's smallest (first) element -- "
                    "e.g. connected_components(['p', 'k'], []) must return [['k'], ['p']] "
                    "even though 'p' was listed first in the input. `nodes` may also contain "
                    "the same name more than once -- every occurrence refers to the same node, "
                    "and it must appear exactly once in the output. Your implementation must "
                    "also work on much larger inputs (e.g. thousands of nodes chained "
                    "edge-by-edge) without erroring -- avoid a design whose call-stack depth "
                    "grows linearly with the number of nodes (a purely recursive root lookup "
                    "over a long uncompressed chain can exceed Python's default recursion "
                    "limit); prefer an iterative walk. There is no pre-existing test file. Use "
                    "write_file to create connected_components.py, then use run_python to "
                    "sanity-check it yourself: build a `nodes` list in a DELIBERATELY "
                    "non-alphabetical order (with no edges, and separately with edges spanning "
                    "several components) and confirm the result is sorted internally and "
                    "ordered by each component's smallest element regardless of the input "
                    "order; also try a `nodes` list with a repeated name, and a much longer "
                    "generated chain (1000+ nodes), to confirm neither causes wrong output or "
                    "an error. Once you're confident, finish."
                ),
                "expect": "connected_components.py written implementing connected_components "
                          "via union-find, with output sorted deterministically regardless of "
                          "input order, duplicate node names deduplicated, and correctness "
                          "preserved on much larger inputs",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm connected_components.py exists and defines "
                        "connected_components.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["connected_components.py"]},
    }
