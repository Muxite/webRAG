"""
codebench task c06 — hard/hidden, requirement-to-spec translation, graph traversal.

First genuinely MULTI-LEAF task in this benchmark (c01-c05 are single-leaf). Two leaves with a
real, structural dependency:
  * leaf_a builds ``graph_types.py`` — a small, independently-useful directed-graph module
    (``Graph`` class: add_node/add_edge/nodes/successors/in_degree).
  * leaf_b (``depends_on: ["leaf_a"]``) builds ``topo_sort.py``'s ``topo_sort(deps) -> list``,
    which IMPORTS AND USES ``graph_types.Graph`` to run Kahn's algorithm.

See ``services/agent/app/testing/execution_compiled_code.py``'s ``_dep_text``/
``UPSTREAM_INCOMPLETE_MARKER`` for how leaf_b's ``{leaf_a}`` placeholder actually gets filled at
runtime: the model's own ``finish(summary)`` text, PLUS a deterministic
``[Files written by this step: ...]`` note from the sandbox's own bookkeeping (true regardless of
what the model said), and marked with ``UPSTREAM_INCOMPLETE_MARKER`` if leaf_a never finished.
Because that summary text is not fully reliable even in the happy path, leaf_b's instruction below
independently RESTATES the graph_types API leaf_a was told to build, as a fallback a weak model can
lean on instead of trusting leaf_a's self-report alone.

Ground truth for ``topo_sort`` independently verified with a throwaway Kahn's-algorithm reference
script (NOT hand-computed) before being embedded here — see the offline test file's own
reimplementation for the second, independent check. Verified cases (deps -> one accepted order):
    {"solo": []} -> ["solo"]
    {"a": [], "b": ["a"], "c": ["b"]} -> ["a", "b", "c"]  (only one valid order for a strict chain)
    {"app": ["lib_a","lib_b"], "lib_a": ["core"], "lib_b": ["core"], "core": []}
        -> e.g. ["core", "lib_a", "lib_b", "app"] (core first, app last; lib_a/lib_b either order)
    {"a": [], "b": ["a"], "x": [], "y": ["x"]} -> e.g. ["a", "x", "b", "y"] (many valid interleavings)
    {"app": ["utillib"]} (utillib never appears as its own key) -> e.g. ["utillib", "app"]
    {"root": [], "a": ["root"], "b": ["root"], "c": ["root"]} -> root first, a/b/c any order after
    {"a": ["b"], "b": ["c"], "c": ["a"]} -> raises ValueError (3-cycle)
    {"a": ["b"], "b": ["a"]} -> raises ValueError (2-cycle)
Multiple valid topological orders exist for most of these graphs, so grading uses a structural
``is_valid_topo_order(deps, order)`` checker (embedded in the canonical test file itself) rather
than comparing against one hardcoded expected list — except the two cases above that mathematically
have exactly one valid order (a strict linear chain, and a single node), which are still asserted
exactly as a sharper signal.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_topo_sort.py"

_TEST_FILE_CONTENT = '''\
import pytest
from topo_sort import topo_sort


def is_valid_topo_order(deps: dict, order: list) -> bool:
    """True iff `order` is a permutation of every node in `deps` (keys and dependency values)
    such that every dependency appears strictly before everything that depends on it. Written
    here rather than comparing to one hardcoded list, because most graphs below admit more than
    one valid topological order."""
    all_nodes = set(deps.keys())
    for dep_list in deps.values():
        all_nodes.update(dep_list)
    if set(order) != all_nodes or len(order) != len(all_nodes):
        return False
    position = {node: i for i, node in enumerate(order)}
    for target, dep_list in deps.items():
        for dep in dep_list:
            if position[dep] >= position[target]:
                return False
    return True


def test_single_node_no_deps():
    assert topo_sort({"solo": []}) == ["solo"]


def test_linear_chain():
    # a strict chain has exactly one valid order
    assert topo_sort({"a": [], "b": ["a"], "c": ["b"]}) == ["a", "b", "c"]


def test_diamond_dependency():
    deps = {"app": ["lib_a", "lib_b"], "lib_a": ["core"], "lib_b": ["core"], "core": []}
    order = topo_sort(deps)
    assert is_valid_topo_order(deps, order)


def test_disconnected_components():
    deps = {"a": [], "b": ["a"], "x": [], "y": ["x"]}
    order = topo_sort(deps)
    assert is_valid_topo_order(deps, order)


def test_dependency_referenced_only_as_value():
    # "utillib" never appears as its own key, only inside another target's dependency list
    deps = {"app": ["utillib"]}
    order = topo_sort(deps)
    assert is_valid_topo_order(deps, order)
    assert set(order) == {"app", "utillib"}


def test_wide_fanin():
    deps = {"root": [], "a": ["root"], "b": ["root"], "c": ["root"]}
    order = topo_sort(deps)
    assert is_valid_topo_order(deps, order)


def test_cycle_raises_value_error():
    with pytest.raises(ValueError):
        topo_sort({"a": ["b"], "b": ["c"], "c": ["a"]})


def test_two_node_cycle_raises_value_error():
    with pytest.raises(ValueError):
        topo_sort({"a": ["b"], "b": ["a"]})
'''

# Single-node and the "referenced only as a value" edge case are supporting/bonus credit, not
# keystone: neither actually exercises a real multi-step traversal decision. The five that gate
# the score are the ones a broken (or non-Kahn's, or cycle-blind) implementation would plausibly
# fail: a real chain, a real diamond, disconnected components, wide fan-in, and both cycle shapes.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_linear_chain",
    f"{_TEST_FILE_PATH}::test_diamond_dependency",
    f"{_TEST_FILE_PATH}::test_disconnected_components",
    f"{_TEST_FILE_PATH}::test_wide_fanin",
    f"{_TEST_FILE_PATH}::test_cycle_raises_value_error",
    f"{_TEST_FILE_PATH}::test_two_node_cycle_raises_value_error",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c06",
        "title": "topo-sort-build-deps",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "You are building a small build-order tool, split across two files.\n\n"
        "## Part 1 — graph_types.py\n\n"
        "Write `graph_types.py` defining a class `Graph` for directed-graph bookkeeping, with "
        "exactly this API:\n"
        "- `Graph()` — construct an empty graph.\n"
        "- `add_node(self, node)` — register `node` if it isn't already present (idempotent).\n"
        "- `add_edge(self, src, dst)` — register a directed edge `src -> dst` (also registers "
        "both endpoints as nodes if new); increments `dst`'s in-degree.\n"
        "- `nodes(self)` — return a list of every registered node, in first-added order.\n"
        "- `successors(self, node)` — return a list of nodes `node` has an edge TO, in the order "
        "the edges were added.\n"
        "- `in_degree(self, node)` — return the current in-degree (int) of `node` (0 if it has "
        "never been an edge's destination).\n\n"
        "## Part 2 — topo_sort.py\n\n"
        "Write `topo_sort.py` that does `from graph_types import Graph` and defines "
        "`topo_sort(deps: dict) -> list`.\n\n"
        "`deps` maps each build target (str) to a list of the names of its DIRECT dependencies "
        "(str) — e.g. `{\"app\": [\"lib_a\", \"lib_b\"], \"lib_a\": [\"core\"], "
        "\"lib_b\": [\"core\"], \"core\": []}`. A dependency name may or may not also appear as "
        "its own key in `deps` (if it doesn't, treat it as having no further dependencies of its "
        "own).\n\n"
        "`topo_sort(deps)` must return a list containing every target and every dependency name "
        "exactly once, ordered so that every dependency appears BEFORE everything that depends on "
        "it (a valid topological order — when a graph has more than one valid order, any of them "
        "is acceptable). If the dependency graph contains a cycle, raise `ValueError` instead of "
        "returning.\n\n"
        "Build `topo_sort` on top of `Graph` (add an edge from each dependency to the target that "
        "needs it, then run a standard Kahn's-algorithm traversal using the graph's `nodes`/"
        "`successors`/`in_degree`) rather than re-deriving your own separate graph representation "
        "from scratch.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check both modules "
        "yourself (construct a small `Graph`, add a few edges, print `nodes()`/`successors()`/"
        "`in_degree()`; call `topo_sort()` on a couple of hand-built `deps` dicts and confirm every "
        "dependency precedes its dependent, and that a cyclic `deps` dict raises `ValueError`) "
        "before finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test — the agent works from the spec alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "modules": [
                {"module": "graph_types", "class": "Graph",
                 "methods": ["add_node", "add_edge", "nodes", "successors", "in_degree"]},
                {"module": "topo_sort", "functions": ["topo_sort"]},
            ]
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Two leaves with a REAL structural dependency (not an artificial split): leaf_b imports and
    uses what leaf_a built. leaf_b's instruction both templates leaf_a's actual outcome via
    ``{leaf_a}`` (see execution_compiled_code.py's ``_dep_text`` for exactly what lands there —
    the model's summary plus a deterministic files-written note, or an UPSTREAM-DID-NOT-COMPLETE
    marker) AND independently restates the expected API, since a weak model's own summary of what
    it built is not fully reliable even when it did finish honestly.
    """
    return {
        "leaves": [
            {
                "id": "leaf_a",
                "instruction": (
                    "Write graph_types.py implementing a directed-graph helper class named "
                    "`Graph`, for later use by a topological-sort module. Expose EXACTLY these "
                    "names: `Graph()` (empty constructor), `add_node(self, node)` (idempotent), "
                    "`add_edge(self, src, dst)` (registers a directed edge src -> dst, adds both "
                    "endpoints as nodes if new, increments dst's in-degree), `nodes(self)` "
                    "(list of every node, first-added order), `successors(self, node)` (list of "
                    "nodes `node` has an edge to, added order), `in_degree(self, node)` (int, 0 "
                    "if node was never an edge's destination). This class must be genuinely "
                    "correct and usable on its own — a later step imports it with "
                    "`from graph_types import Graph` and calls every one of these methods, so the "
                    "class/method names must match EXACTLY. Use write_file to create it, then use "
                    "run_python to construct a small Graph, add a couple of edges, and print "
                    "nodes()/successors()/in_degree() to confirm they behave as described. Then "
                    "finish, and in your summary explicitly name the class and its methods."
                ),
                "expect": "graph_types.py written, defining Graph with add_node/add_edge/nodes/"
                          "successors/in_degree, sanity-checked with run_python",
                "depends_on": [],
            },
            {
                "id": "leaf_b",
                "instruction": (
                    "Leaf A already wrote graph_types.py, a small directed-graph module. Leaf A "
                    "reported: {leaf_a}\n\n"
                    "Regardless of exactly what that summary said, graph_types.py is expected to "
                    "define a `Graph` class with this API: `Graph()`, `add_node(self, node)`, "
                    "`add_edge(self, src, dst)` (registers a directed edge src -> dst and "
                    "increments dst's in-degree), `nodes(self)` (list of all registered nodes), "
                    "`successors(self, node)` (list of nodes reachable via one edge from node), "
                    "and `in_degree(self, node)` (int). If graph_types.py does not actually "
                    "expose that API, use read_file to inspect it first and adapt your usage "
                    "accordingly — do not guess blindly.\n\n"
                    "Now write topo_sort.py that does `from graph_types import Graph` and "
                    "implements `topo_sort(deps: dict) -> list`. `deps` maps each build target "
                    "(str) to a list of the names of its DIRECT dependencies (str), e.g. "
                    "`{\"app\": [\"lib_a\", \"lib_b\"], \"lib_a\": [\"core\"], "
                    "\"lib_b\": [\"core\"], \"core\": []}`. Build a Graph where, for every target "
                    "T and every dependency D in deps[T], you add_edge(D, T) (D must come before "
                    "T) — remember to register every dependency name as a node even if it never "
                    "appears as its own key in deps. Then run Kahn's algorithm using the Graph's "
                    "nodes()/successors()/in_degree(): repeatedly take a not-yet-output node "
                    "whose remaining in-degree is 0, append it to the result, and 'remove' it by "
                    "decrementing the remaining in-degree of its successors (track the "
                    "decrementing yourself, e.g. in a plain dict — do not mutate the Graph's own "
                    "counts); continue until every node has been output. If you run out of "
                    "zero-in-degree nodes before every node is placed, the graph has a cycle — "
                    "raise ValueError instead of returning. Any order where every dependency "
                    "precedes everything that depends on it is acceptable; there is no single "
                    "'correct' order when a graph has more than one. Use write_file to create "
                    "topo_sort.py, then use run_python to sanity-check it against a couple of "
                    "hand-built deps dicts (including one with a cycle, to confirm ValueError is "
                    "raised). Fix issues with patch_file/run_python until you're confident, then "
                    "finish."
                ),
                "expect": "topo_sort.py written, importing graph_types.Graph and implementing "
                          "topo_sort(deps) via Kahn's algorithm with cycle detection",
                "depends_on": ["leaf_a"],
            },
        ],
        "aggregation": "Confirm graph_types.py and topo_sort.py both exist and report the "
                        "sanity-check results from each leaf.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["graph_types.py", "topo_sort.py"]},
    }
