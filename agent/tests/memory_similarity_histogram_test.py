"""Unit tests for the E3 similarity calibration (scripts/histogram_memory_similarity.py).

The script picks the value `memory_retrieval_similarity_floor` should take, so its numbers are
only worth anything if it addresses the SAME corpus and asks the SAME question the engine does.
Three pieces of pure logic carry that, and each is pinned here against the engine's own code
rather than against a copy of it:

  * the namespace/collection derivation must equal ``IdeaDagEngine._memo_namespace`` and
    ``MemoryManager``'s naming, or the histogram measures some other run's memory;
  * the query rebuilt from a stored result JSON must match the assembly in
    ``idea_engine._expand_or_execute`` (title, clipped justification, clipped parent goal,
    clipped mandate, then the node-context suffix), because similarity is a function of the
    query text and nothing else;
  * a floor bites in two different ways — rows cut, and retrieval calls emptied outright — and
    the recommendation rests on the second staying small while the first is non-trivial.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from histogram_memory_similarity import (  # noqa: E402
    collection_name,
    floor_effects,
    load_run,
    memo_namespace,
    percentile,
)

from agent.app.idea_engine import IdeaDagEngine  # noqa: E402
from agent.app.idea_memory import MemoryManager  # noqa: E402


def test_namespace_and_collection_match_the_engine():
    mandate = "Find the main span of the Cincinnati suspension bridge."
    namespace = memo_namespace(mandate)
    assert namespace == IdeaDagEngine._memo_namespace(mandate)
    manager = MemoryManager(connector_chroma=None, namespace=namespace)
    assert collection_name(namespace) == manager.collection_name


def _result(nodes, root_id="root", mandate="M" * 300):
    nodes = dict(nodes)
    nodes[root_id] = {"node_id": root_id, "title": mandate, "details": {"mandate": mandate}}
    return {"execution": {"graph": {"root_id": root_id, "nodes": nodes}}}


def test_load_run_rebuilds_the_engine_query_shape(tmp_path):
    import json

    mandate = "M" * 300
    justification = "J" * 300
    parent_goal = "P" * 300
    payload = _result(
        {
            "child": {
                "node_id": "child",
                "title": "Open the bridge page",
                "details": {
                    "justification": justification,
                    "parent_goal": parent_goal,
                    "action": "visit",
                    "action_error": "timeout",
                },
            }
        },
        mandate=mandate,
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))

    loaded = load_run(str(path))
    assert loaded is not None
    got_mandate, queries = loaded
    assert got_mandate == mandate
    assert len(queries) == 1  # the root is not a retrieval site
    query = queries[0]
    # Every part clipped exactly as the engine clips it, in the engine's order.
    assert query.startswith("Open the bridge page " + "J" * 100 + " " + "P" * 100 + " " + "M" * 100)
    assert "J" * 101 not in query and "P" * 101 not in query
    # ...then retrieve_memories_split's node-context suffix.
    assert query.endswith("Open the bridge page action: visit error: timeout")


def test_load_run_skips_a_result_without_a_mandate_or_graph(tmp_path):
    import json

    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"execution": {"graph": {"root_id": "root", "nodes": {}}}}))
    assert load_run(str(path)) is None
    path.write_text("not json")
    assert load_run(str(path)) is None


def test_floor_effects_separates_rows_cut_from_calls_emptied():
    rows = [
        # one call whose rows straddle the floor: trimmed, not emptied
        {"query": "a", "kind": "internal_thoughts", "rank": 0, "similarity": 0.9},
        {"query": "a", "kind": "internal_thoughts", "rank": 1, "similarity": 0.1},
        # one call entirely below it: emptied
        {"query": "b", "kind": "observations", "rank": 0, "similarity": 0.2},
        {"query": "b", "kind": "observations", "rank": 1, "similarity": 0.1},
    ]
    effects = {e["floor"]: e for e in floor_effects(rows, floors=[0.05, 0.3])}
    assert effects[0.05]["row_drop_rate"] == 0.0
    assert effects[0.05]["emptied_call_rate"] == 0.0
    assert effects[0.3]["row_drop_rate"] == 0.75
    assert effects[0.3]["emptied_call_rate"] == 0.5


def test_percentile_is_inclusive_at_the_ends():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert percentile(values, 0.0) == 0.1
    assert percentile(values, 1.0) == 0.5
    assert percentile(values, 0.5) == 0.3
    assert percentile([], 0.5) != percentile([], 0.5)  # nan
