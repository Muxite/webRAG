"""Branch-scoped memory retrieval: the lineage filter that makes a race actually blind.

``write_memory`` has always stamped ``node_id`` on every chunk, but
``retrieve_relevant_memories`` only ever filtered on ``memory_type`` — so any node could
read any other node's memories, including a sibling branch's. Race-and-merge
(``idea_policies/alternative_branch.py``) is premised on racing branches being independent
routes to the same fact, which that leak contradicts by construction.

The load-bearing assertions are the OFF ones: the flag ships off, and with it off the
``where`` dict handed to chroma must carry no ``node_id`` key at all — not an empty one —
so every measurement taken to date stays comparable.
"""
from __future__ import annotations

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_memory import MemoryManager
from agent.app.idea_policies.config import MemoryConfig


class _FakeCollection:
    def __init__(self):
        self.configuration_json = {"hnsw": {"space": "cosine"}}
        self.metadata = {"hnsw:space": "cosine"}


class _FakeChroma:
    """Holds written chunks and applies the ``where`` clause the way chroma would."""

    def __init__(self):
        self.rows: list[dict] = []
        self.wheres: list[dict | None] = []

    async def get_or_create_collection(self, collection, metadata=None):
        return _FakeCollection()

    async def add_to_chroma(self, collection, ids, metadatas, documents):
        for i, doc in enumerate(documents):
            self.rows.append({"id": ids[i], "metadata": metadatas[i], "document": doc})
        return True

    async def add_to_chroma_parallel(self, collection, ids, metadatas, documents):
        return await self.add_to_chroma(collection, ids, metadatas, documents)

    @staticmethod
    def _matches(metadata, where):
        if not where:
            return True
        if "$and" in where:
            return all(_FakeChroma._matches(metadata, w) for w in where["$and"])
        for key, cond in where.items():
            if isinstance(cond, dict):
                if "$in" in cond and metadata.get(key) not in cond["$in"]:
                    return False
            elif metadata.get(key) != cond:
                return False
        return True

    async def query_chroma(self, collection, query_texts, n_results=5, where=None):
        self.wheres.append(where)
        hits = [r for r in self.rows if self._matches(r["metadata"], where)][:n_results]
        return {
            "documents": [[r["document"] for r in hits]],
            "metadatas": [[r["metadata"] for r in hits]],
            "distances": [[0.1 for _ in hits]],
            "ids": [[r["id"] for r in hits]],
        }


def _two_branch_graph():
    """root -> (branch A, branch B), each with one child. Two independent lineages."""
    graph = IdeaDag(root_title="mandate")
    branch_a, branch_b = graph.expand(
        graph.root_id(), [{"title": "branch A"}, {"title": "branch B"}]
    )
    a_leaf = graph.expand(branch_a.node_id, [{"title": "A leaf"}])[0]
    b_leaf = graph.expand(branch_b.node_id, [{"title": "B leaf"}])[0]
    return graph, a_leaf.node_id, b_leaf.node_id


async def _manager_with_branch_a_memory():
    chroma = _FakeChroma()
    manager = MemoryManager(connector_chroma=chroma, namespace="idea_dag:test")
    graph, a_leaf, b_leaf = _two_branch_graph()
    await manager.write_memory(
        content="the answer branch A found",
        node_id=a_leaf,
        node_title="A leaf",
        memory_type="observation",
    )
    return chroma, manager, graph, a_leaf, b_leaf


# --- the flag ships off ---------------------------------------------------------------


def test_branch_scoping_ships_off():
    assert MemoryConfig().branch_scoped_retrieval_enabled is False


def test_where_clause_is_unchanged_when_unscoped():
    build = MemoryManager._build_where
    assert build("observation", None) == {"memory_type": "observation"}
    assert build(None, None) is None


@pytest.mark.asyncio
async def test_off_adds_no_node_id_key_to_the_where_clause():
    chroma, manager, _graph, _a_leaf, _b_leaf = await _manager_with_branch_a_memory()
    await manager.retrieve_relevant_memories("answer", memory_type="observation")
    assert chroma.wheres == [{"memory_type": "observation"}]


# --- the scoped clause ----------------------------------------------------------------


def test_scoped_clause_combines_both_filters_with_and():
    assert MemoryManager._build_where("observation", ["n1", "n2"]) == {
        "$and": [{"memory_type": "observation"}, {"node_id": {"$in": ["n1", "n2"]}}]
    }


def test_scoped_clause_stays_bare_when_type_is_absent():
    """chroma rejects an ``$and`` with fewer than two operands."""
    assert MemoryManager._build_where(None, ["n1"]) == {"node_id": {"$in": ["n1"]}}


def test_empty_scope_is_honoured_rather_than_dropped():
    assert MemoryManager._build_where(None, []) == {"node_id": {"$in": []}}


# --- the leak, and its fix ------------------------------------------------------------


@pytest.mark.asyncio
async def test_sibling_branch_memory_leaks_when_scoping_is_off():
    _chroma, manager, _graph, _a_leaf, _b_leaf = await _manager_with_branch_a_memory()
    got = await manager.retrieve_relevant_memories("answer", memory_type="observation")
    assert [m["content"] for m in got] == ["the answer branch A found"]


@pytest.mark.asyncio
async def test_sibling_branch_memory_is_excluded_when_scoping_is_on():
    _chroma, manager, graph, _a_leaf, b_leaf = await _manager_with_branch_a_memory()
    lineage = [n.node_id for n in graph.path_to_root(b_leaf)]
    got = await manager.retrieve_relevant_memories(
        "answer", memory_type="observation", scope_node_ids=lineage
    )
    assert got == []


@pytest.mark.asyncio
async def test_own_lineage_memory_is_still_visible_when_scoping_is_on():
    _chroma, manager, graph, a_leaf, _b_leaf = await _manager_with_branch_a_memory()
    lineage = [n.node_id for n in graph.path_to_root(a_leaf)]
    got = await manager.retrieve_relevant_memories(
        "answer", memory_type="observation", scope_node_ids=lineage
    )
    assert [m["content"] for m in got] == ["the answer branch A found"]


@pytest.mark.asyncio
async def test_split_retrieval_forwards_the_scope_to_both_halves():
    chroma, manager, graph, _a_leaf, b_leaf = await _manager_with_branch_a_memory()
    lineage = [n.node_id for n in graph.path_to_root(b_leaf)]
    got = await manager.retrieve_memories_split("answer", scope_node_ids=lineage)
    assert got == {"internal_thoughts": [], "observations": []}
    assert all("$and" in w for w in chroma.wheres)
    assert all(w["$and"][1] == {"node_id": {"$in": lineage}} for w in chroma.wheres)
