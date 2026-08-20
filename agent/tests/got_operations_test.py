"""
Unit tests for GoTOperations: dynamic beam width, dedup threshold, pruning.
No external services required.
"""
from __future__ import annotations

import pytest

from agent.app.got_operations import GoTOperations
from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import IdeaNodeStatus


def _make_ops(**overrides):
    settings = {
        "got_dynamic_beam_enabled": True,
        "got_beam_min": 2,
        "got_beam_max": 5,
        "got_adaptive_policies": True,
        "got_prune_enabled": True,
        "got_prune_min_nodes_before_prune": 6,
        "got_dedup_enabled": True,
    }
    settings.update(overrides)
    return GoTOperations(settings=settings, io=None, memory_manager=None)


def _graph_with_scored_children(parent_id_holder, scores):
    g = IdeaDag(root_title="root")
    for score in scores:
        g.add_child(parent_id=g.root_id(), title=f"n-{score}", score=score)
    parent_id_holder.append(g.root_id())
    return g


def test_beam_width_returns_max_when_no_scores():
    ops = _make_ops()
    g = IdeaDag(root_title="root")
    assert ops.compute_dynamic_beam_width(g) == 5


def test_beam_width_adaptive_widens_with_spread():
    ops = _make_ops()
    holder = []
    g_wide = _graph_with_scored_children(holder, [0.1, 0.2, 0.8, 0.9, 0.1, 0.9])
    g_tight = _graph_with_scored_children(holder, [0.7, 0.71, 0.72, 0.73, 0.74, 0.75])
    wide = ops.compute_dynamic_beam_width(g_wide)
    tight = ops.compute_dynamic_beam_width(g_tight)
    assert 2 <= tight <= wide <= 5
    assert wide > tight


def test_beam_width_legacy_path_when_adaptive_off():
    ops = _make_ops(got_adaptive_policies=False, got_beam_score_high=0.7, got_beam_score_low=0.3)
    holder = []
    g_high = _graph_with_scored_children(holder, [0.9, 0.91, 0.92])
    g_low = _graph_with_scored_children(holder, [0.1, 0.1, 0.1])
    assert ops.compute_dynamic_beam_width(g_high) == 2
    assert ops.compute_dynamic_beam_width(g_low) == 5


def test_adaptive_dedup_threshold_within_bounds():
    ops = _make_ops()
    g = IdeaDag(root_title="root")
    threshold_empty = ops._adaptive_dedup_threshold(g)
    assert 0.75 <= threshold_empty <= 0.92
    # Add a parent with many children to push toward upper bound.
    for i in range(10):
        g.add_child(parent_id=g.root_id(), title=f"n-{i}")
    threshold_dense = ops._adaptive_dedup_threshold(g)
    assert threshold_dense >= threshold_empty
    assert threshold_dense <= 0.92


def test_adaptive_dedup_disabled_returns_fixed():
    ops = _make_ops(got_adaptive_policies=False, got_dedup_similarity_threshold=0.85)
    g = IdeaDag(root_title="root")
    assert ops._adaptive_dedup_threshold(g) == 0.85


# --- dedup kill-switch and all-filtered fallback ------------------------------------------
# Groundwork for ASSUMPTION_AUDIT.md's E1 (ablate dedup entirely). The experiment's OFF arm is
# ``got_dedup_enabled: false`` and nothing else, so what it actually measures depends on two
# behaviours that nothing pinned before: that the flag short-circuits BOTH entry points before
# any memory query (otherwise the arms differ in retrieval cost as well as in dedup), and that
# the all-filtered fallback hands back ``candidates[:1]``. The second is not a corner case: in
# the recorded log corpus 70 of 154 firing batches (45.5%) flagged every candidate, and 50 of
# those were single-candidate batches where the fallback restores the only candidate and the
# arms are byte-identical despite the log announcing a filter.


class _RecordingMemory:
    """Memory manager that fails the test if dedup queries it."""

    distance_space = "cosine"

    def __init__(self, memories=None):
        self.queries = []
        self._memories = memories or []

    async def retrieve_relevant_memories(self, query, n_results, memory_type):
        self.queries.append(query)
        return self._memories


def _dup_memory(node_id="existing", distance=0.0):
    return _RecordingMemory([{"distance": distance, "metadata": {"node_id": node_id}}])


@pytest.mark.asyncio
async def test_dedup_disabled_short_circuits_before_touching_memory():
    memory = _dup_memory()
    settings = {"got_dedup_enabled": False}
    ops = GoTOperations(settings=settings, io=None, memory_manager=memory)
    g = IdeaDag(root_title="root")
    assert await ops.is_duplicate_thought("t", "goal", g) == (False, None)
    candidates = [{"title": "a"}, {"title": "b"}]
    assert await ops.filter_duplicate_candidates(candidates, g) == candidates
    assert memory.queries == []   # the OFF arm spends no retrieval either


@pytest.mark.asyncio
async def test_dedup_enabled_drops_the_duplicate_candidate():
    ops = GoTOperations(
        settings={"got_dedup_enabled": True, "got_adaptive_policies": False,
                  "got_dedup_similarity_threshold": 0.85},
        io=None,
        memory_manager=_dup_memory(),
    )
    g = IdeaDag(root_title="root")
    is_dup, node_id = await ops.is_duplicate_thought("t", "goal", g)
    assert (is_dup, node_id) == (True, "existing")


@pytest.mark.asyncio
async def test_all_filtered_fallback_restores_the_first_candidate():
    # Every candidate matches the same stored memory, so `filtered` ends up empty. Returning
    # nothing would leave the parent childless, so the fallback keeps the head of the list --
    # which means a "filtered 1 out of 1" batch is a no-op, not a removal.
    ops = GoTOperations(
        settings={"got_dedup_enabled": True, "got_adaptive_policies": False,
                  "got_dedup_similarity_threshold": 0.85},
        io=None,
        memory_manager=_dup_memory(),
    )
    g = IdeaDag(root_title="root")
    assert await ops.filter_duplicate_candidates([{"title": "only"}], g) == [{"title": "only"}]
    assert await ops.filter_duplicate_candidates(
        [{"title": "a"}, {"title": "b"}], g
    ) == [{"title": "a"}]


@pytest.mark.asyncio
async def test_dedup_without_a_memory_manager_is_inert():
    ops = GoTOperations(settings={"got_dedup_enabled": True}, io=None, memory_manager=None)
    g = IdeaDag(root_title="root")
    assert await ops.is_duplicate_thought("t", "goal", g) == (False, None)


def test_prune_below_minimum_node_count_returns_empty():
    ops = _make_ops(got_prune_min_nodes_before_prune=6)
    g = IdeaDag(root_title="root")
    g.add_child(parent_id=g.root_id(), title="a", score=0.0)
    assert ops.identify_prune_candidates(g) == []


def test_prune_adaptive_uses_stddev_threshold():
    ops = _make_ops(got_prune_min_nodes_before_prune=6, got_prune_stddev_factor=1.0)
    g = IdeaDag(root_title="root")
    # 6 scored children, one much lower than the rest -> should be pruned.
    scores = [0.6, 0.65, 0.7, 0.75, 0.68, 0.05]
    nodes = [g.add_child(parent_id=g.root_id(), title=f"n-{s}", score=s) for s in scores]
    prune_ids = ops.identify_prune_candidates(g)
    assert nodes[-1].node_id in prune_ids
    for n in nodes[:-1]:
        assert n.node_id not in prune_ids


def test_prune_nodes_sets_status_skipped_and_marks_detail():
    ops = _make_ops()
    g = IdeaDag(root_title="root")
    n = g.add_child(parent_id=g.root_id(), title="x", score=0.0)
    pruned = ops.prune_nodes(g, [n.node_id])
    assert pruned == 1
    assert n.status == IdeaNodeStatus.SKIPPED
    assert n.details.get("_got_pruned") is True


def test_prune_respects_terminal_status():
    ops = _make_ops()
    g = IdeaDag(root_title="root")
    n = g.add_child(parent_id=g.root_id(), title="x", score=0.0, status=IdeaNodeStatus.DONE)
    pruned = ops.prune_nodes(g, [n.node_id])
    assert pruned == 0
    assert n.status == IdeaNodeStatus.DONE
