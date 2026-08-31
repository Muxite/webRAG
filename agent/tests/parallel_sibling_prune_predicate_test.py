"""N3c: scoring auto-parallel siblings is what makes prune's ``score is not None`` reachable.

``auto_parallel_siblings`` ships ON and ``evaluate_parallel_siblings`` ships OFF, so the batch
path executes every eligible child and returns with each ``score`` still None. Every consumer
of ``node.score`` then reads nothing: ``identify_prune_candidates`` fired in 14 of 59 runs and
``should_backtrack`` in none, because both are gated on a score that was never computed.

``evaluation_ordering_invariant_test`` pins the flag's own behaviour (off = no batch call, on =
every sibling scored). This module pins the *consequence*: the same graph is prune-inert with
the flag off and prune-eligible with it on. The harness is imported from that module so the two
files cannot drift apart.
"""
from __future__ import annotations

import pytest

from agent.app.got_operations import GoTOperations
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.tests.evaluation_ordering_invariant_test import _graph, _make_engine


async def _run_batch(**overrides):
    """Expand, then run one auto-parallel batch step; returns (graph, root, evaluation)."""
    engine, evaluation, _executed = _make_engine(auto_parallel_siblings=True, **overrides)
    graph = _graph()
    root = graph.root_id()
    await engine._handle_expansion_node(graph, root, 0, None)
    await engine._handle_intermediate_node(graph, root, 1, None)
    return graph, root, evaluation


def _ops():
    return GoTOperations(
        settings={
            "got_prune_enabled": True,
            "got_prune_min_nodes_before_prune": 3,
            "got_adaptive_policies": False,
            "got_prune_score_threshold": 0.15,
        },
        io=None,
        memory_manager=None,
    )


def _add_unexecuted_low_scorer(graph, root, score):
    """A pending sibling the batch never ran -- prune's only legal target."""
    node = graph.add_child(root, "late candidate", details={DetailKey.IS_LEAF.value: True})
    if score is not None:
        graph.evaluate(node.node_id, score)
    return node.node_id


@pytest.mark.asyncio
async def test_flag_off_leaves_the_batch_unscored_and_unevaluated():
    graph, root, evaluation = await _run_batch()

    assert evaluation.batches == []
    for cid in graph.get_node(root).children:
        child = graph.get_node(cid)
        assert child.score is None
        assert DetailKey.EVALUATION.value not in child.details


@pytest.mark.asyncio
async def test_flag_on_scores_every_batch_executed_sibling():
    graph, root, evaluation = await _run_batch(evaluate_parallel_siblings=True)

    assert len(evaluation.batches) == 1
    assert all(graph.get_node(cid).score is not None for cid in graph.get_node(root).children)


@pytest.mark.asyncio
async def test_prune_is_starved_while_the_batch_is_unscored():
    graph, root, _ev = await _run_batch()
    late = _add_unexecuted_low_scorer(graph, root, score=None)

    assert _ops().identify_prune_candidates(graph) == []
    assert graph.get_node(late).score is None


@pytest.mark.asyncio
async def test_prune_predicate_is_satisfiable_once_the_batch_is_scored():
    graph, root, _ev = await _run_batch(evaluate_parallel_siblings=True)
    late = _add_unexecuted_low_scorer(graph, root, score=0.05)

    scored = [n.node_id for n in graph.iter_depth_first() if n.score is not None]
    assert len(scored) > 1, "the batch siblings must be scored, not just the late candidate"
    assert _ops().identify_prune_candidates(graph) == [late]

    ops = _ops()
    assert ops.prune_nodes(graph, [late]) == 1
    assert graph.get_node(late).status == IdeaNodeStatus.SKIPPED
