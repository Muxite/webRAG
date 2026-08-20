"""Re-planning a degenerate fallback parent (`got.reexpand_fallback_nodes_enabled`).

`ExpansionPolicy._create_fallback_candidate` emits ONE guessed candidate when a node's
expansion parsed to nothing, and tags it `DetailKey.FALLBACK_EXPANSION`. The tagged leaf is
already eligible for the ordinary re-expansion triggers; the gap (DAG_FORMATION_REVIEW F6)
is one level up. The PARENT now "has children" (exactly that one leaf) forever, and every
existing gate refuses a node with children, so a collapsed plan could never be supplemented.

Contracts pinned here:
  * default OFF is byte-identical (the parent is never re-targeted);
  * flag ON re-plans the parent, marks the guessed leaf SKIPPED + FALLBACK_SUPERSEDED, and
    threads a corrective hint into the retry's expansion;
  * the carve-out is STRUCTURAL: it fires only when the parent's whole child set is the one
    fallback leaf, so a retry that degenerates again is not retried a third time no matter
    how high `reexpand_max_iterations` is set;
  * the node ceiling and the lineage iteration cap still bound it;
  * the corrective hint is independent of `got_reexpand_corrective_context_enabled`;
  * the documented `max_branching + 1` soft-cap overshoot after a successful retry;
  * the batch/auto-parallel path offers every completed sibling to it (parity wiring).
"""
from __future__ import annotations

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.got_operations import GoTOperations
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies.base import (
    DetailKey,
    ExpansionPolicy,
    EvaluationPolicy,
    DecompositionPolicy,
    IdeaActionType,
    IdeaNodeStatus,
)
from agent.app.idea_policies.actions import LeafAction


class DummyIO:
    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    """Return `n_candidates` real children, or a single fallback-tagged one.

    Records the corrective `REEXPAND_REASON` it saw and consumes it, mirroring the real
    policy's `_build_messages` single-use handling.
    """

    def __init__(self, settings=None, *, n_candidates=3, degenerate=False):
        super().__init__(settings=settings)
        self.calls = 0
        self.n_candidates = n_candidates
        self.degenerate = degenerate
        self.seen_reasons = []

    async def expand(self, graph: IdeaDag, node_id: str, memories=None):
        self.calls += 1
        node = graph.get_node(node_id)
        self.seen_reasons.append(node.details.pop(DetailKey.REEXPAND_REASON.value, None))
        if self.degenerate:
            return [
                {
                    "title": "Analyze and plan next steps",
                    "details": {
                        DetailKey.ACTION.value: IdeaActionType.THINK.value,
                        DetailKey.IS_LEAF.value: True,
                        DetailKey.FALLBACK_EXPANSION.value: True,
                    },
                    "score": None,
                }
            ]
        return [
            {
                "title": f"Real sub-problem {i}",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.THINK.value,
                    DetailKey.IS_LEAF.value: True,
                    DetailKey.JUSTIFICATION.value: "a genuine decomposition step",
                },
                "score": None,
            }
            for i in range(self.n_candidates)
        ]


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        for node_id in candidate_ids:
            graph.evaluate(node_id, 0.6)
        return {node_id: 0.6 for node_id in candidate_ids}


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class AlwaysSuccessAction(LeafAction):
    async def execute(self, graph, node_id, io):
        return {"action": IdeaActionType.THINK.value, "success": True}


class FakeRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type):
        return AlwaysSuccessAction(settings=self.settings)


def _make_engine(*, enabled, max_iters=1, n_candidates=3, degenerate=False,
                 corrective=False, extra=None):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_reexpand_fallback_nodes_enabled": enabled,
        "got_reexpand_corrective_context_enabled": corrective,
        "got_reexpand_enabled": False,
        "got_step_confidence_judge_enabled": False,
        "got_step_confidence_reexpand_enabled": False,
        "got_reexpand_max_iterations": max_iters,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
    }
    if extra:
        settings.update(extra)
    expansion = FakeExpansion(settings, n_candidates=n_candidates, degenerate=degenerate)
    engine = IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=expansion,
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeRegistry(settings),
    )
    engine._got = GoTOperations(settings=settings, io=engine.io, memory_manager=None)
    return engine, expansion


def _fallback_subtree(graph: IdeaDag, *, tagged=True):
    """A parent whose WHOLE expansion collapsed to one (optionally tagged) guessed leaf."""
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    details = {
        DetailKey.ACTION.value: IdeaActionType.THINK.value,
        DetailKey.IS_LEAF.value: True,
    }
    if tagged:
        details[DetailKey.FALLBACK_EXPANSION.value] = True
    leaf = graph.add_child(parent.node_id, "Analyze and plan next steps", details=details)
    return parent, leaf


@pytest.mark.asyncio
async def test_flag_off_degenerate_parent_is_never_retargeted():
    """Regression: flag OFF -> a collapsed parent keeps its single guessed child."""
    engine, expansion = _make_engine(enabled=False)
    graph = IdeaDag(root_title="root")
    parent, leaf = _fallback_subtree(graph)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert result == parent.node_id
    assert parent.children == [leaf.node_id], "flag off -> the parent is never re-planned"
    assert expansion.calls == 0
    assert leaf.status == IdeaNodeStatus.DONE
    assert DetailKey.FALLBACK_SUPERSEDED.value not in leaf.details
    assert "_got_fallback_reexpand_attempted" not in parent.details


@pytest.mark.asyncio
async def test_fallback_parent_is_replanned_and_leaf_superseded():
    """Flag ON -> the parent gains real children, the guessed leaf is superseded."""
    engine, expansion = _make_engine(enabled=True, n_candidates=3)
    graph = IdeaDag(root_title="root")
    parent, leaf = _fallback_subtree(graph)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1, "the parent is re-planned through the real expansion path"
    assert len(parent.children) == 4, "the superseded leaf plus the three real candidates"
    new_children = [graph.get_node(cid) for cid in parent.children[1:]]
    assert all(
        DetailKey.FALLBACK_EXPANSION.value not in c.details for c in new_children
    ), "the retry produced a genuine plan, not another guess"
    assert leaf.status == IdeaNodeStatus.SKIPPED
    assert leaf.details.get(DetailKey.FALLBACK_SUPERSEDED.value) is True
    assert result == parent.node_id, "control returns to the re-planned parent"
    # The corrective hint was threaded into the retry, and consumed by the expansion.
    assert expansion.seen_reasons == [engine._FALLBACK_REEXPAND_REASON]
    assert DetailKey.REEXPAND_REASON.value not in parent.details
    assert parent.details.get("_got_fallback_reexpand_attempted") is True
    assert parent.details.get("_got_fallback_reexpand_recovered") is True


@pytest.mark.asyncio
async def test_untagged_leaf_parent_is_untouched():
    """A parent whose single child is NOT fallback-tagged is not this trigger's business."""
    engine, expansion = _make_engine(enabled=True)
    graph = IdeaDag(root_title="root")
    parent, leaf = _fallback_subtree(graph, tagged=False)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert parent.children == [leaf.node_id]
    assert expansion.calls == 0
    assert leaf.status == IdeaNodeStatus.DONE


@pytest.mark.asyncio
async def test_second_degenerate_retry_is_bounded_by_the_shape_guard():
    """A retry that ALSO degenerates is not retried again, even at a high iteration cap.

    The bound here is structural (the parent's child set is no longer the single fallback
    leaf), not the `reexpand_max_iterations` knob, so pin it with the knob wide open.
    """
    engine, expansion = _make_engine(enabled=True, degenerate=True, max_iters=10)
    graph = IdeaDag(root_title="root")
    parent, leaf = _fallback_subtree(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)
    assert len(parent.children) == 2, "one re-plan, which produced a second guessed leaf"
    second = graph.get_node(parent.children[1])
    assert second.details.get(DetailKey.FALLBACK_EXPANSION.value) is True
    assert parent.details.get("_got_fallback_reexpand_recovered") is None

    # Completing the second fallback leaf must NOT drive a third attempt.
    await engine._handle_leaf_node(graph, second.node_id, 1, None)

    assert len(parent.children) == 2, "the shape guard bounds the repair to a single attempt"
    assert expansion.calls == 1
    assert second.status == IdeaNodeStatus.DONE


@pytest.mark.asyncio
async def test_respects_node_ceiling():
    """A saturated max_total_nodes budget blocks the repair."""
    graph = IdeaDag(root_title="root")
    parent, leaf = _fallback_subtree(graph)
    ceiling = graph.node_count()
    engine, expansion = _make_engine(enabled=True, extra={"max_total_nodes": ceiling})

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert parent.children == [leaf.node_id]
    assert expansion.calls == 0
    assert DetailKey.REEXPAND_REASON.value not in parent.details


@pytest.mark.asyncio
async def test_respects_lineage_iteration_cap():
    """A parent already at its lineage re-expansion budget is not repaired."""
    engine, expansion = _make_engine(enabled=True, max_iters=1)
    graph = IdeaDag(root_title="root")
    parent, leaf = _fallback_subtree(graph)
    parent.details["_got_reexpand_count"] = 1

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert parent.children == [leaf.node_id]
    assert expansion.calls == 0
    assert DetailKey.REEXPAND_REASON.value not in parent.details


@pytest.mark.asyncio
async def test_corrective_hint_is_independent_of_the_generic_flag():
    """The hint is written even with `got_reexpand_corrective_context_enabled` OFF.

    That flag gates only the GENERIC triggers' write site; a known parse failure always
    tells the retry what went wrong.
    """
    engine, expansion = _make_engine(enabled=True, corrective=False)
    assert engine._cfg.got.reexpand_corrective_context_enabled is False
    graph = IdeaDag(root_title="root")
    parent, leaf = _fallback_subtree(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.seen_reasons == [engine._FALLBACK_REEXPAND_REASON]


@pytest.mark.asyncio
async def test_successful_retry_may_overshoot_max_branching_by_one():
    """Documented, accepted soft-cap overshoot: the superseded leaf still occupies a slot.

    `max_branching` caps each EXPANSION, and the repair is a second expansion of the same
    parent, so a repaired parent ends with `max_branching + 1` children. Pinned as
    understood behavior; the extra child is the SKIPPED guess, not live work.
    """
    engine, expansion = _make_engine(
        enabled=True, n_candidates=6, extra={"max_branching": 3},
    )
    graph = IdeaDag(root_title="root")
    parent, leaf = _fallback_subtree(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert len(parent.children) == 4, "max_branching (3) + the superseded fallback leaf"
    live = [
        cid for cid in parent.children
        if graph.get_node(cid).status != IdeaNodeStatus.SKIPPED
    ]
    assert len(live) == 3, "live children still respect max_branching"


@pytest.mark.asyncio
async def test_batch_path_offers_every_completed_sibling_to_the_repair():
    """Parity wiring: the auto-parallel batch path bypasses `_apply_action_result`, so it
    offers each completed sibling to the repair itself.

    The repair necessarily DECLINES here: auto-parallel needs more than one eligible child,
    while the repair's shape guard needs the parent's whole child set to be one fallback
    leaf, so the two cannot both hold. The wiring is defensive parity, kept so the trigger
    cannot silently lose coverage if either condition is ever relaxed. What is pinned is
    that the batch path calls it for every completed sibling and that the shape guard
    correctly refuses a multi-child parent.
    """
    engine, expansion = _make_engine(
        enabled=True, n_candidates=2, extra={"auto_parallel_siblings": True},
    )
    parent_details = {}
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "parent goal", details=parent_details)
    leaves = [
        graph.add_child(
            parent.node_id, f"guessed step {i}",
            details={
                DetailKey.ACTION.value: IdeaActionType.THINK.value,
                DetailKey.IS_LEAF.value: True,
                DetailKey.FALLBACK_EXPANSION.value: True,
            },
        )
        for i in range(2)
    ]
    for leaf in leaves:
        graph.evaluate(leaf.node_id, 0.6)

    offered = []
    original = engine._maybe_reexpand_fallback_parent

    async def _spy(g, node_id, step_index):
        offered.append(node_id)
        return await original(g, node_id, step_index)

    engine._maybe_reexpand_fallback_parent = _spy  # type: ignore[assignment]

    await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    assert offered == [leaf.node_id for leaf in leaves]
    assert expansion.calls == 0, "a multi-child parent fails the shape guard"
    assert parent.children == [leaf.node_id for leaf in leaves]
    assert all(leaf.status == IdeaNodeStatus.DONE for leaf in leaves)
