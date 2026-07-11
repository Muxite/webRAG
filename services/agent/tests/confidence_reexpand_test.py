"""Tests for the confidence->action loop (`got.step_confidence_reexpand_enabled`).

The decorrelated per-step confidence judge (`got.step_confidence_judge_enabled`)
previously only *logged* a score. This closes the loop: a completed leaf the judge
distrusts (confidence below `got.step_confidence_reexpand_threshold`) DRIVES a
bounded re-expansion — a genuine "observe the step's trustworthiness, then decide
to take another step". It reuses the same `_apply_reexpand` machinery and the same
`reexpand_max_iterations` / `max_total_nodes` bounds as the follow-up path, so
termination is guaranteed.

Key contracts pinned here:
  * default OFF is byte-identical (no re-expansion off a low score);
  * a low score re-expands, a high score does not;
  * the trigger is independent of `got_reexpand_enabled` (the follow-up detector);
  * the same iteration cap / node ceiling that bound the follow-up path bound this
    path too, so an enabled >=2 iteration loop terminates and never spins.
"""
from __future__ import annotations

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.got_operations import GoTOperations
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies.action_constants import ActionResultKey
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
    """Return a single deterministic follow-up child on every expand call."""

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.calls = 0

    async def expand(self, graph: IdeaDag, node_id: str, memories=None):
        self.calls += 1
        return [
            {
                "title": "Follow-up: re-investigate the distrusted leaf",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.THINK.value,
                    DetailKey.JUSTIFICATION.value: "low step confidence",
                },
                "score": None,
            }
        ]


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        return {nid: 0.6 for nid in candidate_ids}


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


class _JudgeStub:
    """Returns a fixed confidence verdict and records call count."""

    def __init__(self, confidence):
        self.confidence = confidence
        self.calls = 0

    async def __call__(self, graph, node_id, model_name=None):
        self.calls += 1
        if self.confidence is None:
            return None
        return {"confidence": self.confidence, "reason": "stub"}


def _make_engine(*, conf_reexpand, threshold=0.5, max_iters=1,
                 judge_enabled=True, followup_enabled=False, extra=None):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_step_confidence_judge_enabled": judge_enabled,
        "got_step_confidence_reexpand_enabled": conf_reexpand,
        "got_step_confidence_reexpand_threshold": threshold,
        "got_reexpand_enabled": followup_enabled,
        "got_reexpand_max_iterations": max_iters,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
    }
    if extra:
        settings.update(extra)
    expansion = FakeExpansion(settings)
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
    return engine, expansion


def _attach_judge(engine, confidence):
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    stub = _JudgeStub(confidence)
    ops.judge_step_confidence = stub  # type: ignore[assignment]
    engine._got = ops
    return stub


def _completed_leaf(graph: IdeaDag):
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    leaf = graph.add_child(
        parent.node_id,
        "Resolve a sub-fact",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.IS_LEAF.value: True},
    )
    return parent, leaf


@pytest.mark.asyncio
async def test_flag_off_low_confidence_does_not_reexpand():
    """Regression: confidence-reexpand OFF -> a low score logs but never re-expands."""
    engine, expansion = _make_engine(conf_reexpand=False)
    _attach_judge(engine, 0.1)  # very low, would trigger if the flag were on
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert result == parent.node_id
    assert leaf.children == [], "flag off -> no re-expansion off a low score"
    assert expansion.calls == 0
    assert "_got_reexpanded" not in leaf.details
    # The score is still logged (the judge instrumentation is unchanged).
    assert len(engine._step_confidences) == 1
    assert leaf.status == IdeaNodeStatus.DONE


@pytest.mark.asyncio
async def test_low_confidence_drives_reexpansion():
    """Flag ON + confidence below threshold -> the distrusted leaf re-expands."""
    engine, expansion = _make_engine(conf_reexpand=True, threshold=0.5)
    stub = _attach_judge(engine, 0.2)
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert stub.calls == 1
    assert expansion.calls == 1, "a low-confidence leaf re-expands via the real policy"
    assert len(leaf.children) == 1
    assert result == leaf.node_id, "engine stays on the re-expanded node"
    assert leaf.details.get("_got_reexpand_count") == 1
    assert leaf.details.get("_got_reexpanded") is True
    assert leaf.status == IdeaNodeStatus.ACTIVE


@pytest.mark.asyncio
async def test_high_confidence_does_not_reexpand():
    """Flag ON but confidence at/above threshold -> a trusted leaf stays terminal."""
    engine, expansion = _make_engine(conf_reexpand=True, threshold=0.5)
    _attach_judge(engine, 0.9)
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert result == parent.node_id
    assert leaf.children == [], "a trusted (high-confidence) leaf does not re-expand"
    assert expansion.calls == 0
    assert leaf.status == IdeaNodeStatus.DONE


@pytest.mark.asyncio
async def test_confidence_reexpand_independent_of_followup_flag():
    """The confidence trigger drives re-expansion even with got_reexpand_enabled OFF
    (the follow-up detector is never consulted)."""
    engine, expansion = _make_engine(
        conf_reexpand=True, threshold=0.5, followup_enabled=False,
    )
    _attach_judge(engine, 0.15)
    # Wire a follow-up checker that would raise if consulted, proving independence.
    async def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("follow-up detector must not be consulted")
    engine._got.check_needs_followup = _boom  # type: ignore[assignment]
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1
    assert len(leaf.children) == 1


@pytest.mark.asyncio
async def test_confidence_reexpand_respects_iteration_cap():
    """Cap already reached -> a low score cannot re-expand again (per-node bound)."""
    engine, expansion = _make_engine(conf_reexpand=True, threshold=0.5, max_iters=1)
    _attach_judge(engine, 0.1)
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)
    leaf.details["_got_reexpand_count"] = 1  # already re-expanded once

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert result == parent.node_id
    assert leaf.children == [], "no re-expansion past the iteration cap"
    assert expansion.calls == 0


@pytest.mark.asyncio
async def test_confidence_reexpand_respects_node_ceiling():
    """A saturated max_total_nodes budget blocks the confidence-driven re-expansion."""
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)
    ceiling = graph.node_count()  # already at the ceiling: no room to grow
    engine, expansion = _make_engine(
        conf_reexpand=True, threshold=0.5, extra={"max_total_nodes": ceiling},
    )
    _attach_judge(engine, 0.1)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert leaf.children == [], "no re-expansion when the node budget is exhausted"
    assert expansion.calls == 0


@pytest.mark.asyncio
async def test_gate_helper_thresholds():
    """`_confidence_triggers_reexpand` fires strictly below the threshold only."""
    engine, _ = _make_engine(conf_reexpand=True, threshold=0.5)
    _attach_judge(engine, 0.5)
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)
    # Give the leaf a successful result so the success guard passes.
    leaf.details[DetailKey.ACTION_RESULT.value] = {ActionResultKey.SUCCESS.value: True}

    assert engine._confidence_triggers_reexpand(graph, leaf.node_id, 0.49) is True
    assert engine._confidence_triggers_reexpand(graph, leaf.node_id, 0.50) is False
    assert engine._confidence_triggers_reexpand(graph, leaf.node_id, 0.99) is False
    assert engine._confidence_triggers_reexpand(graph, leaf.node_id, None) is False


class _StepJudge:
    """Judge that reports a fixed confidence, incrementing calls each time."""

    def __init__(self, confidence):
        self.confidence = confidence
        self.calls = 0

    async def __call__(self, graph, node_id, model_name=None):
        self.calls += 1
        return {"confidence": self.confidence, "reason": "low"}


async def _drive(engine, graph, start_id, max_steps=60):
    current = start_id
    for step in range(max_steps):
        current = await engine.step(graph, current, step)
        if current is None:
            return step + 1
    return max_steps


@pytest.mark.asyncio
async def test_two_iteration_confidence_reexpand_terminates():
    """With reexpand_max_iterations=2 and a judge that ALWAYS distrusts every leaf, the
    confidence-driven loop must still TERMINATE (bounded by the per-node iteration cap
    plus the global node ceiling) rather than spin forever.

    Every leaf that completes gets a low score, so each is a re-expansion candidate. The
    per-node count cap stops a node re-expanding more than twice, and once
    max_total_nodes is hit no further growth is possible — so the full step loop drives
    to a natural stop within the step budget."""
    engine, expansion = _make_engine(
        conf_reexpand=True, threshold=0.9, max_iters=2,
        extra={"max_total_nodes": 12},
    )
    judge = _StepJudge(0.1)  # always below threshold
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    ops.judge_step_confidence = judge  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    steps_taken = await _drive(engine, graph, parent.node_id, max_steps=60)

    # It terminated within budget (did NOT hit the 60-step ceiling forever-loop guard).
    assert steps_taken < 60, "the confidence-reexpand loop must terminate, not spin"
    # And it respected the global node ceiling throughout.
    assert graph.node_count() <= 12, "node count must never exceed max_total_nodes"
    # No single node re-expanded more than the iteration cap.
    for node in graph.iter_depth_first():
        assert int(node.details.get("_got_reexpand_count", 0)) <= 2
