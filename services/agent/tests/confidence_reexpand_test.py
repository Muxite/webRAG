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
        scores = {}
        for node_id in candidate_ids:
            graph.evaluate(node_id, 0.6)
            scores[node_id] = 0.6
        return scores


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


def _reexpanded_lineage(graph):
    reexpanded = [n for n in graph.iter_depth_first() if n.details.get("_got_reexpanded")]
    return sorted(reexpanded, key=lambda n: graph.depth(n.node_id))


@pytest.mark.asyncio
async def test_confidence_trigger_honors_lineage_budget_two_cycles():
    """The confidence trigger honors the SAME inherited lineage budget as the follow-up
    detector: with an always-distrusting judge and max_iterations=2, one lineage
    re-expands exactly twice, then the tip (inherited count == 2) stops."""
    engine, expansion = _make_engine(
        conf_reexpand=True, threshold=0.9, max_iters=2,
        extra={"max_total_nodes": 50},
    )
    judge = _StepJudge(0.1)  # always below threshold
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    ops.judge_step_confidence = judge  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await _drive(engine, graph, parent.node_id, max_steps=60)

    lineage = _reexpanded_lineage(graph)
    assert len(lineage) == 2, (
        f"confidence trigger must re-expand a lineage exactly twice at max_iterations=2, "
        f"got {len(lineage)}"
    )
    assert lineage[0].node_id == leaf.node_id
    assert graph.node_count() < 50, "the iteration knob (not max_total_nodes) bounds the lineage"


@pytest.mark.asyncio
async def test_two_iteration_confidence_reexpand_reaches_fixed_point():
    """With reexpand_max_iterations=2 and a judge that ALWAYS distrusts every leaf, the
    re-expansion machinery must reach a FIXED POINT — the lineage stops growing after its
    budget is spent, rather than re-expanding forever until max_total_nodes.

    (Whether the outer step() loop then returns None is a separate merge-finalization
    concern for a single-leaf parent; here we pin the property that matters for A2: the
    re-expansion itself terminates and respects both the iteration knob and the ceiling.)
    Every completing leaf gets a low score, so the ONLY thing that can halt the chain is
    the inherited lineage budget. We drive a batch of steps, snapshot the graph, drive
    more, and assert nothing further re-expanded."""
    engine, expansion = _make_engine(
        conf_reexpand=True, threshold=0.9, max_iters=2,
        extra={"max_total_nodes": 50},
    )
    judge = _StepJudge(0.1)  # always below threshold
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    ops.judge_step_confidence = judge  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await _drive(engine, graph, parent.node_id, max_steps=30)
    nodes_after_30 = graph.node_count()
    expands_after_30 = expansion.calls
    await _drive(engine, graph, parent.node_id, max_steps=30)

    # Fixed point: no further growth or re-expansion in the second batch of steps.
    assert graph.node_count() == nodes_after_30, "re-expansion must reach a fixed point"
    assert expansion.calls == expands_after_30
    # Exactly the iteration budget was spent on the single lineage — not the node ceiling.
    assert len(_reexpanded_lineage(graph)) == 2
    assert graph.node_count() < 50, "the iteration knob (not max_total_nodes) bounds growth"
    for node in graph.iter_depth_first():
        assert int(node.details.get("_got_reexpand_count", 0)) <= 2


@pytest.mark.asyncio
async def test_both_triggers_enabled_confidence_wins_no_double_reexpand():
    """When BOTH the confidence trigger and the follow-up detector are enabled and a
    completed leaf trips both (low confidence AND a positive follow-up verdict), the
    node re-expands EXACTLY ONCE. `_handle_leaf_node` runs the confidence trigger
    first; the follow-up path's read-only `_reexpand_check` then sees the node
    already has children (both checks guard on `node.children`) and skips it without
    even consulting the follow-up detector."""
    engine, expansion = _make_engine(
        conf_reexpand=True, threshold=0.5, followup_enabled=True, max_iters=1,
    )
    _attach_judge(engine, 0.1)  # below threshold -> confidence trigger fires

    followup_calls = {"count": 0}

    async def _followup_stub(*args, **kwargs):
        followup_calls["count"] += 1
        return {"needs_followup": True, "reason": "stub follow-up"}

    engine._got.check_needs_followup = _followup_stub  # type: ignore[assignment]

    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1, "only ONE re-expansion despite both triggers tripping"
    assert len(leaf.children) == 1
    assert leaf.details.get("_got_reexpand_count") == 1
    assert followup_calls["count"] == 0, (
        "the follow-up detector must never be consulted once the confidence trigger "
        "already gave the node children"
    )
    assert result == leaf.node_id
