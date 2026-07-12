"""Corrective re-expansion context (`got.reexpand_corrective_context_enabled`).

Today a re-expanded node's new expansion is BLIND to *why* the prior step was
inadequate (a low confidence score or a genuine follow-up), so it can re-target
the same wrong/insufficient page. This opt-in flag threads the triggering
judge/detector's own ``reason`` onto the re-expanded node as a single-use
``DetailKey.REEXPAND_REASON`` detail so the corrective hint reaches the
expansion call. Default OFF is byte-identical: the detail is never written.

Engine-side plumbing (this file) proves the reason reaches the expansion call
for BOTH triggers (confidence and follow-up); ``expansion_reexpand_reason_test.py``
proves the expansion policy's ``_build_messages`` actually surfaces it in the
prompt and consumes-and-clears it (single-use).
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


class CapturingExpansion(ExpansionPolicy):
    """Records the expanding node's details (incl. REEXPAND_REASON) at call time."""

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.calls = 0
        self.seen_details = []

    async def expand(self, graph: IdeaDag, node_id: str, memories=None):
        self.calls += 1
        node = graph.get_node(node_id)
        self.seen_details.append(dict(node.details) if node else {})
        return [
            {
                "title": "Follow-up: course-correct",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.THINK.value,
                    DetailKey.JUSTIFICATION.value: "corrective re-expansion",
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


def _make_engine(*, corrective_enabled, followup_enabled=False, conf_reexpand=False,
                  threshold=0.5, max_iters=1, extra=None):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_reexpand_enabled": followup_enabled,
        "got_reexpand_max_iterations": max_iters,
        "got_step_confidence_judge_enabled": conf_reexpand,
        "got_step_confidence_reexpand_enabled": conf_reexpand,
        "got_step_confidence_reexpand_threshold": threshold,
        "got_reexpand_corrective_context_enabled": corrective_enabled,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
    }
    if extra:
        settings.update(extra)
    expansion = CapturingExpansion(settings)
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


def _completed_leaf(graph: IdeaDag):
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    leaf = graph.add_child(
        parent.node_id,
        "Resolve a sub-fact",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.IS_LEAF.value: True},
    )
    return parent, leaf


@pytest.mark.asyncio
async def test_flag_off_follow_up_reexpand_never_threads_reason():
    """Regression: corrective-context flag OFF -> a follow-up-triggered re-expansion's
    expansion context is byte-identical (no REEXPAND_REASON detail reaches the call
    or lingers on the node)."""
    engine, expansion = _make_engine(corrective_enabled=False, followup_enabled=True)
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)

    async def _followup(*args, **kwargs):
        return {"needs_followup": True, "reason": "survivor points elsewhere"}

    ops.check_needs_followup = _followup  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1
    assert DetailKey.REEXPAND_REASON.value not in expansion.seen_details[0]
    assert DetailKey.REEXPAND_REASON.value not in leaf.details


@pytest.mark.asyncio
async def test_flag_off_confidence_reexpand_never_threads_reason():
    """Regression: corrective-context flag OFF -> a confidence-triggered re-expansion's
    expansion context is byte-identical."""
    engine, expansion = _make_engine(corrective_enabled=False, conf_reexpand=True, threshold=0.5)
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)

    async def _judge(graph, node_id, model_name=None):
        return {"confidence": 0.1, "reason": "stub judge reason"}

    ops.judge_step_confidence = _judge  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1
    assert DetailKey.REEXPAND_REASON.value not in expansion.seen_details[0]
    assert DetailKey.REEXPAND_REASON.value not in leaf.details


@pytest.mark.asyncio
async def test_followup_reexpand_threads_detector_reason_when_enabled():
    """Flag ON -> a follow-up-triggered re-expansion threads the follow-up detector's
    own reason onto the re-expanded node, visible to the expansion call."""
    engine, expansion = _make_engine(corrective_enabled=True, followup_enabled=True)
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)

    async def _followup(*args, **kwargs):
        return {"needs_followup": True, "reason": "survivor points to River Stour, Dorset"}

    ops.check_needs_followup = _followup  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1
    assert expansion.seen_details[0].get(DetailKey.REEXPAND_REASON.value) == (
        "survivor points to River Stour, Dorset"
    )


@pytest.mark.asyncio
async def test_confidence_reexpand_threads_synthesized_reason_when_enabled():
    """Flag ON -> a confidence-triggered re-expansion threads the confidence-loop's
    own synthesized low-confidence reason onto the re-expanded node.

    The confidence trigger constructs its own verdict `reason` (rather than passing
    the judge's raw text through unmodified) — see `_maybe_confidence_reexpand_batch` —
    so this pins that whatever reason drives `_apply_reexpand` reaches the node."""
    engine, expansion = _make_engine(corrective_enabled=True, conf_reexpand=True, threshold=0.5)
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)

    async def _judge(graph, node_id, model_name=None):
        return {"confidence": 0.2, "reason": "stub judge reason"}

    ops.judge_step_confidence = _judge  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1
    reason = expansion.seen_details[0].get(DetailKey.REEXPAND_REASON.value)
    assert reason is not None
    assert "low step confidence" in reason


@pytest.mark.asyncio
async def test_reason_not_threaded_when_verdict_reason_blank():
    """Flag ON but the triggering verdict's reason is blank/absent -> nothing threaded
    (mirrors the blank-value handling of HUMAN_FEEDBACK: no empty-string noise)."""
    engine, expansion = _make_engine(corrective_enabled=True, followup_enabled=True)
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)

    async def _followup(*args, **kwargs):
        return {"needs_followup": True, "reason": "   "}

    ops.check_needs_followup = _followup  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1
    assert DetailKey.REEXPAND_REASON.value not in expansion.seen_details[0]


@pytest.mark.asyncio
async def test_reason_detail_not_inherited_by_spawned_child():
    """The threaded reason is scoped to the re-expanded node's OWN expansion call only:
    the freshly spawned follow-up child (which inherits the lineage's
    `_got_reexpand_count` budget) must NOT also inherit the parent's REEXPAND_REASON
    detail — that would leak a stale corrective hint into an unrelated later expansion
    of the child."""
    engine, expansion = _make_engine(corrective_enabled=True, followup_enabled=True)
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)

    async def _followup(*args, **kwargs):
        return {"needs_followup": True, "reason": "first genuine follow-up"}

    ops.check_needs_followup = _followup  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert expansion.calls == 1
    assert expansion.seen_details[0].get(DetailKey.REEXPAND_REASON.value) == "first genuine follow-up"
    assert len(leaf.children) == 1
    child = graph.get_node(leaf.children[0])
    assert DetailKey.REEXPAND_REASON.value not in child.details
