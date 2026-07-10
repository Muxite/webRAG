"""Tests for the opt-in decorrelated per-step confidence judge
(`got.step_confidence_judge_enabled`).

Two layers:

* ``GoTOperations.judge_step_confidence`` — the LLM call itself: gated OFF by
  default, builds a *leak-free* prompt (never carries the grep validators or a
  ground-truth answer, which is the whole point of a decorrelated substrate) and
  clamps the returned confidence to ``[0, 1]``.
* ``IdeaDagEngine._maybe_judge_step_confidence`` — the loop hook: no-op (and no
  LLM call) when the flag is off, fires once per leaf-completion when on, honors
  ``sample_every`` subsampling, and never crashes on a bad/absent judge verdict.
"""
from __future__ import annotations

import json

import pytest

from agent.app.got_operations import GoTOperations
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
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


# ---------------------------------------------------------------------------
# GoTOperations.judge_step_confidence — the LLM call
# ---------------------------------------------------------------------------
class _CaptureIO:
    """Fake AgentIO that records the payload and returns a canned JSON response."""

    def __init__(self, response: str):
        self.response = response
        self.last_messages = None

    def set_telemetry(self, telemetry):
        return None

    def build_llm_payload(self, messages=None, json_mode=None, model_name=None, temperature=None):
        self.last_messages = messages
        return {"messages": messages, "json_mode": json_mode, "temperature": temperature}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None, timeout_seconds=None):
        return self.response


def _ops(response: str, **overrides):
    settings = {"got_step_confidence_judge_enabled": True}
    settings.update(overrides)
    return GoTOperations(settings=settings, io=_CaptureIO(response), memory_manager=None)


def _graph_with_done_leaf(content="Neruda was born in Parral in 1904.", mandate="Find the poet's birthplace."):
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = mandate
    leaf = g.add_child(
        g.root_id(),
        "Resolve the poet's birthplace",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.ACTION_RESULT.value: {
                ActionResultKey.SUCCESS.value: True,
                "content": content,
            },
        },
    )
    return g, leaf


@pytest.mark.asyncio
async def test_judge_disabled_by_default_returns_none_and_no_call():
    ops = GoTOperations(settings={}, io=_CaptureIO('{"confidence": 0.9}'), memory_manager=None)
    g, leaf = _graph_with_done_leaf()
    verdict = await ops.judge_step_confidence(g, leaf.node_id)
    assert verdict is None
    assert ops.io.last_messages is None, "no LLM payload should be built when the flag is off"


@pytest.mark.asyncio
async def test_judge_prompt_is_leak_free():
    # The judge must see only trajectory-visible context, never validator/label data.
    ops = _ops('{"confidence": 0.5, "reason": "plausible"}')
    g, leaf = _graph_with_done_leaf(
        content="PAGE TEXT: birthplace evidence here.",
        mandate="Find the birthplace.",
    )
    await ops.judge_step_confidence(g, leaf.node_id)
    blob = json.dumps(ops.io.last_messages)
    # The visible substrate IS included...
    assert "PAGE TEXT: birthplace evidence here." in blob
    assert "Find the birthplace." in blob
    # ...but nothing that would correlate the score with the final grep label.
    for leaked in ("grep_validations", "overall_passed", "overall_score", "validator", "ground_truth"):
        assert leaked not in blob, f"judge prompt leaked {leaked!r}"


@pytest.mark.asyncio
async def test_judge_clamps_confidence_above_one():
    ops = _ops('{"confidence": 1.7, "reason": "overconfident"}')
    g, leaf = _graph_with_done_leaf()
    verdict = await ops.judge_step_confidence(g, leaf.node_id)
    assert verdict == {"confidence": 1.0, "reason": "overconfident"}


@pytest.mark.asyncio
async def test_judge_clamps_confidence_below_zero():
    ops = _ops('{"confidence": -0.4}')
    g, leaf = _graph_with_done_leaf()
    verdict = await ops.judge_step_confidence(g, leaf.node_id)
    assert verdict["confidence"] == 0.0


@pytest.mark.asyncio
async def test_judge_returns_none_on_unparseable_confidence():
    ops = _ops('{"confidence": "not-a-number"}')
    g, leaf = _graph_with_done_leaf()
    assert await ops.judge_step_confidence(g, leaf.node_id) is None


@pytest.mark.asyncio
async def test_judge_returns_none_on_missing_confidence():
    ops = _ops('{"reason": "no score field"}')
    g, leaf = _graph_with_done_leaf()
    assert await ops.judge_step_confidence(g, leaf.node_id) is None


@pytest.mark.asyncio
async def test_judge_skips_unsuccessful_leaf():
    ops = _ops('{"confidence": 0.9}')
    g = IdeaDag(root_title="root")
    leaf = g.add_child(
        g.root_id(),
        "failed leaf",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {ActionResultKey.SUCCESS.value: False},
        },
    )
    assert await ops.judge_step_confidence(g, leaf.node_id) is None


# ---------------------------------------------------------------------------
# IdeaDagEngine._maybe_judge_step_confidence — the loop hook
# ---------------------------------------------------------------------------
class _FakeExpansion(ExpansionPolicy):
    def __init__(self, settings=None):
        super().__init__(settings=settings)

    async def expand(self, graph, node_id, memories=None):
        return []


class _FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        return {nid: 0.6 for nid in candidate_ids}


class _FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class _AlwaysSuccess(LeafAction):
    async def execute(self, graph, node_id, io):
        return {"action": IdeaActionType.THINK.value, "success": True}


class _FakeRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type):
        return _AlwaysSuccess(settings=self.settings)


class _DummyIO:
    def set_telemetry(self, telemetry):
        return None


class _JudgeStub:
    """Records calls and returns a canned verdict (or raises / returns None)."""

    def __init__(self, verdict, raises=False):
        self.verdict = verdict
        self.raises = raises
        self.calls = 0

    async def __call__(self, graph, node_id, model_name=None):
        self.calls += 1
        if self.raises:
            raise RuntimeError("judge boom")
        return self.verdict


def _make_engine(enabled: bool, sample_every: int = 1, extra=None):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_reexpand_enabled": False,
        "got_step_confidence_judge_enabled": enabled,
        "got_step_confidence_judge_sample_every": sample_every,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
    }
    if extra:
        settings.update(extra)
    engine = IdeaDagEngine(
        io=_DummyIO(),
        settings=settings,
        expansion=_FakeExpansion(settings),
        evaluation=_FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=_FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=_FakeRegistry(settings),
    )
    return engine


def _attach_judge(engine, verdict, raises=False):
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    stub = _JudgeStub(verdict, raises=raises)
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


def _parent_with_leaves(graph: IdeaDag, n=4):
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    leaves = []
    for i in range(n):
        leaf = graph.add_child(
            parent.node_id,
            f"candidate {i}",
            details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.IS_LEAF.value: True},
        )
        leaves.append(leaf)
    return parent, leaves


@pytest.mark.asyncio
async def test_hook_off_is_noop_and_no_judge_call():
    engine = _make_engine(enabled=False)
    stub = _attach_judge(engine, {"confidence": 0.8, "reason": "x"})
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert stub.calls == 0, "flag off -> the judge is never invoked"
    assert engine._step_confidences == []
    assert leaf.status == IdeaNodeStatus.DONE


@pytest.mark.asyncio
async def test_hook_on_records_confidence_per_completion():
    engine = _make_engine(enabled=True)
    stub = _attach_judge(engine, {"confidence": 0.72, "reason": "looks right"})
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 3, None)

    assert stub.calls == 1
    assert len(engine._step_confidences) == 1
    entry = engine._step_confidences[0]
    assert entry["confidence"] == 0.72
    assert entry["node_id"] == leaf.node_id
    assert entry["step"] == 3
    assert entry["kind"] == IdeaActionType.SEARCH.value


@pytest.mark.asyncio
async def test_hook_sample_every_subsamples_completions():
    # 4 siblings complete via the auto-parallel batch; sample_every=2 judges 2 of them.
    engine = _make_engine(
        enabled=True,
        sample_every=2,
        extra={"auto_parallel_siblings": True, "allow_execute_all_children": True},
    )
    stub = _attach_judge(engine, {"confidence": 0.5, "reason": "ok"})
    graph = IdeaDag(root_title="root")
    parent, leaves = _parent_with_leaves(graph, n=4)

    await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    assert engine._leaf_completion_count == 4, "every completion is counted"
    assert stub.calls == 2, "sample_every=2 judges completions 1 and 3 only"
    assert len(engine._step_confidences) == 2


@pytest.mark.asyncio
async def test_hook_tolerates_none_verdict():
    engine = _make_engine(enabled=True)
    stub = _attach_judge(engine, None)
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert stub.calls == 1
    assert engine._step_confidences == [], "a None verdict records nothing but does not crash"
    assert leaf.status == IdeaNodeStatus.DONE


@pytest.mark.asyncio
async def test_hook_tolerates_judge_exception():
    engine = _make_engine(enabled=True)
    stub = _attach_judge(engine, {"confidence": 0.5}, raises=True)
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    # Must not propagate — instrumentation never crashes a run.
    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert stub.calls == 1
    assert engine._step_confidences == []
    assert leaf.status == IdeaNodeStatus.DONE
