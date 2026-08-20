"""The evaluation-ordering invariant: a decision that consumes a score runs AFTER that score.

Two engine sites violated it (ENGINE_DESIGN_REVIEW.md PART 3), and each is fixed behind its
own opt-in, default-OFF engine flag:

* ``beam_after_evaluation`` -- ``candidates[:max_branching]`` cut the candidate list at
  expansion time, before anything was scored, so the "beam" kept whichever candidates the
  model emitted first. On: every candidate becomes a node, the evaluator scores them all, and
  ``_apply_beam_after_evaluation`` keeps the top ``max_branching`` BY SCORE.
* ``evaluate_parallel_siblings`` -- the auto-parallel batch path executes every eligible child
  and returns before the evaluation block, so those siblings keep ``score is None`` forever
  (the documented root cause behind dormant prune/backtrack/confidence machinery). On: the
  batch is scored once its results are in.

Both flags-off cases are pinned too, since default byte-identity is the repo's opt-in
discipline. Everything here is offline: `plan_library_auto_shortcircuit_test`'s stub harness
with no LLM, no Chroma, and a stubbed action executor.
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import _BEAM_PRUNED_KEY, _PENDING_BEAM_KEY, IdeaDagEngine
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies.base import (
    DecompositionPolicy,
    DetailKey,
    EvaluationPolicy,
    ExpansionPolicy,
    IdeaActionType,
    IdeaNodeStatus,
)


# Deliberately WORST-FIRST in emission order: arrival-order truncation keeps "cand 0"/"cand 1",
# score-order truncation keeps "cand 3"/"cand 2".
_CANDIDATES = [
    {"title": f"cand {i}", "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}}
    for i in range(4)
]
_SCORE_BY_TITLE = {"cand 0": 0.1, "cand 1": 0.2, "cand 2": 0.7, "cand 3": 0.9}


class DummyIO:
    connector_chroma = None
    telemetry = None

    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    async def expand(self, graph, node_id, memories=None):
        return [json.loads(json.dumps(c)) for c in _CANDIDATES]


class FakeEvaluation(EvaluationPolicy):
    """Scores by title, and records the order it was asked in (for the ordering assertions)."""

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.batches = []

    async def evaluate(self, graph, node_id):
        score = _SCORE_BY_TITLE.get(graph.get_node(node_id).title, 0.5)
        graph.evaluate(node_id, score)
        return score

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        self.batches.append(
            [(graph.get_node(c).title, graph.get_node(c).details.get(DetailKey.ACTION_RESULT.value) is not None)
             for c in candidate_ids]
        )
        return {cid: await self.evaluate(graph, cid) for cid in candidate_ids}


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class _NoContractAction:
    """Only the hook `_handle_action_result` calls on a successful leaf."""

    def post_execute_provides(self, node, result):
        return None


class FakeActionRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type):
        return _NoContractAction()


def _make_engine(**overrides):
    settings = {
        "max_branching": 2,
        "got_dynamic_beam_enabled": False,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "auto_parallel_siblings": False,
        "semantic_dedup_visits_enabled": False,
        "plan_library_enabled": False,
    }
    settings.update(overrides)
    evaluation = FakeEvaluation(settings)
    engine = IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=FakeExpansion(settings),
        evaluation=evaluation,
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeActionRegistry(settings),
        post_expansion_hooks=[],
    )
    engine._got = None  # no GoT ops: keeps beam width == max_branching and skips embedding
    executed = []

    async def _fake_execute(graph, parent_id, node_id, *a, **kw):
        executed.append(node_id)
        node = graph.get_node(node_id)
        node.details[DetailKey.ACTION_RESULT.value] = {"success": True, "results": ["x"]}
        return node.details[DetailKey.ACTION_RESULT.value]

    engine._execute_action_guarded = _fake_execute
    engine._execute_action = _fake_execute
    return engine, evaluation, executed


def _graph():
    return IdeaDag(root_title="mandate", root_details={"mandate": "mandate"})


def _titles(graph, node_id):
    return [graph.get_node(c).title for c in graph.get_node(node_id).children]


# ---- beam_after_evaluation --------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_still_truncates_at_expansion_in_arrival_order():
    engine, _ev, _ex = _make_engine()
    graph = _graph()

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    assert _titles(graph, graph.root_id()) == ["cand 0", "cand 1"]
    assert _PENDING_BEAM_KEY not in graph.get_node(graph.root_id()).details


@pytest.mark.asyncio
async def test_flag_on_expands_every_candidate_and_defers_the_width():
    engine, _ev, _ex = _make_engine(beam_after_evaluation=True)
    graph = _graph()

    await engine._handle_expansion_node(graph, graph.root_id(), 0, None)

    assert _titles(graph, graph.root_id()) == ["cand 0", "cand 1", "cand 2", "cand 3"]
    assert graph.get_node(graph.root_id()).details[_PENDING_BEAM_KEY] == 2


@pytest.mark.asyncio
async def test_beam_keeps_the_highest_scored_candidates_not_the_first_emitted():
    engine, _ev, executed = _make_engine(beam_after_evaluation=True)
    graph = _graph()
    root = graph.root_id()

    await engine._handle_expansion_node(graph, root, 0, None)
    await engine._handle_intermediate_node(graph, root, 1, None)

    survivors = {
        graph.get_node(c).title
        for c in graph.get_node(root).children
        if graph.get_node(c).status != IdeaNodeStatus.SKIPPED
    }
    assert survivors == {"cand 2", "cand 3"}, "beam still cut by arrival order"
    # The two losers are kept, SKIPPED and attributable rather than deleted.
    losers = [graph.get_node(c) for c in graph.get_node(root).children if graph.get_node(c).title in {"cand 0", "cand 1"}]
    assert all(n.status == IdeaNodeStatus.SKIPPED and n.details[_BEAM_PRUNED_KEY] for n in losers)
    # And the child actually executed is the best-scored survivor.
    assert executed == [next(c for c in graph.get_node(root).children if graph.get_node(c).title == "cand 3")]


@pytest.mark.asyncio
async def test_deferred_width_is_consumed_once():
    engine, _ev, _ex = _make_engine(beam_after_evaluation=True)
    graph = _graph()
    root = graph.root_id()

    await engine._handle_expansion_node(graph, root, 0, None)
    await engine._handle_intermediate_node(graph, root, 1, None)

    assert _PENDING_BEAM_KEY not in graph.get_node(root).details


@pytest.mark.asyncio
async def test_unscored_child_loses_to_every_scored_one():
    engine, evaluation, _ex = _make_engine(beam_after_evaluation=True)
    graph = _graph()
    root = graph.root_id()

    async def _score_all_but_the_first(graph_, parent_id, candidate_ids):
        return {cid: await evaluation.evaluate(graph_, cid) for cid in candidate_ids[1:]}

    engine.evaluation.evaluate_batch = _score_all_but_the_first
    await engine._handle_expansion_node(graph, root, 0, None)
    await engine._handle_intermediate_node(graph, root, 1, None)

    kept = {
        graph.get_node(c).title
        for c in graph.get_node(root).children
        if graph.get_node(c).status != IdeaNodeStatus.SKIPPED
    }
    assert "cand 0" not in kept


@pytest.mark.asyncio
async def test_pending_beam_defers_the_auto_parallel_batch_for_one_step():
    """The batch path returns before evaluation, so it would execute the over-beam
    candidates and leave the beam inert. A pending width therefore forces the evaluating
    (sequential) path for that one step; survivors batch normally afterwards."""
    engine, _ev, executed = _make_engine(beam_after_evaluation=True, auto_parallel_siblings=True)
    graph = _graph()
    root = graph.root_id()

    await engine._handle_expansion_node(graph, root, 0, None)
    await engine._handle_intermediate_node(graph, root, 1, None)

    assert len(executed) == 1, "all four candidates ran before the beam could cut them"
    survivors = [
        c for c in graph.get_node(root).children
        if graph.get_node(c).status != IdeaNodeStatus.SKIPPED
    ]
    assert {graph.get_node(c).title for c in survivors} == {"cand 2", "cand 3"}

    # Next visit: width consumed, so the surviving sibling batches as usual.
    await engine._handle_intermediate_node(graph, root, 2, None)
    assert len(executed) == 2


# ---- evaluate_parallel_siblings ---------------------------------------------


async def _run_parallel_batch(**overrides):
    engine, evaluation, executed = _make_engine(auto_parallel_siblings=True, **overrides)
    graph = _graph()
    root = graph.root_id()
    await engine._handle_expansion_node(graph, root, 0, None)
    await engine._handle_intermediate_node(graph, root, 1, None)
    return graph, root, evaluation, executed


@pytest.mark.asyncio
async def test_parallel_batch_leaves_siblings_unscored_with_the_flag_off():
    graph, root, evaluation, executed = await _run_parallel_batch()

    assert len(executed) == 2, "the batch path should have run both children in one step"
    assert evaluation.batches == []
    assert all(graph.get_node(c).score is None for c in graph.get_node(root).children)


@pytest.mark.asyncio
async def test_parallel_batch_is_scored_after_execution_with_the_flag_on():
    graph, root, evaluation, executed = await _run_parallel_batch(evaluate_parallel_siblings=True)

    assert len(executed) == 2
    assert all(graph.get_node(c).score is not None for c in graph.get_node(root).children)
    # One batch call, and every sibling in it already carried its action result: the score
    # describes what the sibling produced, not what it promised.
    assert len(evaluation.batches) == 1
    assert all(has_result for _title, has_result in evaluation.batches[0])


@pytest.mark.asyncio
async def test_parallel_evaluation_failure_never_breaks_the_batch():
    engine, evaluation, executed = _make_engine(
        auto_parallel_siblings=True, evaluate_parallel_siblings=True
    )
    graph = _graph()
    root = graph.root_id()

    async def _boom(*a, **kw):
        raise RuntimeError("scorer down")

    engine.evaluation.evaluate_batch = _boom
    await engine._handle_expansion_node(graph, root, 0, None)
    assert await engine._handle_intermediate_node(graph, root, 1, None) == root
    assert len(executed) == 2
