"""Tests for C1a tool-failure recovery (opt-in; default byte-identical).

Two mechanisms, each behind its own flag:
  * ``connector_retry_on_failure_enabled`` — bounded in-place retry of a TOOL failure
    (empty/timeout/HTTP-error fetch, no search results) so a transient failure recovers at
    the source instead of surfacing as empty grounding.
  * ``tool_failure_recovery_enabled`` — route the low step-confidence re-expansion trigger
    AWAY from a leaf whose low score was caused by a tool failure (a fresh subtree would just
    repeat the failing fetch).

Plus the shared ``ActionResultExtractor.is_tool_failure`` predicate.
"""
from __future__ import annotations

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.action_constants import ActionResultBuilder, ActionResultExtractor
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
# is_tool_failure predicate
# ---------------------------------------------------------------------------
def test_is_tool_failure_missing_or_failure():
    assert ActionResultExtractor.is_tool_failure(None) is True
    assert ActionResultExtractor.is_tool_failure("nope") is True
    assert ActionResultExtractor.is_tool_failure(
        ActionResultBuilder.failure(action="visit", error="timeout", retryable=True)
    ) is True


def test_is_tool_failure_empty_search_is_failure():
    empty = ActionResultBuilder.success(action="search", results=[])
    assert ActionResultExtractor.is_tool_failure(empty) is True
    nonempty = ActionResultBuilder.success(action="search", results=[{"url": "x"}])
    assert ActionResultExtractor.is_tool_failure(nonempty) is False


def test_is_tool_failure_empty_visit_is_failure():
    empty = ActionResultBuilder.success(action="visit", content="   ")
    assert ActionResultExtractor.is_tool_failure(empty) is True
    empty_full = ActionResultBuilder.success(action="visit", content="", content_full="")
    assert ActionResultExtractor.is_tool_failure(empty_full) is True
    real = ActionResultBuilder.success(action="visit", content="Neruda was born in 1904.")
    assert ActionResultExtractor.is_tool_failure(real) is False


def test_is_tool_failure_other_success_not_a_failure():
    # A think/save success with no search/visit payload is not a tool failure.
    assert ActionResultExtractor.is_tool_failure(
        ActionResultBuilder.success(action="think")
    ) is False


# ---------------------------------------------------------------------------
# Engine construction + fakes
# ---------------------------------------------------------------------------
class DummyIO:
    def set_telemetry(self, telemetry):
        return None


class _FakeExpansion(ExpansionPolicy):
    async def expand(self, graph, node_id, memories=None):
        return []


class _FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        for nid in candidate_ids:
            graph.evaluate(nid, 0.6)
        return {nid: 0.6 for nid in candidate_ids}


class _FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class _CountingAction(LeafAction):
    """Executes a scripted sequence of results, counting calls."""

    def __init__(self, outcomes, settings=None):
        super().__init__(settings=settings)
        self._outcomes = list(outcomes)
        self.calls = 0

    async def execute(self, graph, node_id, io):
        self.calls += 1
        idx = min(self.calls - 1, len(self._outcomes) - 1)
        return self._outcomes[idx]


class _FakeRegistry:
    def __init__(self, action):
        self._action = action

    def get(self, action_type):
        return self._action


def _make_engine(action, **flags):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
    }
    settings.update(flags)
    return IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=_FakeExpansion(settings),
        evaluation=_FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=_FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=_FakeRegistry(action),
    )


_EMPTY_SEARCH = ActionResultBuilder.success(action="search", results=[])
_GOOD_SEARCH = ActionResultBuilder.success(action="search", results=[{"url": "https://x"}])
_PERM_FAILURE = ActionResultBuilder.failure(action="visit", error="403", retryable=False)
_TRANSIENT_FAILURE = ActionResultBuilder.failure(action="visit", error="timeout", retryable=True)
_GOOD_VISIT = ActionResultBuilder.success(action="visit", content="real page content")


# ---------------------------------------------------------------------------
# _maybe_retry_tool_failure — connector_retry_on_failure_enabled
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retry_flag_off_is_byte_identical():
    action = _CountingAction([_EMPTY_SEARCH, _GOOD_SEARCH])
    engine = _make_engine(action)  # flag off
    graph = IdeaDag(root_title="root")
    out = await engine._maybe_retry_tool_failure(graph, "n", action, _EMPTY_SEARCH)
    # No retry: the initial (empty) result is returned unchanged and execute is never re-called.
    assert out is _EMPTY_SEARCH
    assert action.calls == 0


@pytest.mark.asyncio
async def test_retry_recovers_transient_empty_then_success():
    action = _CountingAction([_GOOD_SEARCH])  # a re-execute returns good results
    engine = _make_engine(
        action,
        connector_retry_on_failure_enabled=True,
        connector_retry_backoff_seconds=0.0,
        connector_retry_max_attempts=2,
    )
    graph = IdeaDag(root_title="root")
    out = await engine._maybe_retry_tool_failure(graph, "n", action, _EMPTY_SEARCH)
    assert out is _GOOD_SEARCH, "a transient empty result recovers on retry"
    assert action.calls == 1, "one retry sufficed"


@pytest.mark.asyncio
async def test_retry_is_bounded_when_always_failing():
    action = _CountingAction([_EMPTY_SEARCH])  # always empty
    engine = _make_engine(
        action,
        connector_retry_on_failure_enabled=True,
        connector_retry_backoff_seconds=0.0,
        connector_retry_max_attempts=2,
    )
    graph = IdeaDag(root_title="root")
    out = await engine._maybe_retry_tool_failure(graph, "n", action, _EMPTY_SEARCH)
    assert ActionResultExtractor.is_tool_failure(out) is True
    assert action.calls == 2, "retry is bounded to max_attempts, never loops"


@pytest.mark.asyncio
async def test_permanent_failure_is_not_retried():
    action = _CountingAction([_GOOD_VISIT])
    engine = _make_engine(
        action,
        connector_retry_on_failure_enabled=True,
        connector_retry_backoff_seconds=0.0,
    )
    graph = IdeaDag(root_title="root")
    out = await engine._maybe_retry_tool_failure(graph, "n", action, _PERM_FAILURE)
    assert out is _PERM_FAILURE, "a permanent (non-retryable) failure is not retried"
    assert action.calls == 0


@pytest.mark.asyncio
async def test_real_content_is_not_retried():
    action = _CountingAction([_GOOD_VISIT])
    engine = _make_engine(action, connector_retry_on_failure_enabled=True)
    graph = IdeaDag(root_title="root")
    out = await engine._maybe_retry_tool_failure(graph, "n", action, _GOOD_VISIT)
    assert out is _GOOD_VISIT
    assert action.calls == 0, "a page that returned real content is never retried"


@pytest.mark.asyncio
async def test_transient_failure_retried_then_recovers():
    action = _CountingAction([_GOOD_VISIT])
    engine = _make_engine(
        action,
        connector_retry_on_failure_enabled=True,
        connector_retry_backoff_seconds=0.0,
    )
    graph = IdeaDag(root_title="root")
    out = await engine._maybe_retry_tool_failure(graph, "n", action, _TRANSIENT_FAILURE)
    assert out is _GOOD_VISIT
    assert action.calls == 1


# ---------------------------------------------------------------------------
# Routing — tool_failure_recovery_enabled suppresses reexpand on tool failure
# ---------------------------------------------------------------------------
def _node_with_result(graph: IdeaDag, result):
    node = graph.add_child(
        graph.root_id(),
        "leaf",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.ACTION_RESULT.value: result,
        },
    )
    return node


def test_routing_off_reexpands_tool_failure_as_before():
    # Flag off: a low-confidence tool-failure leaf still qualifies for re-expansion (unchanged).
    action = _CountingAction([_GOOD_SEARCH])
    engine = _make_engine(action, got_step_confidence_reexpand_enabled=True,
                          got_step_confidence_reexpand_threshold=0.5)
    graph = IdeaDag(root_title="root")
    node = _node_with_result(graph, _EMPTY_SEARCH)
    assert engine._confidence_triggers_reexpand(graph, node.node_id, 0.1) is True


def test_routing_on_suppresses_reexpand_for_tool_failure():
    action = _CountingAction([_GOOD_SEARCH])
    engine = _make_engine(action, got_step_confidence_reexpand_enabled=True,
                          got_step_confidence_reexpand_threshold=0.5,
                          tool_failure_recovery_enabled=True)
    graph = IdeaDag(root_title="root")
    node = _node_with_result(graph, _EMPTY_SEARCH)
    # A tool failure must NOT re-expand when routing is on (retry is the right recovery).
    assert engine._confidence_triggers_reexpand(graph, node.node_id, 0.1) is False


def test_routing_on_still_reexpands_genuine_insufficiency():
    # A page that genuinely loaded with content (not a tool failure) still re-expands.
    action = _CountingAction([_GOOD_SEARCH])
    engine = _make_engine(action, got_step_confidence_reexpand_enabled=True,
                          got_step_confidence_reexpand_threshold=0.5,
                          tool_failure_recovery_enabled=True)
    graph = IdeaDag(root_title="root")
    node = _node_with_result(graph, _GOOD_VISIT)
    assert engine._confidence_triggers_reexpand(graph, node.node_id, 0.1) is True
