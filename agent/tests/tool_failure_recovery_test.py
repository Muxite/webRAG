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
from agent.app.idea_engine import IdeaDagEngine, _reformulate_multi_entity_query
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
# _reformulate_multi_entity_query — pure function
# ---------------------------------------------------------------------------
def test_reformulate_or_joins_multiple_quoted_phrases():
    query = '"Lake A" "Lake B" "Lake C" site:en.wikipedia.org'
    out = _reformulate_multi_entity_query(query)
    assert out == '"Lake A" OR "Lake B" OR "Lake C" site:en.wikipedia.org'


def test_reformulate_no_op_for_single_quoted_phrase():
    assert _reformulate_multi_entity_query('"Lake A" site:en.wikipedia.org') is None


def test_reformulate_no_op_for_unquoted_query():
    assert _reformulate_multi_entity_query("lake depth comparison wikipedia") is None


def test_reformulate_no_op_when_already_or_joined():
    already = '"Lake A" OR "Lake B" site:en.wikipedia.org'
    assert _reformulate_multi_entity_query(already) is None


def test_reformulate_no_op_for_empty_or_none():
    assert _reformulate_multi_entity_query("") is None
    assert _reformulate_multi_entity_query(None) is None


def test_reformulate_with_no_remainder_after_quotes():
    assert _reformulate_multi_entity_query('"Lake A" "Lake B"') == '"Lake A" OR "Lake B"'


# ---------------------------------------------------------------------------
# _reformulate_search_query_if_multi_entity + retry-loop wiring
# ---------------------------------------------------------------------------
def _search_node(graph: IdeaDag, query: str):
    return graph.add_child(
        graph.root_id(),
        "leaf",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.QUERY.value: query,
        },
    )


_EMPTY_SEARCH_RESULT = ActionResultBuilder.success(action="search", results=[])


@pytest.mark.asyncio
async def test_multi_entity_query_reformulated_before_retry():
    action = _CountingAction([_GOOD_SEARCH])
    engine = _make_engine(
        action,
        connector_retry_on_failure_enabled=True,
        connector_retry_backoff_seconds=0.0,
        connector_retry_max_attempts=2,
    )
    graph = IdeaDag(root_title="root")
    node = _search_node(graph, '"Lake A" "Lake B" "Lake C" site:en.wikipedia.org')

    out = await engine._maybe_retry_tool_failure(graph, node.node_id, action, _EMPTY_SEARCH_RESULT)

    assert out is _GOOD_SEARCH
    reformulated_query = graph.get_node(node.node_id).details[DetailKey.QUERY.value]
    assert reformulated_query == '"Lake A" OR "Lake B" OR "Lake C" site:en.wikipedia.org'


@pytest.mark.asyncio
async def test_single_entity_query_left_unchanged_on_retry():
    action = _CountingAction([_GOOD_SEARCH])
    engine = _make_engine(
        action,
        connector_retry_on_failure_enabled=True,
        connector_retry_backoff_seconds=0.0,
        connector_retry_max_attempts=2,
    )
    graph = IdeaDag(root_title="root")
    original_query = '"Lake A" site:en.wikipedia.org'
    node = _search_node(graph, original_query)

    await engine._maybe_retry_tool_failure(graph, node.node_id, action, _EMPTY_SEARCH_RESULT)

    assert graph.get_node(node.node_id).details[DetailKey.QUERY.value] == original_query


@pytest.mark.asyncio
async def test_non_search_tool_failure_query_not_reformulated():
    # A visit failure must never trigger query reformulation, even with a multi-quoted title.
    action = _CountingAction([_GOOD_VISIT])
    engine = _make_engine(
        action,
        connector_retry_on_failure_enabled=True,
        connector_retry_backoff_seconds=0.0,
    )
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "leaf",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.QUERY.value: '"Lake A" "Lake B" site:en.wikipedia.org',
        },
    )
    original_query = node.details[DetailKey.QUERY.value]

    await engine._maybe_retry_tool_failure(graph, node.node_id, action, _TRANSIENT_FAILURE)

    assert graph.get_node(node.node_id).details[DetailKey.QUERY.value] == original_query


@pytest.mark.asyncio
async def test_reformulation_is_idempotent_across_repeated_retries():
    # Always-empty search: reformulation should apply once on the first retry, then the
    # already-OR-joined query is left alone on the second retry attempt (no double-join).
    action = _CountingAction([_EMPTY_SEARCH_RESULT])
    engine = _make_engine(
        action,
        connector_retry_on_failure_enabled=True,
        connector_retry_backoff_seconds=0.0,
        connector_retry_max_attempts=2,
    )
    graph = IdeaDag(root_title="root")
    node = _search_node(graph, '"Lake A" "Lake B" site:en.wikipedia.org')

    await engine._maybe_retry_tool_failure(graph, node.node_id, action, _EMPTY_SEARCH_RESULT)

    assert graph.get_node(node.node_id).details[DetailKey.QUERY.value] == (
        '"Lake A" OR "Lake B" site:en.wikipedia.org'
    )


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


# ---------------------------------------------------------------------------
# Routing — the same suppression must also cover the follow-up-detector path
# (`_reexpand_check`, driven by `got_reexpand_enabled`), not just the
# confidence-trigger path. Regression for a composition gap: previously
# `tool_failure_recovery_enabled` only gated `_confidence_triggers_reexpand`,
# so a tool-failure leaf (e.g. an empty SEARCH) could still be handed to the
# follow-up detector and re-expanded — defeating "retry, don't re-expand".
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_followup_path_off_still_consults_detector_for_tool_failure():
    # Baseline (flag off): the follow-up detector is consulted even for a tool-failure
    # result — this is the pre-existing (unchanged) behavior.
    action = _CountingAction([_GOOD_SEARCH])
    engine = _make_engine(action, got_reexpand_enabled=True, got_reexpand_max_iterations=1)
    graph = IdeaDag(root_title="root")
    node = _node_with_result(graph, _EMPTY_SEARCH)

    calls = {"n": 0}

    async def _checker(graph, node_id, model_name=None):
        calls["n"] += 1
        return {"needs_followup": True, "reason": "stub"}

    from agent.app.got_operations import GoTOperations
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    ops.check_needs_followup = _checker  # type: ignore[assignment]
    engine._got = ops

    verdict = await engine._reexpand_check(graph, node.node_id, 0)
    assert calls["n"] == 1
    assert verdict is not None


@pytest.mark.asyncio
async def test_followup_path_on_suppresses_reexpand_for_tool_failure():
    # With BOTH `got_reexpand_enabled` and `tool_failure_recovery_enabled` on, a
    # tool-failure leaf must be gated out BEFORE the follow-up detector is even
    # consulted (mirrors the confidence-trigger suppression).
    action = _CountingAction([_GOOD_SEARCH])
    engine = _make_engine(
        action,
        got_reexpand_enabled=True,
        got_reexpand_max_iterations=1,
        tool_failure_recovery_enabled=True,
    )
    graph = IdeaDag(root_title="root")
    node = _node_with_result(graph, _EMPTY_SEARCH)

    calls = {"n": 0}

    async def _checker(graph, node_id, model_name=None):
        calls["n"] += 1
        return {"needs_followup": True, "reason": "stub"}

    from agent.app.got_operations import GoTOperations
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    ops.check_needs_followup = _checker  # type: ignore[assignment]
    engine._got = ops

    verdict = await engine._reexpand_check(graph, node.node_id, 0)
    assert verdict is None, "a tool-failure leaf must not re-expand when routing is on"
    assert calls["n"] == 0, "the follow-up detector must never be consulted for a tool failure"


@pytest.mark.asyncio
async def test_followup_path_on_still_reexpands_genuine_insufficiency():
    # Genuine content-insufficiency (not a tool failure) still reaches the detector.
    action = _CountingAction([_GOOD_SEARCH])
    engine = _make_engine(
        action,
        got_reexpand_enabled=True,
        got_reexpand_max_iterations=1,
        tool_failure_recovery_enabled=True,
    )
    graph = IdeaDag(root_title="root")
    node = _node_with_result(graph, _GOOD_VISIT)

    async def _checker(graph, node_id, model_name=None):
        return {"needs_followup": True, "reason": "stub"}

    from agent.app.got_operations import GoTOperations
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    ops.check_needs_followup = _checker  # type: ignore[assignment]
    engine._got = ops

    verdict = await engine._reexpand_check(graph, node.node_id, 0)
    assert verdict is not None
