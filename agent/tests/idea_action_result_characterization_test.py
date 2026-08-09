"""Characterization (behavior-pinning) tests for the stateful result-handling
core of :class:`agent.app.idea_engine.IdeaDagEngine`.

These pin the CURRENT observable behavior of two private methods so the planned
extraction of `_handle_action_result` into `idea_action_result.py` (HANDOFF.md
debt #5b) can be proven behavior-preserving. They deliberately match whatever the
code does today and do not assert any "should be" behavior.

Cases covered:
  (a) VISIT success with empty content -> the Fix #9 retryable flip.
  (b) retryable failure with attempts remaining -> BLOCKED + cooldown set.
  (c) terminal / attempts-exhausted failure -> FAILED + error_details recorded.
  (d) successful result -> provides_data / post_execute contract assigned + DONE.
  (e) highest-scored pruned (SKIPPED) sibling revived exactly once per parent.

The methods are driven directly on a constructed engine (the idiom used by
`idea_engine_features_test`); the harness fakes mirror `idea_dag_recovery_test`.
"""
import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import (
    DetailKey,
    IdeaActionType,
    IdeaNodeStatus,
)
from agent.app.idea_policies.action_constants import (
    ActionResultBuilder,
    ActionResultKey,
)


class StubIO:
    """Minimal AgentIO stand-in (mirrors idea_engine_features_test)."""

    telemetry = None
    connector_chroma = None

    def set_telemetry(self, telemetry):
        return None


def _make_engine(extra_settings=None):
    """Construct an engine with the real default policies/registry.

    `_handle_action_result` resolves `self.actions.get(action_type)` to call
    `post_execute_provides`, so we keep the default `LeafActionRegistry` rather
    than faking it. Retry/recovery knobs are left at their JSON defaults
    (max_retries=2, retry_backoff_steps=1, visit_empty_content_retryable=True,
    sequential_sibling_recovery_enabled=True) unless overridden.
    """
    settings = {"allow_unscored_selection": True, "min_score_threshold": 0.0}
    if extra_settings:
        settings.update(extra_settings)
    return IdeaDagEngine(io=StubIO(), settings=settings)


# ---------------------------------------------------------------------------
# (a) VISIT action returning empty/insufficient content -> retryable flip
# ---------------------------------------------------------------------------


def test_visit_empty_content_flips_to_retryable_and_blocks():
    """Fix #9: a VISIT result claiming success but with no content is flipped to
    a retryable failure in-place, and (because attempts remain) the node goes
    BLOCKED with a cooldown rather than DONE."""
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Visit page",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    # success=True but empty content; no prior attempts.
    result = ActionResultBuilder.success(
        action=IdeaActionType.VISIT.value,
        url="https://example.com",
        content="",
    )
    node.details[DetailKey.ACTION_RESULT.value] = result

    status = engine._handle_action_result(graph, node.node_id, step_index=0)

    # The result dict is mutated in place: success flipped off, marked retryable.
    assert result[ActionResultKey.SUCCESS.value] is False
    assert result[ActionResultKey.RETRYABLE.value] is True
    assert result[ActionResultKey.ERROR_TYPE.value] == "ValidationError"
    # Falls through to the retry branch -> BLOCKED with a cooldown.
    assert status == "retry"
    refreshed = graph.get_node(node.node_id)
    assert refreshed.status == IdeaNodeStatus.BLOCKED
    assert refreshed.details[DetailKey.ACTION_COOLDOWN_UNTIL.value] == 1
    assert refreshed.details[DetailKey.ACTION_RETRYABLE.value] is True


def test_visit_empty_content_exhausted_attempts_fails():
    """Same empty-content flip, but with attempts already exhausted -> FAILED
    (the flip alone never forces DONE once the kill-switch default is on)."""
    engine = _make_engine()  # max_retries defaults to 2
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Visit page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_ATTEMPTS.value: 3,  # > max_retries (2)
        },
    )
    result = ActionResultBuilder.success(
        action=IdeaActionType.VISIT.value,
        url="https://example.com",
        content="   ",  # whitespace-only counts as empty
    )
    node.details[DetailKey.ACTION_RESULT.value] = result

    status = engine._handle_action_result(graph, node.node_id, step_index=0)

    assert result[ActionResultKey.SUCCESS.value] is False
    assert status == "failed"
    assert graph.get_node(node.node_id).status == IdeaNodeStatus.FAILED


# ---------------------------------------------------------------------------
# (b) retryable failure WITH attempts remaining -> BLOCKED + cooldown
# ---------------------------------------------------------------------------


def test_retryable_failure_with_attempts_remaining_blocks_with_cooldown():
    engine = _make_engine()  # max_retries=2, retry_backoff_steps=1
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Search something",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.ACTION_ATTEMPTS.value: 1,  # <= max_retries
        },
    )
    node.details[DetailKey.ACTION_RESULT.value] = ActionResultBuilder.failure(
        action=IdeaActionType.SEARCH.value,
        error="transient timeout",
        error_type="Timeout",
        retryable=True,
    )

    status = engine._handle_action_result(graph, node.node_id, step_index=5)

    assert status == "retry"
    refreshed = graph.get_node(node.node_id)
    assert refreshed.status == IdeaNodeStatus.BLOCKED
    # cooldown = step_index + max(1, backoff) = 5 + 1
    assert refreshed.details[DetailKey.ACTION_COOLDOWN_UNTIL.value] == 6
    assert refreshed.details[DetailKey.ACTION_RETRYABLE.value] is True


def test_retryable_failure_cooldown_uses_backoff_steps():
    """Cooldown honors a larger retry_backoff_steps setting."""
    engine = _make_engine({"action_retry_backoff_steps": 3})
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Search something",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.ACTION_ATTEMPTS.value: 0,
        },
    )
    node.details[DetailKey.ACTION_RESULT.value] = ActionResultBuilder.failure(
        action=IdeaActionType.SEARCH.value,
        error="transient",
        retryable=True,
    )

    engine._handle_action_result(graph, node.node_id, step_index=2)

    refreshed = graph.get_node(node.node_id)
    assert refreshed.status == IdeaNodeStatus.BLOCKED
    assert refreshed.details[DetailKey.ACTION_COOLDOWN_UNTIL.value] == 5  # 2 + 3


# ---------------------------------------------------------------------------
# (c) terminal / non-retryable (or attempts exhausted) -> FAILED + error_details
# ---------------------------------------------------------------------------


def test_non_retryable_failure_marks_failed_with_error_details():
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Visit blocked site",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    result = ActionResultBuilder.failure(
        action=IdeaActionType.VISIT.value,
        error="403 Forbidden",
        error_type="BlockedSite",
        retryable=False,
    )
    result[ActionResultKey.HTTP_STATUS.value] = 403
    result[ActionResultKey.ROOT_CAUSE.value] = "bot detection"
    result[ActionResultKey.TRACEBACK_SUMMARY.value] = "tb-summary"
    node.details[DetailKey.ACTION_RESULT.value] = result

    status = engine._handle_action_result(graph, node.node_id, step_index=0)

    assert status == "failed"
    refreshed = graph.get_node(node.node_id)
    assert refreshed.status == IdeaNodeStatus.FAILED
    assert refreshed.details[DetailKey.ACTION_ERROR.value] == "403 Forbidden"
    error_details = refreshed.details[f"{DetailKey.ACTION_ERROR.value}_details"]
    assert error_details["error"] == "403 Forbidden"
    assert error_details["error_type"] == "BlockedSite"
    assert error_details["root_cause"] == "bot detection"
    assert error_details["http_status"] == 403
    assert error_details["traceback_summary"] == "tb-summary"


def test_retryable_failure_attempts_exhausted_marks_failed():
    """A retryable failure whose attempts exceed max_retries is terminal."""
    engine = _make_engine()  # max_retries=2
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Search something",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.ACTION_ATTEMPTS.value: 3,  # > max_retries (2)
        },
    )
    node.details[DetailKey.ACTION_RESULT.value] = ActionResultBuilder.failure(
        action=IdeaActionType.SEARCH.value,
        error="still timing out",
        retryable=True,
    )

    status = engine._handle_action_result(graph, node.node_id, step_index=0)

    assert status == "failed"
    refreshed = graph.get_node(node.node_id)
    assert refreshed.status == IdeaNodeStatus.FAILED
    # retryable flag is still recorded on the node even though it failed.
    assert refreshed.details[DetailKey.ACTION_RETRYABLE.value] is True
    assert refreshed.details[DetailKey.ACTION_ERROR.value] == "still timing out"


def test_non_dict_result_marks_failed():
    """A non-dict action result short-circuits to FAILED."""
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Broken action",
        details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
    )
    node.details[DetailKey.ACTION_RESULT.value] = "not-a-dict"

    status = engine._handle_action_result(graph, node.node_id, step_index=0)

    assert status == "failed"
    assert graph.get_node(node.node_id).status == IdeaNodeStatus.FAILED


def test_missing_node_returns_missing():
    """An unknown node id short-circuits with the 'missing' sentinel."""
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    status = engine._handle_action_result(graph, "no-such-node", step_index=0)
    assert status == "missing"


# ---------------------------------------------------------------------------
# (d) successful result -> provides_data / post_execute contract assigned
# ---------------------------------------------------------------------------


def test_successful_visit_assigns_provides_data_contract_and_done():
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Visit page",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
    )
    node.details[DetailKey.ACTION_RESULT.value] = ActionResultBuilder.success(
        action=IdeaActionType.VISIT.value,
        url="https://example.com",
        content="Real page content here.",
        content_total_chars=23,
    )

    status = engine._handle_action_result(graph, node.node_id, step_index=0)

    assert status == "success"
    refreshed = graph.get_node(node.node_id)
    assert refreshed.status == IdeaNodeStatus.DONE
    # VisitLeafAction.post_execute_provides -> "urls_from_visit" when content present.
    assert refreshed.details[DetailKey.PROVIDES_DATA.value] == {"type": "urls_from_visit"}
    # VISIT success side-effects recorded on the node.
    assert refreshed.details["visit_url"] == "https://example.com"
    assert refreshed.details["visit_content_length"] == 23


def test_successful_search_assigns_provides_data_contract_and_done():
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Search something",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
    )
    node.details[DetailKey.ACTION_RESULT.value] = ActionResultBuilder.success(
        action=IdeaActionType.SEARCH.value,
        query="quantum",
        results=[{"title": "QC", "url": "https://qc.example"}],
    )

    status = engine._handle_action_result(graph, node.node_id, step_index=0)

    assert status == "success"
    refreshed = graph.get_node(node.node_id)
    assert refreshed.status == IdeaNodeStatus.DONE
    # SearchLeafAction.post_execute_provides -> "urls_from_search".
    assert refreshed.details[DetailKey.PROVIDES_DATA.value] == {"type": "urls_from_search"}


def test_successful_think_done_without_contract():
    """THINK has no post_execute contract: DONE but no provides_data tag."""
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "Think it over",
        details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
    )
    node.details[DetailKey.ACTION_RESULT.value] = ActionResultBuilder.success(
        action=IdeaActionType.THINK.value,
        thinking_content="some reasoning",
    )

    status = engine._handle_action_result(graph, node.node_id, step_index=0)

    assert status == "success"
    refreshed = graph.get_node(node.node_id)
    assert refreshed.status == IdeaNodeStatus.DONE
    assert DetailKey.PROVIDES_DATA.value not in refreshed.details


# ---------------------------------------------------------------------------
# (e) _recover_pruned_sibling: highest-scored pruned sibling revived once/parent
# ---------------------------------------------------------------------------


def _build_parent_with_pruned_siblings(graph, parent_id, failed_score=0.2):
    """Create a failed node plus two sequential-pruned SKIPPED siblings."""
    failed = graph.add_child(
        parent_id,
        "selected-but-failed",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
        score=failed_score,
        status=IdeaNodeStatus.FAILED,
    )
    low = graph.add_child(
        parent_id,
        "pruned-low",
        details={
            DetailKey.ACTION.value: IdeaActionType.THINK.value,
            "__sequential_pruned": True,
        },
        score=0.3,
        status=IdeaNodeStatus.SKIPPED,
    )
    high = graph.add_child(
        parent_id,
        "pruned-high",
        details={
            DetailKey.ACTION.value: IdeaActionType.THINK.value,
            "__sequential_pruned": True,
        },
        score=0.7,
        status=IdeaNodeStatus.SKIPPED,
    )
    return failed, low, high


def test_recover_revives_highest_scored_pruned_sibling():
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    parent_id = graph.root_id()
    failed, low, high = _build_parent_with_pruned_siblings(graph, parent_id)

    engine._recover_pruned_sibling(graph, failed, step_index=0)

    # Highest-scored pruned sibling is un-SKIPped to PENDING with a marker.
    high = graph.get_node(high.node_id)
    low = graph.get_node(low.node_id)
    assert high.status == IdeaNodeStatus.PENDING
    assert high.details["__sequential_recovery_invoked"] is True
    # The lower-scored sibling stays SKIPPED.
    assert low.status == IdeaNodeStatus.SKIPPED
    # Parent is marked so recovery only fires once.
    assert graph.get_node(parent_id).details["__sequential_recovery_used"] is True


def test_recover_fires_at_most_once_per_parent():
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    parent_id = graph.root_id()
    failed, low, high = _build_parent_with_pruned_siblings(graph, parent_id)

    engine._recover_pruned_sibling(graph, failed, step_index=0)
    # Re-skip the revived sibling to simulate a second failure cycle.
    high = graph.get_node(high.node_id)
    high.status = IdeaNodeStatus.SKIPPED
    engine._recover_pruned_sibling(graph, failed, step_index=1)

    # Second invocation is a no-op because the parent is already marked used.
    assert graph.get_node(high.node_id).status == IdeaNodeStatus.SKIPPED
    assert graph.get_node(low.node_id).status == IdeaNodeStatus.SKIPPED


def test_recover_skips_non_sequential_pruned_siblings():
    """A SKIPPED sibling lacking the __sequential_pruned marker is not revived."""
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    parent_id = graph.root_id()
    failed = graph.add_child(
        parent_id,
        "selected-but-failed",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
        score=0.2,
        status=IdeaNodeStatus.FAILED,
    )
    plain_skipped = graph.add_child(
        parent_id,
        "plain-skipped",
        details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
        score=0.9,
        status=IdeaNodeStatus.SKIPPED,
    )

    engine._recover_pruned_sibling(graph, failed, step_index=0)

    assert graph.get_node(plain_skipped.node_id).status == IdeaNodeStatus.SKIPPED
    assert "__sequential_recovery_used" not in graph.get_node(parent_id).details


def test_handle_failed_result_triggers_sibling_recovery_end_to_end():
    """Integration through _handle_action_result: a terminal failure with a
    pruned sibling present revives the sibling (recovery is gated on by default)."""
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    parent_id = graph.root_id()
    failed, low, high = _build_parent_with_pruned_siblings(graph, parent_id)
    # Give the failed node a terminal (non-retryable) result.
    failed.details[DetailKey.ACTION_RESULT.value] = ActionResultBuilder.failure(
        action=IdeaActionType.SEARCH.value,
        error="permanent",
        retryable=False,
    )

    status = engine._handle_action_result(graph, failed.node_id, step_index=0)

    assert status == "failed"
    assert graph.get_node(failed.node_id).status == IdeaNodeStatus.FAILED
    assert graph.get_node(high.node_id).status == IdeaNodeStatus.PENDING
    assert graph.get_node(parent_id).details["__sequential_recovery_used"] is True
