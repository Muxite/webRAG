"""Tests for the novelty / churn guard (opt-in; default byte-identical).

Phase 0's `graph_no_reexpand` axis (docs/DAG_V3_LEDGER_MASTER_PLAN_2026-08-25.md section 3). Two
layers: the pure key/counter module (``agent/app/novelty_guard.py``) and its pre-dispatch veto in
``IdeaDagEngine._maybe_block_repeated_action``.

The load-bearing claims:

* flag off (the shipped default) dispatches exactly as before — no key computed, no attempt
  counted, no node ever refused;
* the key is ARGUMENT-level, so re-wording a sub-goal around the same call does not buy a fresh
  budget, while a genuinely different URL/query does;
* the budget is spent only by NO-PROGRESS attempts: new evidence since the last attempt of a key
  resets it, so a productive step is never blocked;
* progress counts per BRANCH, not run-wide: a sibling branch that keeps learning must not refresh
  a dead end's budget (the regression that made the guard a no-op on multi-branch runs);
* a blocked action fails without spending a tool call;
* an armed run reports what it blocked in ``final_payload["novelty_guard"]``.
"""
from __future__ import annotations

import pytest

from agent.app import novelty_guard as ng
from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.action_constants import ActionResultBuilder
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.config import IdeaConfig

from agent.tests.tool_failure_recovery_test import _CountingAction, _make_engine


# ---------------------------------------------------------------------------
# novelty_key / canonical_target
# ---------------------------------------------------------------------------
def test_the_key_is_the_argument_not_the_title():
    """Churn is argument-level: the sub-goal gets re-worded, the call does not."""
    first = ng.novelty_key("visit", {"url": "https://example.com/a", "goal": "find the date"})
    second = ng.novelty_key("visit", {"url": "https://example.com/a", "goal": "TRY AGAIN: date?"})
    assert first == second


def test_distinct_targets_are_distinct_keys():
    a = ng.novelty_key("visit", {"url": "https://example.com/a"})
    b = ng.novelty_key("visit", {"url": "https://example.com/b"})
    search = ng.novelty_key("search", {"query": "example a"})
    assert len({a, b, search}) == 3


def test_url_normalization_is_case_slash_and_fragment_only():
    base = ng.novelty_key("visit", {"url": "https://example.com/a"})
    assert ng.novelty_key("visit", {"url": "https://Example.com/A/"}) == base
    assert ng.novelty_key("visit", {"url": "https://example.com/a#section"}) == base
    # Conservative on purpose: a different query string is a different target, not a merge.
    assert ng.novelty_key("visit", {"url": "https://example.com/a?p=2"}) != base


def test_query_normalization_collapses_whitespace_and_case():
    assert ng.novelty_key("search", {"query": "  Erie   Canal "}) == ng.novelty_key(
        "search", {"query": "erie canal"}
    )


def test_the_unresolved_requirement_set_is_part_of_the_key():
    """Re-issuing a step after the run's open requirements moved is a different step."""
    same = ng.novelty_key("search", {"query": "q"}, ["a", "b"])
    reordered = ng.novelty_key("search", {"query": "q"}, ["b", "a", "b"])
    moved = ng.novelty_key("search", {"query": "q"}, ["a"])
    assert same == reordered      # set semantics, stable order
    assert same != moved


def test_an_argumentless_action_falls_back_to_its_goal():
    assert ng.novelty_key("think", {"goal": "Summarize"}) == ng.novelty_key("think", {"goal": "summarize"})
    assert ng.novelty_key("think", {}) == "think||"


# ---------------------------------------------------------------------------
# NoveltyGuard counting
# ---------------------------------------------------------------------------
def test_the_third_no_progress_attempt_is_blocked():
    guard = ng.NoveltyGuard(max_attempts=2)
    assert guard.is_blocked("k", 0) is False
    guard.record_attempt("k", 0)
    assert guard.is_blocked("k", 0) is False
    guard.record_attempt("k", 0)
    assert guard.is_blocked("k", 0) is True


def test_new_evidence_resets_the_budget():
    guard = ng.NoveltyGuard(max_attempts=2)
    guard.record_attempt("k", 0)
    guard.record_attempt("k", 0)
    assert guard.is_blocked("k", 0) is True
    # The watermark moved (some other branch learned something): this key is live again.
    assert guard.is_blocked("k", 1) is False
    assert guard.record_attempt("k", 1) == 1
    assert guard.is_blocked("k", 1) is False


def test_a_zero_threshold_disables_blocking_entirely():
    guard = ng.NoveltyGuard(max_attempts=0)
    for _ in range(5):
        guard.record_attempt("k", 0)
    assert guard.is_blocked("k", 0) is False


def test_keys_are_counted_independently():
    guard = ng.NoveltyGuard(max_attempts=2)
    for _ in range(3):
        guard.record_attempt("a", 0)
    assert guard.is_blocked("a", 0) is True
    assert guard.is_blocked("b", 0) is False


def test_reset_clears_the_run_state():
    guard = ng.NoveltyGuard(max_attempts=1)
    guard.record_attempt("k", 0)
    assert guard.is_blocked("k", 0) is True
    guard.reset()
    assert guard.is_blocked("k", 0) is False


# ---------------------------------------------------------------------------
# evidence_watermark
# ---------------------------------------------------------------------------
def test_the_watermark_counts_successful_results_when_the_evidence_store_is_off():
    graph = IdeaDag(root_title="root")
    assert ng.evidence_watermark(graph) == 0
    graph.add_child(
        graph.root_id(), "visit",
        details={DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(action="visit", content="text")},
    )
    assert ng.evidence_watermark(graph) == 1
    graph.add_child(
        graph.root_id(), "failed visit",
        details={DetailKey.ACTION_RESULT.value: ActionResultBuilder.failure(action="visit", error="403")},
    )
    assert ng.evidence_watermark(graph) == 1     # a failure is not evidence


def test_the_watermark_counts_the_evidence_and_claim_sidecars():
    graph = IdeaDag(root_title="root")
    graph.add_child(
        graph.root_id(), "visit",
        details={
            DetailKey.EVIDENCE.value: {"url": "https://x"},
            DetailKey.CLAIMS.value: [{"text": "a"}, {"text": "b"}],
        },
    )
    assert ng.evidence_watermark(graph) == 3     # one Evidence + two Claims


def test_the_watermark_is_scoped_to_the_nodes_branch():
    """Evidence on a SIBLING branch is not this branch's progress."""
    graph = IdeaDag(root_title="root")
    branch_a = graph.add_child(graph.root_id(), "branch A")
    branch_b = graph.add_child(graph.root_id(), "branch B")
    stuck = graph.add_child(
        branch_b.node_id, "the dead end",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value, "url": "https://x"},
    )
    graph.add_child(
        branch_a.node_id, "branch A learned something",
        details={DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
            action="visit", content="text")},
    )
    assert ng.evidence_watermark(graph) == 1                      # run-wide
    assert ng.evidence_watermark(graph, stuck.node_id) == 0       # branch B's own
    graph.add_child(
        branch_b.node_id, "branch B learned something",
        details={DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
            action="visit", content="text")},
    )
    assert ng.evidence_watermark(graph, stuck.node_id) == 1


def test_the_branch_scope_is_the_roots_own_child():
    graph = IdeaDag(root_title="root")
    branch = graph.add_child(graph.root_id(), "branch")
    deep = graph.add_child(graph.add_child(branch.node_id, "mid").node_id, "leaf")
    assert ng.branch_scope_id(graph, deep.node_id) == branch.node_id
    assert ng.branch_scope_id(graph, branch.node_id) == branch.node_id
    # The root has no narrower scope than the whole graph, and neither does an unknown node.
    assert ng.branch_scope_id(graph, graph.root_id()) is None
    assert ng.branch_scope_id(graph, "nope") is None


def test_the_sub_goal_scope_is_the_nearest_non_action_ancestor():
    """The coarse scope groups one sub-goal's retries; unrelated sub-goals stay apart."""
    graph = IdeaDag(root_title="root")
    sub_goal = graph.add_child(graph.root_id(), "resolve dam X")
    other_sub_goal = graph.add_child(graph.root_id(), "resolve dam Y")
    visit = graph.add_child(
        sub_goal.node_id, "visit trap A",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value, "url": "https://x/a"},
    )
    # A retry authored UNDER the failed action node still belongs to the same sub-goal.
    retry = graph.add_child(
        visit.node_id, "search another phrasing",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, "query": "dam X height"},
    )
    assert ng.sub_goal_scope_id(graph, visit.node_id) == sub_goal.node_id
    assert ng.sub_goal_scope_id(graph, retry.node_id) == sub_goal.node_id
    assert ng.sub_goal_scope_id(graph, other_sub_goal.node_id) == graph.root_id()
    assert ng.sub_goal_scope_id(
        graph, graph.add_child(
            other_sub_goal.node_id, "visit Y",
            details={DetailKey.ACTION.value: IdeaActionType.VISIT.value, "url": "https://y"},
        ).node_id
    ) == other_sub_goal.node_id
    assert ng.sub_goal_scope_id(graph, graph.root_id()) is None
    assert ng.sub_goal_scope_id(graph, "nope") is None


def test_the_sub_goal_scope_of_a_flat_plan_is_the_root():
    """The shape the engine actually produces: every action node a direct child of the root.

    ``branch_scope_id`` gives each such node ITSELF as its scope, so nothing can ever accumulate
    across them; the sub-goal scope is the root, which is what makes the retries one budget.
    """
    graph = IdeaDag(root_title="root")
    first = graph.add_child(
        graph.root_id(), "search one phrasing",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, "query": "victoria dam"},
    )
    second = graph.add_child(
        graph.root_id(), "search another phrasing",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                 "query": "victoria dam cape town height"},
    )
    assert ng.branch_scope_id(graph, first.node_id) != ng.branch_scope_id(graph, second.node_id)
    assert ng.sub_goal_scope_id(graph, first.node_id) == graph.root_id()
    assert ng.sub_goal_scope_id(graph, second.node_id) == graph.root_id()


def test_an_explicit_watermark_scope_overrides_the_branch():
    graph = IdeaDag(root_title="root")
    branch_a = graph.add_child(graph.root_id(), "branch A")
    branch_b = graph.add_child(graph.root_id(), "branch B")
    stuck = graph.add_child(
        branch_b.node_id, "the dead end",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value, "url": "https://x"},
    )
    graph.add_child(
        branch_a.node_id, "branch A learned something",
        details={DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
            action="visit", content="text")},
    )
    assert ng.evidence_watermark(graph, stuck.node_id) == 0
    # ``None`` is an explicit whole-graph scope, not "no argument given".
    assert ng.evidence_watermark(graph, stuck.node_id, scope_id=None) == 1
    assert ng.evidence_watermark(graph, stuck.node_id, scope_id=branch_a.node_id) == 1


def test_a_root_level_action_falls_back_to_the_whole_graph():
    graph = IdeaDag(root_title="root")
    graph.add_child(
        graph.root_id(), "visit",
        details={DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
            action="visit", content="text")},
    )
    assert ng.evidence_watermark(graph, graph.root_id()) == 1


def test_the_watermark_never_raises_on_a_broken_graph():
    class Broken:
        def iter_breadth_first(self):
            raise RuntimeError("no")

    assert ng.evidence_watermark(Broken()) == 0


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_the_flags_ship_absent_and_therefore_off():
    settings = load_idea_dag_settings()
    assert "run_policy_novelty_guard_enabled" not in settings
    run_policy = IdeaConfig.from_settings(settings).run_policy
    assert run_policy.novelty_guard_enabled is False
    assert run_policy.novelty_guard_max_attempts == 2


# ---------------------------------------------------------------------------
# engine veto
# ---------------------------------------------------------------------------
def _graph_with_visit(url: str = "https://example.com/a") -> tuple:
    graph = IdeaDag(root_title="root", root_details={"mandate": "Research"})
    node = graph.add_child(
        graph.root_id(), "visit it",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value, "url": url},
    )
    return graph, node.node_id


@pytest.mark.asyncio
async def test_the_veto_is_inert_with_the_flag_off():
    engine = _make_engine(_CountingAction([]))
    graph, node_id = _graph_with_visit()
    for _ in range(5):
        assert engine._maybe_block_repeated_action(graph, node_id) is None
    # Nothing was even instantiated: no per-run counter state exists on the engine.
    assert getattr(engine, "_novelty_guard", None) is None


@pytest.mark.asyncio
async def test_the_veto_blocks_the_third_identical_no_progress_attempt():
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph, node_id = _graph_with_visit()
    assert engine._maybe_block_repeated_action(graph, node_id) is None
    assert engine._maybe_block_repeated_action(graph, node_id) is None
    blocked = engine._maybe_block_repeated_action(graph, node_id)
    assert blocked is not None
    assert blocked["success"] is False
    assert blocked["novelty_blocked"] is True
    assert blocked["retryable"] is False
    assert graph.get_node(node_id).status == IdeaNodeStatus.FAILED
    assert graph.get_node(node_id).details[DetailKey.ACTION_RESULT.value] == blocked


@pytest.mark.asyncio
async def test_progress_on_another_branch_does_not_unblock_a_dead_end():
    """The regression: the watermark is BRANCH-scoped, not run-scoped.

    A multi-branch run (the shape the guard exists for --
    ``agent/app/idea_tests/test_305_mech_dead_end_retry_cap.py`` has three resolvable branches
    plus one deliberate dead end) kept the run-wide watermark climbing off the HEALTHY branches,
    which read as "new evidence appeared" for the stuck one and meant the dead end was never
    blocked at all.
    """
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph = IdeaDag(root_title="root", root_details={"mandate": "Research"})
    branch_a = graph.add_child(graph.root_id(), "branch A")
    branch_b = graph.add_child(graph.root_id(), "branch B")
    dead_end = graph.add_child(
        branch_b.node_id, "visit the dead end",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                 "url": "https://example.com/dead-end"},
    )
    blocked = None
    for i in range(3):
        blocked = engine._maybe_block_repeated_action(graph, dead_end.node_id)
        if blocked is not None:
            break
        # Branch A keeps learning things that have nothing to do with branch B's requirement.
        graph.add_child(
            branch_a.node_id, f"branch A progress {i}",
            details={DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
                action="visit", content="a page that resolves branch A")},
        )
    assert blocked is not None
    assert blocked["novelty_blocked"] is True


@pytest.mark.asyncio
async def test_progress_on_the_same_branch_still_unblocks():
    """The other half: a branch that IS learning keeps its budget refreshed."""
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph = IdeaDag(root_title="root", root_details={"mandate": "Research"})
    branch = graph.add_child(graph.root_id(), "branch B")
    node = graph.add_child(
        branch.node_id, "visit it",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                 "url": "https://example.com/a"},
    )
    for i in range(6):
        assert engine._maybe_block_repeated_action(graph, node.node_id) is None
        graph.add_child(
            branch.node_id, f"progress {i}",
            details={DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
                action="visit", content="new page")},
        )


@pytest.mark.asyncio
async def test_a_different_url_is_not_blocked_while_the_sub_goal_is_moving():
    """A burned target does not condemn its sibling -- as long as the scope is learning."""
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph, node_id = _graph_with_visit()
    for _ in range(3):
        engine._maybe_block_repeated_action(graph, node_id)
    graph.add_child(
        graph.root_id(), "something worked",
        details={DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
            action="visit", content="a real page")},
    )
    other = graph.add_child(
        graph.root_id(), "visit elsewhere",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value, "url": "https://example.com/b"},
    )
    assert engine._maybe_block_repeated_action(graph, other.node_id) is None


# ---------------------------------------------------------------------------
# the sub-goal-scoped second key (Finding A: the strict per-target key fans out)
# ---------------------------------------------------------------------------
def _flat_search_node(graph, query: str):
    return graph.add_child(
        graph.root_id(), f"search: {query}",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, "query": query},
    )


@pytest.mark.asyncio
async def test_distinct_targets_for_one_sub_goal_share_a_budget():
    """Task 305's actual shape: one dead end, several wordings, none of them ever striking.

    Every phrasing is its own strict key AND (flat plan) its own branch scope, so the per-target
    counter can watch the whole run burn on one sub-goal and never reach 2. The sub-goal-scoped
    key is what accumulates across them.
    """
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph = IdeaDag(root_title="root", root_details={"mandate": "Research"})
    phrasings = [
        "victoria dam height",
        "victoria dam cape town height",
        "table mountain victoria dam",
    ]
    results = [
        engine._maybe_block_repeated_action(graph, _flat_search_node(graph, q).node_id)
        for q in phrasings
    ]
    assert results[0] is None and results[1] is None
    assert results[2] is not None
    assert results[2]["novelty_blocked"] is True
    assert results[2]["novelty_block_scope"] == "sub_goal"
    # Every strict key is still a first attempt: the strict gate alone could not have fired.
    assert engine._novelty_guard.attempts(
        f"{ng.branch_scope_id(graph, graph.get_node(graph.root_id()).children[-1])}::"
        f"{ng.novelty_key(IdeaActionType.SEARCH.value, {'query': phrasings[-1]})}"
    ) <= 1


@pytest.mark.asyncio
async def test_a_different_sub_goal_keeps_its_own_budget():
    """The negative: two genuinely different sub-goals must not cross-block."""
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph = IdeaDag(root_title="root", root_details={"mandate": "Research"})
    dead_end = graph.add_child(graph.root_id(), "resolve the dead end")
    healthy = graph.add_child(graph.root_id(), "resolve dam Y")
    blocked = None
    for query in ("victoria dam height", "victoria dam cape town", "table mountain dam"):
        node = graph.add_child(
            dead_end.node_id, f"search: {query}",
            details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, "query": query},
        )
        blocked = engine._maybe_block_repeated_action(graph, node.node_id) or blocked
    assert blocked is not None and blocked["novelty_block_scope"] == "sub_goal"
    other = graph.add_child(
        healthy.node_id, "search dam Y",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value, "query": "dam Y height"},
    )
    assert engine._maybe_block_repeated_action(graph, other.node_id) is None


@pytest.mark.asyncio
async def test_a_stuck_search_loop_does_not_block_a_visit():
    """Only the target dimension is coarsened; the action type stays in the key."""
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph = IdeaDag(root_title="root", root_details={"mandate": "Research"})
    for query in ("a", "b", "c"):
        engine._maybe_block_repeated_action(graph, _flat_search_node(graph, query).node_id)
    visit = graph.add_child(
        graph.root_id(), "visit a page",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value, "url": "https://example.com/a"},
    )
    assert engine._maybe_block_repeated_action(graph, visit.node_id) is None


@pytest.mark.asyncio
async def test_a_productive_sub_goal_is_never_blocked():
    """The safety half: the coarse key only fires on a scope that is standing still."""
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph = IdeaDag(root_title="root", root_details={"mandate": "Research"})
    for i in range(6):
        node = _flat_search_node(graph, f"query {i}")
        assert engine._maybe_block_repeated_action(graph, node.node_id) is None
        graph.update_details(node.node_id, {
            DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
                action="search", content="results")})


@pytest.mark.asyncio
async def test_a_blocked_action_never_reaches_the_tool():
    """The whole point: the refusal happens BEFORE the action executes."""
    action = _CountingAction([ActionResultBuilder.failure(action="visit", error="404")])
    engine = _make_engine(action, run_policy_novelty_guard_enabled=True)
    graph, node_id = _graph_with_visit()
    for _ in range(2):
        await engine._execute_action_inner(graph, graph.root_id(), node_id)
    assert action.calls == 2
    result = await engine._execute_action_inner(graph, graph.root_id(), node_id)
    assert result["novelty_blocked"] is True
    assert action.calls == 2      # the third attempt cost nothing


@pytest.mark.asyncio
async def test_dispatch_is_unchanged_with_the_flag_off():
    action = _CountingAction([ActionResultBuilder.failure(action="visit", error="404")])
    engine = _make_engine(action)
    graph, node_id = _graph_with_visit()
    for _ in range(4):
        result = await engine._execute_action_inner(graph, graph.root_id(), node_id)
        assert "novelty_blocked" not in result
    assert action.calls == 4


# ---------------------------------------------------------------------------
# run-level telemetry (`final_payload["novelty_guard"]`)
# ---------------------------------------------------------------------------
async def _finalize(monkeypatch, engine, graph):
    import agent.app.idea_engine as engine_mod

    async def _fake_final_payload(*args, **kwargs):
        return {"final_deliverable": "answer", "goal_achieved": True, "has_failures": False}

    monkeypatch.setattr(engine_mod, "build_final_payload", _fake_final_payload)
    return await engine.finalize(graph, "Research", pending_check=False)


@pytest.mark.asyncio
async def test_the_payload_reports_what_the_guard_blocked(monkeypatch):
    """Without this a run cannot be asked, from its result alone, whether the guard ever fired."""
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph, node_id = _graph_with_visit()
    for _ in range(3):
        engine._maybe_block_repeated_action(graph, node_id)
    payload = await _finalize(monkeypatch, engine, graph)
    record = payload["novelty_guard"]
    assert record["blocked_actions"] == 1
    assert record["blocks"][0]["node_id"] == node_id
    assert record["blocks"][0]["novelty_key"] == ng.novelty_key(
        IdeaActionType.VISIT.value, graph.get_node(node_id).details
    )
    assert record["blocks"][0]["attempts"] == 2


@pytest.mark.asyncio
async def test_the_payload_reports_zero_when_nothing_was_blocked(monkeypatch):
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph, node_id = _graph_with_visit()
    engine._maybe_block_repeated_action(graph, node_id)
    payload = await _finalize(monkeypatch, engine, graph)
    assert payload["novelty_guard"]["blocked_actions"] == 0
    assert payload["novelty_guard"]["blocks"] == []


@pytest.mark.asyncio
async def test_the_payload_reports_the_near_miss_fan_out(monkeypatch):
    """The Finding A diagnostic: N distinct targets each stopping short of the threshold.

    Task 305 spends its budget on ONE dead end via two trap URLs and several query phrasings.
    Each is its own strict key, so none of them ever strikes; the count of such keys under ONE
    sub-goal scope is the measurement that says so.
    """
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph = IdeaDag(root_title="root", root_details={"mandate": "Research"})
    for i in range(4):
        node = graph.add_child(
            graph.root_id(), f"another phrasing {i}",
            details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                     "query": f"victoria dam height phrasing {i}"},
        )
        assert engine._maybe_block_repeated_action(graph, node.node_id) is None
        # Each attempt does return SOMETHING, so nothing strikes; the fan-out is still real and
        # this is exactly the run shape that reported "the guard never fired".
        graph.update_details(node.node_id, {
            DetailKey.ACTION_RESULT.value: ActionResultBuilder.success(
                action="search", content="results")})
    payload = await _finalize(monkeypatch, engine, graph)
    record = payload["novelty_guard"]
    assert record["near_miss_keys"] == 4
    assert record["near_miss_total_attempts"] == 4
    scope_key = f"{graph.root_id()}::{IdeaActionType.SEARCH.value}"
    assert record["near_miss_by_scope"][scope_key] == {"keys": 4, "attempts": 4}
    # ...and the other half of "why did nothing block": the sub-goal budget was reset by the
    # scope's own progress every time, so it never reached the threshold either.
    assert record["sub_goal_attempts_recorded"] == 4
    assert record["sub_goal_progress_resets"] == 3
    assert record["sub_goal_max_attempts"] == 1


@pytest.mark.asyncio
async def test_a_blocked_key_is_not_counted_as_a_near_miss(monkeypatch):
    engine = _make_engine(_CountingAction([]), run_policy_novelty_guard_enabled=True)
    graph, node_id = _graph_with_visit()
    for _ in range(3):
        engine._maybe_block_repeated_action(graph, node_id)
    payload = await _finalize(monkeypatch, engine, graph)
    assert payload["novelty_guard"]["blocked_actions"] == 1
    assert payload["novelty_guard"]["near_miss_keys"] == 0


@pytest.mark.asyncio
async def test_the_payload_carries_no_record_with_the_flag_off(monkeypatch):
    engine = _make_engine(_CountingAction([]))
    graph, _node_id = _graph_with_visit()
    payload = await _finalize(monkeypatch, engine, graph)
    assert "novelty_guard" not in payload
