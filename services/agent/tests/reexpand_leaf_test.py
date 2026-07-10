"""Tests for bounded leaf re-expansion (`got.reexpand_enabled`).

When a leaf completes with a successful result and the follow-up check says a
genuine follow-up exists, the engine re-expands that leaf into new children via
the real expansion policy. Gated by `got_reexpand_enabled` (default OFF) and
bounded by `got_reexpand_max_iterations`.

The most important test is the regression one: with the flag OFF, a leaf that
*would* trigger a follow-up under the flag stays a normal terminal leaf and no
follow-up check LLM call is ever issued.
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
    """Return a single deterministic follow-up child on every expand call."""

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.calls = 0

    async def expand(self, graph: IdeaDag, node_id: str, memories=None):
        self.calls += 1
        return [
            {
                "title": "Follow-up: investigate the newly revealed target",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.THINK.value,
                    DetailKey.JUSTIFICATION.value: "revealed by parent result",
                },
                "score": None,
            }
        ]


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph: IdeaDag, node_id: str) -> float:
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph: IdeaDag, parent_id: str, candidate_ids):
        scores = {}
        for node_id in candidate_ids:
            graph.evaluate(node_id, 0.6)
            scores[node_id] = 0.6
        return scores


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph: IdeaDag, node_id: str) -> bool:
        return False


class AlwaysSuccessAction(LeafAction):
    async def execute(self, graph: IdeaDag, node_id: str, io):
        return {"action": IdeaActionType.THINK.value, "success": True}


class FakeRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type: IdeaActionType) -> LeafAction:
        return AlwaysSuccessAction(settings=self.settings)


def _make_engine(reexpand_enabled: bool, max_iters: int = 1, extra=None):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_reexpand_enabled": reexpand_enabled,
        "got_reexpand_max_iterations": max_iters,
        # Keep the noisy GoT policies quiet in tests.
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


class _Verdict:
    """Stub follow-up check that records whether it was called."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    async def __call__(self, graph, node_id, model_name=None):
        self.calls += 1
        return self.verdict


def _attach_got(engine, verdict):
    """Attach a real GoTOperations with a stubbed follow-up check."""
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    checker = _Verdict(verdict)
    ops.check_needs_followup = checker  # type: ignore[assignment]
    engine._got = ops
    return checker


def _completed_leaf(graph: IdeaDag):
    """A parent with one ready, executable leaf child (no children of its own).

    The leaf has no result yet: `_handle_leaf_node` executes it fresh via the
    action registry (which returns a success), driving it through the shared
    `_apply_action_result` completion point — the real path on which the
    re-expansion check now fires (rather than pre-baking a DONE result, which
    would bypass execution and never reach the completion point)."""
    root = graph.root_id()
    parent = graph.add_child(root, "parent goal", details={})
    leaf = graph.add_child(
        parent.node_id,
        "Disambiguate the 4 River Avon candidates",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.IS_LEAF.value: True,
        },
    )
    return parent, leaf


@pytest.mark.asyncio
async def test_flag_off_no_reexpansion_and_no_llm_check():
    """Regression: flag OFF -> a leaf that would trigger a follow-up stays terminal."""
    engine, expansion = _make_engine(reexpand_enabled=False)
    checker = _attach_got(engine, {"needs_followup": True, "reason": "should never be read"})
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert result == parent.node_id, "completed leaf should return to its parent"
    assert leaf.children == [], "no children should be created when flag is off"
    assert checker.calls == 0, "follow-up check must not be invoked when flag is off"
    assert expansion.calls == 0, "expansion policy must not run when flag is off"
    assert "_got_reexpanded" not in leaf.details
    assert leaf.status == IdeaNodeStatus.DONE


@pytest.mark.asyncio
async def test_flag_on_needs_followup_creates_children():
    """Flag ON + needs_followup=True + under cap -> new children spawned from the leaf."""
    engine, expansion = _make_engine(reexpand_enabled=True, max_iters=1)
    checker = _attach_got(engine, {"needs_followup": True, "reason": "survivor points to River Stour"})
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert checker.calls == 1
    assert expansion.calls == 1
    assert len(leaf.children) == 1, "re-expansion should create a follow-up child"
    assert result == leaf.node_id, "engine should stay on the re-expanded node"
    assert leaf.details.get("_got_reexpand_count") == 1
    assert leaf.details.get("_got_reexpanded") is True
    assert leaf.status == IdeaNodeStatus.ACTIVE


@pytest.mark.asyncio
async def test_flag_on_cap_reached_falls_through_to_merge():
    """Flag ON but iteration cap already reached -> no further re-expansion."""
    engine, expansion = _make_engine(reexpand_enabled=True, max_iters=1)
    checker = _attach_got(engine, {"needs_followup": True, "reason": "another follow-up"})
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)
    # Simulate the node having already been re-expanded once.
    leaf.details["_got_reexpand_count"] = 1

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert result == parent.node_id, "cap reached -> return to parent (normal merge path)"
    assert leaf.children == [], "no new children past the iteration cap"
    assert checker.calls == 0, "check should be skipped once the cap is reached"
    assert expansion.calls == 0


@pytest.mark.asyncio
async def test_flag_on_no_followup_normal_merge():
    """Flag ON + needs_followup=False -> normal merge, no re-expansion attempted."""
    engine, expansion = _make_engine(reexpand_enabled=True, max_iters=1)
    checker = _attach_got(engine, {"needs_followup": False, "reason": "nothing new revealed"})
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert result == parent.node_id
    assert leaf.children == [], "no children when follow-up check says no"
    assert checker.calls == 1, "the check runs, but its negative verdict is honored"
    assert expansion.calls == 0
    assert "_got_reexpanded" not in leaf.details
    assert leaf.status == IdeaNodeStatus.DONE


def _parent_with_sibling_leaves(graph: IdeaDag, n: int = 4):
    """A parent with `n` unexecuted, ready action-leaf siblings (auto-parallel shape).

    Mirrors test 095 Stage 1: several candidate leaves created together, none with
    children of their own, all executable via the batch/gather path.
    """
    root = graph.root_id()
    parent = graph.add_child(root, "parent goal", details={})
    leaves = []
    for i in range(n):
        leaf = graph.add_child(
            parent.node_id,
            f"Investigate River Avon candidate {i}",
            details={
                DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                DetailKey.IS_LEAF.value: True,
            },
        )
        leaves.append(leaf)
    return parent, leaves


@pytest.mark.asyncio
async def test_auto_parallel_path_reexpands_completed_siblings():
    """AUTO-PARALLEL path: sibling leaves completing via the batch gather each get
    the same re-expansion check (this is the path that previously bypassed it)."""
    engine, expansion = _make_engine(
        reexpand_enabled=True,
        max_iters=1,
        extra={"auto_parallel_siblings": True, "allow_execute_all_children": True},
    )
    checker = _attach_got(engine, {"needs_followup": True, "reason": "survivor revealed"})
    graph = IdeaDag(root_title="root")
    parent, leaves = _parent_with_sibling_leaves(graph, n=4)

    # Drives the auto-parallel block: >1 ready executable leaf siblings.
    result = await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    assert result == parent.node_id
    # Every sibling completed via the batch path and was offered the check.
    assert checker.calls == 4, "every completed sibling must get the follow-up check"
    assert expansion.calls == 4, "each qualifying sibling re-expands via the real policy"
    for leaf in leaves:
        node = graph.get_node(leaf.node_id)
        assert len(node.children) == 1, "each sibling should spawn a follow-up child"
        assert node.details.get("_got_reexpand_count") == 1
        assert node.details.get("_got_reexpanded") is True
        assert node.status == IdeaNodeStatus.ACTIVE


@pytest.mark.asyncio
async def test_auto_parallel_path_flag_off_is_unchanged():
    """Regression: flag OFF -> the auto-parallel batch path issues no follow-up check
    and no expansion; completed siblings stay terminal DONE leaves."""
    engine, expansion = _make_engine(
        reexpand_enabled=False,
        extra={"auto_parallel_siblings": True, "allow_execute_all_children": True},
    )
    checker = _attach_got(engine, {"needs_followup": True, "reason": "never read"})
    graph = IdeaDag(root_title="root")
    parent, leaves = _parent_with_sibling_leaves(graph, n=4)

    result = await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    assert result == parent.node_id
    assert checker.calls == 0, "flag off -> no follow-up check on the auto-parallel path"
    assert expansion.calls == 0, "flag off -> no expansion on the auto-parallel path"
    for leaf in leaves:
        node = graph.get_node(leaf.node_id)
        assert node.children == [], "no children when flag is off"
        assert "_got_reexpanded" not in node.details
        assert node.status == IdeaNodeStatus.DONE


@pytest.mark.asyncio
async def test_sequential_branch_reexpands_selected_child():
    """SEQUENTIAL branch: with auto-parallel OFF, `_handle_intermediate_node`
    evaluates the eligible siblings, selects the single best child, and executes
    only that one via the sequential-execution branch (idea_engine ~1031-1057).

    This branch was NEVER patched per-hand for re-expansion — it only fires the
    check because every path now routes its executed action through the shared
    `_apply_action_result` completion point. This test proves centralization
    generalizes to the branch that test 095 actually took, with no branch-specific
    wiring."""
    engine, expansion = _make_engine(
        reexpand_enabled=True,
        max_iters=1,
        # auto_parallel OFF (default in _make_engine) -> single best child is
        # executed sequentially instead of the whole batch.
        extra={"auto_parallel_siblings": False},
    )
    checker = _attach_got(engine, {"needs_followup": True, "reason": "survivor -> River Stour, Dorset"})
    graph = IdeaDag(root_title="root")
    parent, leaves = _parent_with_sibling_leaves(graph, n=4)

    result = await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    # Exactly ONE child was executed (the selected best) and it got the check.
    assert checker.calls == 1, "the sequentially-executed child must get the follow-up check"
    assert expansion.calls == 1, "only the selected child re-expands via the real policy"

    reexpanded = [graph.get_node(l.node_id) for l in leaves
                  if graph.get_node(l.node_id).children]
    assert len(reexpanded) == 1, "exactly one (the selected) child should re-expand"
    node = reexpanded[0]
    assert len(node.children) == 1
    assert node.details.get("_got_reexpand_count") == 1
    assert node.details.get("_got_reexpanded") is True
    assert node.status == IdeaNodeStatus.ACTIVE
    # The sequential branch returns to the parent; the re-expanded selected child
    # is left ACTIVE with children and step() descends into it on the next step.
    assert result == parent.node_id


class _OverlapVerdict:
    """Follow-up check that records concurrency by tracking in-flight overlap.

    Each call increments an in-flight counter, yields the event loop once (so
    concurrent callers can interleave), records the max simultaneous in-flight
    count, then returns the verdict. If the batch path runs the checks
    concurrently, ``max_in_flight`` will exceed 1; if it runs them one-await-at-a-time
    it stays pinned at 1.
    """

    def __init__(self, verdict):
        self.verdict = verdict
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    async def __call__(self, graph, node_id, model_name=None):
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        # Two loop turns so siblings started by gather genuinely overlap.
        import asyncio as _asyncio
        await _asyncio.sleep(0)
        await _asyncio.sleep(0)
        self.in_flight -= 1
        return self.verdict


@pytest.mark.asyncio
async def test_auto_parallel_reexpand_checks_run_concurrently():
    """The per-sibling follow-up checks must run concurrently (not serially).

    Uses a checker that records the maximum number of simultaneously in-flight
    calls; with the gather-based batch path, all N siblings' checks overlap.
    """
    engine, expansion = _make_engine(
        reexpand_enabled=True,
        max_iters=1,
        extra={"auto_parallel_siblings": True, "allow_execute_all_children": True},
    )
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    checker = _OverlapVerdict({"needs_followup": True, "reason": "survivor revealed"})
    ops.check_needs_followup = checker  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaves = _parent_with_sibling_leaves(graph, n=4)

    result = await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    assert result == parent.node_id
    assert checker.calls == 4
    assert checker.max_in_flight == 4, (
        f"all 4 sibling checks should overlap concurrently, "
        f"got max_in_flight={checker.max_in_flight}"
    )


@pytest.mark.asyncio
async def test_auto_parallel_reexpand_respects_max_total_nodes_under_concurrency():
    """Concurrent siblings must not overshoot the max_total_nodes ceiling.

    All siblings pass the read-only gate together (checked before any expands), but
    the sequential apply phase re-reads the ceiling, so only as many expand as the
    node budget allows. With a tight ceiling, fewer than N siblings re-expand.
    """
    graph = IdeaDag(root_title="root")
    parent, leaves = _parent_with_sibling_leaves(graph, n=4)
    # Budget: root + parent + 4 leaves == 6 nodes already present. Allow exactly 2
    # more nodes (each re-expansion adds one child), so only 2 of the 4 may expand.
    ceiling = graph.node_count() + 2
    engine, expansion = _make_engine(
        reexpand_enabled=True,
        max_iters=1,
        extra={
            "auto_parallel_siblings": True,
            "allow_execute_all_children": True,
            "max_total_nodes": ceiling,
        },
    )
    checker = _attach_got(engine, {"needs_followup": True, "reason": "survivor revealed"})

    result = await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    assert result == parent.node_id
    # Every DONE sibling got the read-only check...
    assert checker.calls == 4
    # ...but the ceiling caps how many actually grew children.
    reexpanded = [graph.get_node(l.node_id) for l in leaves if graph.get_node(l.node_id).children]
    assert len(reexpanded) == 2, (
        f"ceiling of +2 nodes should allow exactly 2 re-expansions, got {len(reexpanded)}"
    )
    assert graph.node_count() <= ceiling, "node count must never exceed max_total_nodes"


@pytest.mark.asyncio
async def test_step_escape_hatch_drives_reexpanded_children():
    """After re-expansion, step() routes the leaf through its children, not the leaf handler."""
    engine, expansion = _make_engine(reexpand_enabled=True, max_iters=1)
    _attach_got(engine, {"needs_followup": True, "reason": "survivor points to River Stour"})
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    # Trigger the re-expansion.
    await engine._handle_leaf_node(graph, leaf.node_id, 0, None)
    assert len(leaf.children) == 1
    child_id = leaf.children[0]

    # A subsequent step() on the re-expanded leaf must process the child (evaluate +
    # execute) rather than short-circuiting straight back to the parent.
    await engine.step(graph, leaf.node_id, 1)
    child = graph.get_node(child_id)
    assert child is not None
    assert child.status in (
        IdeaNodeStatus.DONE,
        IdeaNodeStatus.ACTIVE,
    ), f"re-expanded child should have been driven, got {child.status}"


class _OneShotVerdict:
    """Positive follow-up verdict on the first call only, negative thereafter.

    Mirrors a real run: a leaf reveals exactly one genuine follow-up; the spawned
    child does not itself reveal another. This gives a finite subtree so the
    end-to-end traversal terminates cleanly instead of cascading forever.
    """

    def __init__(self):
        self.calls = 0

    async def __call__(self, graph, node_id, model_name=None):
        self.calls += 1
        if self.calls == 1:
            return {"needs_followup": True, "reason": "survivor -> River Stour, Dorset"}
        return {"needs_followup": False, "reason": "no further follow-up"}


async def _drive(engine, graph, start_id, max_steps=40):
    """Drive the engine like `run()` does: current_id = await step(...) each turn.

    Stops when a step returns None or the budget is exhausted, exactly like the
    real loop. Returns the number of steps actually taken.
    """
    current = start_id
    for step in range(max_steps):
        current = await engine.step(graph, current, step)
        if current is None:
            return step + 1
    return max_steps


@pytest.mark.asyncio
async def test_reexpanded_child_is_driven_to_done_end_to_end():
    """The engine's step loop must actually traverse INTO a re-expanded node's new
    child, execute it, and land it DONE — not spin on the parent treating the
    re-expanded node as a terminal leaf (the live-run regression).

    Before the fix, once a leaf re-expanded (ACTIVE, with an action_result AND a
    fresh child), the parent's children-processing re-executed it as a leaf,
    clobbering its status back to DONE and orphaning the new child, which never
    executed. Here we drive the full step loop starting from the parent and prove
    the grandchild reaches DONE with retrievable content."""
    engine, expansion = _make_engine(reexpand_enabled=True, max_iters=1)
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    ops.check_needs_followup = _OneShotVerdict()  # type: ignore[assignment]
    engine._got = ops
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    await _drive(engine, graph, parent.node_id)

    # The leaf really re-expanded and the engine descended into the new subtree.
    assert leaf.details.get("_got_reexpanded") is True
    assert len(leaf.children) == 1, "re-expansion should have spawned exactly one child"
    child = graph.get_node(leaf.children[0])
    assert child is not None

    # The core assertion: the NEW child was actually executed and driven to DONE
    # (not left orphaned as pending/active), and its action result is retrievable
    # from the graph — proving its work reaches the final state.
    assert child.status == IdeaNodeStatus.DONE, (
        f"re-expanded child must be driven to DONE, got {child.status}"
    )
    child_result = child.details.get(DetailKey.ACTION_RESULT.value)
    assert isinstance(child_result, dict) and child_result.get("success") is True, (
        f"re-expanded child's action_result must surface, got {child_result}"
    )
    # And the re-expanded node itself completes (no infinite spin / no stale ACTIVE).
    assert leaf.status == IdeaNodeStatus.DONE, (
        f"re-expanded node should complete once its child is done, got {leaf.status}"
    )
    # The child is a proper descendant reachable from the re-expanded leaf, so a
    # leaf-collecting finalize sees it (rather than only the pre-reexpansion leaf).
    assert child.parent_id == leaf.node_id
