"""Regression for the `good_adaptive` self-loop deadlock (dev-cycle HANDOFF item 1).

Root cause: when a leaf re-expands (`_maybe_reexpand_leaf` -> `_apply_reexpand` ->
`_handle_expansion_node`), the expanding node is passed as its own `parent_node_id`
into `LlmExpansionPolicy._parse_candidates`. If the model's follow-up candidate is a
bare "visit" action with no explicit URL, `_extract_url_from_path_context_with_source`
walks `graph.path_to_root(parent_node_id)` — which is INCLUSIVE of `parent_node_id`
itself — looking for a node with a search/visit `action_result` to pull a URL from.
On the exact task shape that triggered this live ("given no URLs, search then
visit"), the re-expanding node's OWN successful search result is the only URL
source on that path, so the resolved `source_node_id` is `parent_node_id` itself.

That self-reference then wires the new visit child's `requires_data.source_node_id`
back to its own parent. `IdeaDagEngine.step()`'s "wait for required data" gate
(~line 690-718) sees the source node's status is ACTIVE (re-expansion sets it ACTIVE,
not DONE, precisely because it now owns this pending child) and returns
`source_node_id` — which IS the current node — so the engine loop calls
`step()` on the exact same node again next turn, forever: a node cannot both be
DONE (required for it to count as a valid data source) and still own an
unexecuted child that is itself waiting on that same DONE-ness. No exception is
raised anywhere in this path, so the run silently burns its entire step budget.
"""
from __future__ import annotations

import logging

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies.action_constants import ActionResultKey
from agent.app.idea_policies.base import (
    DetailKey,
    DecompositionPolicy,
    EvaluationPolicy,
    ExpansionPolicy,
    IdeaActionType,
    IdeaNodeStatus,
)
from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.expansion import LlmExpansionPolicy
from agent.app.got_operations import GoTOperations


def _expansion_policy() -> LlmExpansionPolicy:
    policy = LlmExpansionPolicy.__new__(LlmExpansionPolicy)  # no LLM/connector needed for parsing
    policy._logger = logging.getLogger("reexpand-self-source-test")
    policy._cfg = IdeaConfig.from_settings({})
    return policy


def _graph_with_completed_search(url: str = "https://example.org/target-page"):
    """A root -> parent -> search_leaf graph where `search_leaf` already carries a
    successful search `action_result` (the state a leaf is in exactly when
    `_apply_reexpand` calls back into expansion with `node_id=search_leaf.node_id`)."""
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "investigate the target", details={})
    search_leaf = graph.add_child(
        parent.node_id,
        "search for the target page",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.ACTION_RESULT.value: {
                ActionResultKey.ACTION.value: IdeaActionType.SEARCH.value,
                ActionResultKey.SUCCESS.value: True,
                ActionResultKey.RESULTS.value: [
                    {"url": url, "title": "Target Page", "snippet": "the target page"},
                ],
            },
        },
    )
    search_leaf.status = IdeaNodeStatus.ACTIVE  # re-expansion sets this before expanding
    return graph, parent, search_leaf


def _fresh_graph_for_live_search(url: str = "https://example.org/target-page"):
    """Same shape, but `search_leaf` is PRISTINE (no result yet, status PENDING) so
    driving it through `engine.step()` exercises the real `_execute_action_guarded`
    -> `_apply_action_result` -> `_maybe_reexpand_leaf` completion path, exactly like
    a live run — rather than a hand-baked "already completed" state."""
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "investigate the target", details={})
    search_leaf = graph.add_child(
        parent.node_id,
        "search for the target page",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.IS_LEAF.value: True,
        },
    )
    return graph, parent, search_leaf, url


_VISIT_FOLLOWUP_CONTENT = (
    '{"candidates": ['
    '{"title": "Visit the target page", "action": "visit", '
    '"details": {"action": "visit", '
    '"justification": "the search revealed a page worth reading"}}'
    ']}'
)


def test_reexpansion_does_not_wire_visit_child_to_require_data_from_itself():
    """Root-cause unit test: the re-expanding node must never become its own
    child's `requires_data.source_node_id` — the URL it just resolved is already
    in hand, so there is nothing left to wait for."""
    policy = _expansion_policy()
    graph, parent, search_leaf = _graph_with_completed_search()

    candidates, _meta = policy._parse_candidates(
        _VISIT_FOLLOWUP_CONTENT, graph=graph, parent_node_id=search_leaf.node_id,
    )

    assert len(candidates) == 1
    details = candidates[0]["details"]
    assert details.get(DetailKey.URL.value) == "https://example.org/target-page", (
        "the URL should still be resolved from the search leaf's own results"
    )
    requires_data = details.get(DetailKey.REQUIRES_DATA.value)
    assert not (requires_data and requires_data.get("source_node_id") == search_leaf.node_id), (
        f"visit candidate must not require data from the node currently being "
        f"expanded (itself) — got requires_data={requires_data!r}"
    )


class DummyIO:
    def set_telemetry(self, telemetry):
        return None


class _RealExpansionForReexpand(ExpansionPolicy):
    """Wraps the real `LlmExpansionPolicy._parse_candidates` so the engine-level
    test exercises the actual URL/source-resolution logic (not a fake), without
    needing a real LLM connector — `expand()` just feeds fixed "LLM output" JSON
    straight to the real parser."""

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self._policy = _expansion_policy()
        self.calls = 0

    async def expand(self, graph: IdeaDag, node_id: str, memories=None):
        self.calls += 1
        candidates, _meta = self._policy._parse_candidates(
            _VISIT_FOLLOWUP_CONTENT, graph=graph, parent_node_id=node_id,
        )
        return candidates


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


class SearchReturningUrlAction(LeafAction):
    """A search action that returns a real, URL-bearing result — the shape
    `_extract_url_from_path_context_with_source` looks for."""

    def __init__(self, url: str, settings=None):
        super().__init__(settings=settings)
        self._url = url

    async def execute(self, graph: IdeaDag, node_id: str, io):
        return {
            ActionResultKey.ACTION.value: IdeaActionType.SEARCH.value,
            ActionResultKey.SUCCESS.value: True,
            ActionResultKey.RESULTS.value: [
                {"url": self._url, "title": "Target Page", "snippet": "the target page"},
            ],
        }


class VisitSucceedsAction(LeafAction):
    async def execute(self, graph: IdeaDag, node_id: str, io):
        return {
            ActionResultKey.ACTION.value: IdeaActionType.VISIT.value,
            ActionResultKey.SUCCESS.value: True,
            ActionResultKey.CONTENT.value: "the target page's content",
        }


class ActionAwareRegistry:
    def __init__(self, settings, url: str):
        self.settings = dict(settings or {})
        self._url = url

    def get(self, action_type) -> LeafAction:
        value = action_type.value if hasattr(action_type, "value") else action_type
        if value == IdeaActionType.SEARCH.value:
            return SearchReturningUrlAction(self._url, settings=self.settings)
        return VisitSucceedsAction(settings=self.settings)


class _OneShotVerdict:
    """Positive follow-up verdict on the first call only (mirrors reexpand_leaf_test.py)."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, graph, node_id, model_name=None):
        self.calls += 1
        if self.calls == 1:
            return {"needs_followup": True, "reason": "revealed a page worth visiting"}
        return {"needs_followup": False, "reason": "no further follow-up"}


def _make_engine(url: str):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 1,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
    }
    expansion = _RealExpansionForReexpand(settings)
    engine = IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=expansion,
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=ActionAwareRegistry(settings, url),
    )
    ops = GoTOperations(settings=engine.settings, io=engine.io, memory_manager=None)
    ops.check_needs_followup = _OneShotVerdict()  # type: ignore[assignment]
    engine._got = ops
    return engine, expansion


async def _drive(engine, graph, start_id, max_steps=40):
    """Mirrors reexpand_leaf_test.py's `_drive`: `run()`'s current_id = step(...) loop."""
    current = start_id
    for step in range(max_steps):
        current = await engine.step(graph, current, step)
        if current is None:
            return step + 1, None
    return max_steps, current


@pytest.mark.asyncio
async def test_reexpanded_search_leaf_does_not_deadlock_on_its_own_visit_followup():
    """End-to-end regression for the reported live symptom: driving the engine's
    step loop from a pristine search leaf — through its real execution, real
    re-expansion, and the real URL/source resolution — whose genuine follow-up is
    a same-page "visit" must NOT spin on the search leaf forever. Before the fix,
    every step returns the search leaf's own node_id (waiting on itself), so
    `_drive` never terminates within the budget and the visit child never runs.
    This is the exact "given no URLs, search then visit" shape that triggered the
    bug live: no URL exists anywhere until the search leaf's own result produces
    one."""
    graph, parent, search_leaf, url = _fresh_graph_for_live_search()
    engine, expansion = _make_engine(url)

    steps_taken, final = await _drive(engine, graph, search_leaf.node_id, max_steps=40)

    assert expansion.calls == 1, "the leaf should re-expand exactly once"
    assert len(search_leaf.children) == 1, "re-expansion should have spawned the visit follow-up"
    visit_child = graph.get_node(search_leaf.children[0])
    assert visit_child is not None

    assert steps_taken < 40, (
        f"drive loop should terminate well under the step budget, ran the full "
        f"{steps_taken} steps without progress (self-loop deadlock)"
    )
    assert visit_child.status == IdeaNodeStatus.DONE, (
        f"the visit follow-up must actually execute and complete, got "
        f"{visit_child.status} — a deadlocked run leaves it perpetually unready"
    )
    assert search_leaf.status == IdeaNodeStatus.DONE, (
        f"the re-expanded search leaf should complete once its child is done, "
        f"got {search_leaf.status}"
    )
