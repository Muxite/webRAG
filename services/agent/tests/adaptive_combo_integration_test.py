"""C1e combined-integration test: the FULL "good adaptive agent" opt-in flag set,
enabled TOGETHER, driven through the real `IdeaDagEngine.run()` loop (not just the
per-mechanism unit tests).

Flags on: `got_reexpand_enabled`, `got_step_confidence_judge_enabled`,
`got_step_confidence_reexpand_enabled`, `got_reexpand_corrective_context_enabled`,
`tool_failure_recovery_enabled`, `connector_retry_on_failure_enabled`,
`native_vote_k_enabled` (k=2), `got_backtrack_enabled`.

Fully offline: expansion/action/judge/follow-up-detector are deterministic fakes;
the only "LLM" call is the finalize call, stubbed via a fake `AgentIO`.

Asserts:
  * the run completes without raising and within the step/node budget (bounded);
  * a low-confidence, genuinely-successful leaf ("good_leaf") is re-expanded, and
    the corrective-context reason threading fired for it (composes A1 + C1c);
  * a low-confidence leaf whose low score is a TOOL failure ("bad_leaf", an empty
    search) is NOT re-expanded by either trigger — the bounded in-place connector
    retry is what recovers it instead (composes C1a's suppression across both the
    confidence-trigger and follow-up-detector re-expansion paths);
  * `got_backtrack_enabled` and `native_vote_k_enabled` (k=2) being on at the same
    time does not crash or otherwise break finalize.
"""
from __future__ import annotations

from unittest.mock import MagicMock

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
)
from agent.app.idea_policies.actions import LeafAction


class ScriptedExpansion(ExpansionPolicy):
    """Root fans out into `good_leaf` + `bad_leaf`; any other node gets a single
    deterministic follow-up child (used by re-expansion)."""

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.calls: list = []

    async def expand(self, graph: IdeaDag, node_id: str, memories=None):
        node = graph.get_node(node_id)
        self.calls.append(node.title if node else None)
        if node_id == graph.root_id():
            return [
                {
                    "title": "good_leaf: resolve the primary fact",
                    "details": {
                        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                        DetailKey.JUSTIFICATION.value: "primary fact",
                    },
                    "score": None,
                },
                {
                    "title": "bad_leaf: resolve a secondary fact",
                    "details": {
                        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                        DetailKey.JUSTIFICATION.value: "secondary fact",
                    },
                    "score": None,
                },
            ]
        return [
            {
                "title": f"Follow-up for {node.title[:40] if node else ''}",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.THINK.value,
                    DetailKey.JUSTIFICATION.value: "course-correct",
                },
                "score": None,
            }
        ]


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        for nid in candidate_ids:
            graph.evaluate(nid, 0.6)
        return {nid: 0.6 for nid in candidate_ids}


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class ScriptedAction(LeafAction):
    """A single shared action instance routes on the node's title/action so the one
    `FakeRegistry` below can serve every action type deterministically."""

    def __init__(self, settings=None):
        super().__init__(settings=settings)
        self.call_counts: dict = {}

    async def execute(self, graph, node_id, io):
        node = graph.get_node(node_id)
        title = node.title if node else ""
        action = node.details.get(DetailKey.ACTION.value) if node else None
        self.call_counts[node_id] = self.call_counts.get(node_id, 0) + 1

        if action == IdeaActionType.MERGE.value:
            return {"action": IdeaActionType.MERGE.value, "success": True}
        if "bad_leaf" in title:
            # Always an empty search — a persistent TOOL failure that survives
            # every in-place connector retry (a real transient failure would
            # eventually recover; this one models a source that just has nothing).
            return {"action": IdeaActionType.SEARCH.value, "success": True, "results": []}
        if "good_leaf" in title:
            return {
                "action": IdeaActionType.SEARCH.value,
                "success": True,
                "results": [{"url": "https://example.test/a", "title": "hit"}],
                "content": "some genuinely resolved content",
            }
        return {"action": IdeaActionType.THINK.value, "success": True}


class FakeRegistry:
    def __init__(self, action):
        self._action = action

    def get(self, action_type):
        return self._action


class FakeAgentIO:
    """Minimal IO stub: never touched except by `finalize()`'s LLM call(s)."""

    connector_chroma = None
    telemetry = None

    def build_llm_payload(self, **kwargs):
        return {"model": kwargs.get("model_name"), "messages": kwargs.get("messages")}

    async def query_llm_with_fallback(self, payload, **kwargs):
        return '{"deliverable": "the answer", "summary": "resolved via adaptive combo"}'


def _judge_step_confidence(title: str):
    if "good_leaf" in title:
        # Genuinely resolved content the judge nonetheless distrusts -> should re-expand.
        return {"confidence": 0.2, "reason": "insufficient depth for the primary fact"}
    if "bad_leaf" in title:
        # Low confidence CAUSED BY a tool failure -> must NOT re-expand when
        # `tool_failure_recovery_enabled` is on; retry is the correct recovery.
        return {"confidence": 0.1, "reason": "no usable search results"}
    return {"confidence": 0.95, "reason": "fine"}


@pytest.mark.asyncio
async def test_full_adaptive_combo_composes_and_terminates(monkeypatch):
    followup_calls: list = []

    async def _judge(self, graph, node_id, model_name=None):
        node = graph.get_node(node_id)
        return _judge_step_confidence(node.title if node else "")

    async def _followup(self, graph, node_id, model_name=None):
        node = graph.get_node(node_id)
        followup_calls.append(node.title if node else None)
        return {"needs_followup": True, "reason": "should never fire in this scenario"}

    monkeypatch.setattr(GoTOperations, "judge_step_confidence", _judge)
    monkeypatch.setattr(GoTOperations, "check_needs_followup", _followup)

    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "max_total_nodes": 60,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        # --- the full "good adaptive agent" opt-in flag combination ---
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 1,
        "got_step_confidence_judge_enabled": True,
        "got_step_confidence_reexpand_enabled": True,
        "got_step_confidence_reexpand_threshold": 0.5,
        "got_reexpand_corrective_context_enabled": True,
        "tool_failure_recovery_enabled": True,
        "connector_retry_on_failure_enabled": True,
        "connector_retry_max_attempts": 2,
        "connector_retry_backoff_seconds": 0.0,
        "native_vote_k_enabled": True,
        "native_vote_k": 2,
        "got_backtrack_enabled": True,
        "got_backtrack_dead_end_threshold": 2,
        "got_backtrack_low_score_threshold": 0.3,
    }

    action = ScriptedAction(settings)
    expansion = ScriptedExpansion(settings)
    engine = IdeaDagEngine(
        io=FakeAgentIO(),
        settings=settings,
        expansion=expansion,
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeRegistry(action),
    )

    payload = await engine.run("adaptive combo mandate", max_steps=20)

    # --- runs to completion, bounded ---
    assert isinstance(payload, dict)
    assert payload.get("final_deliverable") == "the answer"
    graph_dict = payload["graph"]
    assert len(graph_dict["nodes"]) < 60, "must terminate well inside the node ceiling"

    graph = IdeaDag.from_dict(graph_dict)
    nodes_by_title = {n.title: n for n in graph.iter_depth_first()}
    good_leaf = next(n for t, n in nodes_by_title.items() if "good_leaf" in t)
    bad_leaf = next(n for t, n in nodes_by_title.items() if "bad_leaf" in t)

    # --- a genuinely low-confidence (but tool-healthy) leaf re-expands, with the
    #     corrective-context reason threaded through `_apply_reexpand` ---
    assert good_leaf.details.get("_got_reexpanded") is True
    assert good_leaf.children, "good_leaf must have spawned a follow-up child"
    assert "low step confidence" in str(good_leaf.details.get("_got_reexpand_reason", ""))

    # --- a tool-failure leaf does NOT re-expand under either trigger, despite also
    #     scoring low confidence and despite the follow-up detector being enabled ---
    assert "_got_reexpanded" not in bad_leaf.details
    assert not bad_leaf.children
    assert action.call_counts.get(bad_leaf.node_id, 0) >= 2, (
        "bad_leaf must have been retried in-place by the connector-retry mechanism"
    )
    assert followup_calls == [], (
        "the follow-up detector must never be consulted for either leaf: good_leaf's "
        "children already gate `_reexpand_check` out, and bad_leaf is suppressed by "
        "the tool-failure gate"
    )

    # --- backtrack + k-vote were enabled throughout and did not break anything ---
    assert "got_stats" in payload
