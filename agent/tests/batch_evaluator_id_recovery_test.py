"""N3b: the batch evaluator's simple-id contract, and what happens when the model breaks it.

``LlmBatchEvaluationPolicy`` presents candidates to the judge under simple ids (1, 2, 3...) and
resolves the reply through ``candidate_id_map``. That indirection is the design, not a bug —
but two things could silently discard scores:

* a candidate missing from the graph consumed its ``idx`` anyway, so the presented list could
  read 1, 3, 4; a model that helpfully renumbers to 1, 2, 3 then misroutes every score;
* a reply using ids outside the map fell through to ``graph.get_node(<raw id>)``, which logs
  ``Skipping unknown node_id`` and drops the score (51 occurrences across 23 of 59 cells).

Positional recovery covers the second when the reply is complete (one score per presented
candidate); anything else must still warn and skip rather than mis-assign.
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey
from agent.app.idea_policies.evaluation import LlmBatchEvaluationPolicy


class _ScriptedIO:
    """Fake AgentIO returning one canned batch-evaluation response."""

    def __init__(self, response: str):
        self.response = response
        self.last_messages = None
        self.telemetry = None

    def set_telemetry(self, telemetry):
        return None

    def build_llm_payload(self, messages=None, **kwargs):
        self.last_messages = messages
        return {"messages": messages}

    async def query_llm_with_fallback(self, payload, **kwargs):
        return self.response


def _policy(response: str, **overrides):
    settings = {"evaluation_weights": {}, "evaluation_batch_max_candidates": 5}
    settings.update(overrides)
    return LlmBatchEvaluationPolicy(io=_ScriptedIO(response), settings=settings)


def _graph(n: int = 3):
    """Root plus ``n`` unexecuted action children, returned as (graph, parent_id, child_ids)."""
    graph = IdeaDag(root_title="mandate", root_details={"mandate": "mandate"})
    root = graph.root_id()
    children = [
        graph.add_child(root, f"cand {i}", details={DetailKey.IS_LEAF.value: True}).node_id
        for i in range(n)
    ]
    return graph, root, children


def _presented(io):
    """The candidate objects the judge was shown, in prompt order."""
    return json.loads(io.last_messages[1]["content"])["candidates"]


def _scores_reply(pairs):
    return json.dumps({"scores": [{"id": i, "score": s} for i, s in pairs]})


@pytest.mark.asyncio
async def test_missing_candidate_does_not_leave_a_gap_in_the_presented_ids():
    graph, root, children = _graph(3)
    # The middle candidate is gone from the graph by the time the batch is built (dedup,
    # pruning, a merge that absorbed it) -- the id survives in the caller's list.
    candidate_ids = [children[0], "vanished-node-id", children[2]]
    policy = _policy(_scores_reply([("1", 0.8), ("2", 0.2)]))

    scores = await policy.evaluate_batch(graph, root, candidate_ids)

    assert [c["id"] for c in _presented(policy.io)] == ["1", "2"]
    assert [c["title"] for c in _presented(policy.io)] == ["cand 0", "cand 2"]
    assert scores == {children[0]: pytest.approx(0.8), children[2]: pytest.approx(0.2)}
    assert graph.get_node(children[2]).score == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_renumbered_reply_is_recovered_positionally(caplog):
    graph, root, children = _graph(3)
    policy = _policy(_scores_reply([("A", 0.9), ("B", 0.5), ("C", 0.1)]))

    with caplog.at_level("WARNING"):
        scores = await policy.evaluate_batch(graph, root, children)

    assert scores == {
        children[0]: pytest.approx(0.9),
        children[1]: pytest.approx(0.5),
        children[2]: pytest.approx(0.1),
    }
    assert "A" in caplog.text, "the raw id must stay visible when recovery fires"


@pytest.mark.asyncio
async def test_incomplete_unknown_reply_is_skipped_not_misassigned(caplog):
    graph, root, children = _graph(3)
    policy = _policy(_scores_reply([("9f3", 0.9)]))

    with caplog.at_level("WARNING"):
        scores = await policy.evaluate_batch(graph, root, children)

    assert all(graph.get_node(cid).score is None for cid in children)
    assert "9f3" in scores, "an unresolvable id is returned as-is, not attached to a node"
    assert "Skipping unknown node_id" in caplog.text


@pytest.mark.asyncio
async def test_wellformed_reply_needs_no_recovery(caplog):
    graph, root, children = _graph(3)
    policy = _policy(_scores_reply([("1", 0.7), ("2", 0.6), ("3", 0.5)]))

    with caplog.at_level("WARNING"):
        scores = await policy.evaluate_batch(graph, root, children)

    assert [graph.get_node(cid).score for cid in children] == [
        pytest.approx(0.7),
        pytest.approx(0.6),
        pytest.approx(0.5),
    ]
    assert scores
    assert "positional" not in caplog.text.lower()
