"""`LlmBatchEvaluationPolicy.evaluate_batch` must score every candidate it is handed.

``evaluation_batch_max_candidates`` bounds how many candidates fit in ONE scoring prompt.
It used to be applied as a truncation (``candidate_ids[:max_candidates]``), so a parent with
more children than the cap left the overflow with ``score is None`` while those children
stayed selectable (``allow_unscored_selection`` defaults true) and stayed eligible for
execution: a decision consuming a score that was never computed. The cap is now a CHUNK size.

Offline: a fake IO that answers each scoring prompt from the candidate ids it was actually
shown, so a dropped candidate shows up as a missing score rather than a wrong one.
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.evaluation import _MAX_BATCH_CHUNKS, LlmBatchEvaluationPolicy


class FakeIO:
    """Scores every candidate the prompt actually contains, 1.0 / 0.9 / 0.8 / ... by position."""

    telemetry = None

    def __init__(self):
        self.calls = []  # one entry per LLM call: the list of simple ids in that prompt

    def build_llm_payload(self, **kwargs):
        return {"messages": kwargs.get("messages")}

    async def query_llm_with_fallback(self, payload, **kwargs):
        user = payload["messages"][-1]["content"]
        candidates = json.loads(user)["candidates"]
        ids = [c["id"] for c in candidates]
        self.calls.append(ids)
        return json.dumps(
            {"scores": [{"id": cid, "score": 1.0 - 0.1 * i} for i, cid in enumerate(ids)]}
        )


def _graph_with_children(n: int) -> tuple[IdeaDag, str, list[str]]:
    graph = IdeaDag(root_title="mandate", root_details={"mandate": "mandate"})
    root = graph.root_id()
    kids = graph.expand(
        root,
        [
            {
                "title": f"leaf {i}",
                # A result is present so the "action but no result" penalty cap (0.5) does
                # not flatten the scores this test reads back.
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                    DetailKey.ACTION_RESULT.value: {"success": True},
                },
            }
            for i in range(n)
        ],
    )
    return graph, root, [k.node_id for k in kids]


def _policy(io, cap=5):
    return LlmBatchEvaluationPolicy(io, settings={"evaluation_batch_max_candidates": cap})


@pytest.mark.asyncio
async def test_within_cap_still_issues_exactly_one_call():
    io = FakeIO()
    graph, root, kids = _graph_with_children(5)

    scores = await _policy(io).evaluate_batch(graph, root, kids)

    assert len(io.calls) == 1
    assert set(scores) == set(kids)


@pytest.mark.asyncio
async def test_overflow_candidates_are_scored_not_dropped():
    io = FakeIO()
    graph, root, kids = _graph_with_children(8)

    scores = await _policy(io).evaluate_batch(graph, root, kids)

    assert set(scores) == set(kids), "candidates past the cap went unscored"
    assert all(graph.get_node(k).score is not None for k in kids)
    # Still bounded per prompt: 8 candidates -> two 5/3 prompts, never one prompt of 8.
    assert [len(c) for c in io.calls] == [5, 3]


@pytest.mark.asyncio
async def test_chunk_ceiling_bounds_the_number_of_calls():
    io = FakeIO()
    graph, root, kids = _graph_with_children(2 * _MAX_BATCH_CHUNKS + 4)

    await _policy(io, cap=1).evaluate_batch(graph, root, kids)

    assert len(io.calls) == _MAX_BATCH_CHUNKS


@pytest.mark.asyncio
async def test_empty_candidate_list_makes_no_call():
    io = FakeIO()
    graph, root, _kids = _graph_with_children(0)

    assert await _policy(io).evaluate_batch(graph, root, []) == {}
    assert io.calls == []
