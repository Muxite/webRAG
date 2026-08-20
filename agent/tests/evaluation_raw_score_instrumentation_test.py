"""The recorded evaluation carries the judge's pre-cap opinion alongside the final score.

`details["evaluation"]["score"]` is what selection/pruning consume, and it is written AFTER the
unexecuted-work cap (`evaluation_no_action_result_score_cap`) or the base-score fallback has
rewritten it. That made the flat-sibling-score census unable to tell a judge that genuinely saw
no difference between candidates from one whose spread the cap flattened. `raw_score` + `capped`
record that distinction; `score` itself is untouched.

`raw_score is None` means the judge was never asked (per-node fallback path) or omitted that
candidate from its response -- NOT that it scored zero.
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.evaluation import LlmBatchEvaluationPolicy, LlmEvaluationPolicy


class FakeIO:
    telemetry = None

    def __init__(self, response):
        self._response = response

    def build_llm_payload(self, **kwargs):
        return {"messages": kwargs.get("messages")}

    async def query_llm_with_fallback(self, payload, **kwargs):
        user = payload["messages"][-1]["content"]
        return self._response(user)


def _batch_io(scores_by_position, omit=()):
    def respond(user):
        ids = [c["id"] for c in json.loads(user)["candidates"]]
        return json.dumps(
            {
                "scores": [
                    {"id": cid, "score": scores_by_position[i]}
                    for i, cid in enumerate(ids)
                    if cid not in omit
                ]
            }
        )

    return FakeIO(respond)


def _graph(n, with_result):
    graph = IdeaDag(root_title="mandate", root_details={"mandate": "mandate"})
    root = graph.root_id()
    details = {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}
    if with_result:
        details[DetailKey.ACTION_RESULT.value] = {"success": True}
    kids = graph.expand(
        root, [{"title": f"leaf {i}", "details": dict(details)} for i in range(n)]
    )
    return graph, root, [k.node_id for k in kids]


def _evaluation(graph, node_id):
    return graph.get_node(node_id).details[DetailKey.EVALUATION.value]


@pytest.mark.asyncio
async def test_executed_candidates_record_uncapped_raw_scores():
    graph, root, kids = _graph(2, with_result=True)

    scores = await LlmBatchEvaluationPolicy(_batch_io([0.9, 0.3])).evaluate_batch(graph, root, kids)

    # `score` is what it always was: the judge's value, unchanged for executed work.
    assert [scores[k] for k in kids] == [0.9, 0.3]
    for node_id, expected in zip(kids, [0.9, 0.3]):
        record = _evaluation(graph, node_id)
        assert record["score"] == expected
        assert record["raw_score"] == expected
        assert record["capped"] is False


@pytest.mark.asyncio
async def test_unexecuted_candidates_keep_the_capped_score_and_the_raw_spread():
    graph, root, kids = _graph(2, with_result=False)

    scores = await LlmBatchEvaluationPolicy(_batch_io([0.9, 0.7])).evaluate_batch(graph, root, kids)

    # Cap unchanged: both collapse onto 0.5, which is exactly the flat-score defect.
    assert [scores[k] for k in kids] == [0.5, 0.5]
    for node_id, raw in zip(kids, [0.9, 0.7]):
        record = _evaluation(graph, node_id)
        assert record["score"] == 0.5
        assert record["raw_score"] == raw, "the judge's pre-cap spread was lost"
        assert record["capped"] is True


@pytest.mark.asyncio
async def test_a_candidate_below_the_cap_is_not_marked_capped():
    graph, root, kids = _graph(1, with_result=False)

    await LlmBatchEvaluationPolicy(_batch_io([0.2])).evaluate_batch(graph, root, kids)

    record = _evaluation(graph, kids[0])
    assert record["score"] == 0.2
    assert record["raw_score"] == 0.2
    assert record["capped"] is False


@pytest.mark.asyncio
async def test_a_candidate_the_judge_omitted_has_no_raw_score():
    graph, root, kids = _graph(2, with_result=False)
    io = _batch_io([0.9, 0.7], omit={"2"})

    await LlmBatchEvaluationPolicy(io).evaluate_batch(graph, root, kids)

    fallback = _evaluation(graph, kids[1])
    assert fallback["score"] == 0.4, "base-score fallback unchanged"
    assert fallback["raw_score"] is None, "fabricated a judge opinion the judge never gave"
    assert fallback["capped"] is True


@pytest.mark.asyncio
async def test_per_node_fallback_never_calls_the_judge_so_raw_score_is_none():
    graph, _root, kids = _graph(1, with_result=False)

    def explode(_user):
        raise AssertionError("the per-node fallback must not call the judge")

    score = await LlmEvaluationPolicy(FakeIO(explode)).evaluate(graph, kids[0])

    record = _evaluation(graph, kids[0])
    assert score == 0.4 and record["score"] == 0.4
    assert record["raw_score"] is None
    assert record["capped"] is True


@pytest.mark.asyncio
async def test_per_node_judge_path_records_the_judge_opinion_uncapped():
    """The per-node path only ever reaches the judge for work that already ran.

    Its own `no_action_result_score_cap` branch is unreachable: the fallback above returns
    before it whenever the action has no result, so `capped` is always False here. Only the
    batch path can report a clipped judge opinion.
    """
    graph, _root, kids = _graph(1, with_result=True)
    io = FakeIO(lambda _user: json.dumps({"score": 0.95, "rationale": "looks good"}))

    score = await LlmEvaluationPolicy(io).evaluate(graph, kids[0])

    record = _evaluation(graph, kids[0])
    assert score == 0.95 and record["score"] == 0.95
    assert record["raw_score"] == 0.95
    assert record["capped"] is False
