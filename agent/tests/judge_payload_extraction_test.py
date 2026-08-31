"""N3a: the confidence judge must see the output of EVERY action kind, not just web ones.

``judge_step_confidence`` builds its payload from three result fields (``content``,
``content_full``, ``results``). ``visit``/``search`` populate them; ``merge``/``think``/
``verify``/``save`` write their real output under their own keys (``synthesized``,
``thinking_content``, ``verdict``/``quote``/``reasoning``, ``count``), so 43.4% of judged steps
were scored on an empty payload (ADAPTIVE_ENGINE.md 260-263, CONFIDENCE_JUDGE_MISCALIBRATION.md
2a). An earlier pass shipped proposal P1(b) — decline those kinds outright — which removed the
false signal but left the judge blind on nearly half the trajectory.

This module pins P1(a): the judge renders each kind's own output keys into the SAME payload
field (``resolved_content``), so the prompt shape is unchanged and the decline path survives for
results that genuinely carry no output.
"""
from __future__ import annotations

import json

import pytest

from agent.app.got_operations import GoTOperations
from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.action_constants import ActionResultKey
from agent.app.idea_policies.base import DetailKey, IdeaActionType


class _CaptureIO:
    """Fake AgentIO recording the built payload and returning a canned judge response."""

    def __init__(self, response: str):
        self.response = response
        self.last_messages = None
        self.llm_attempts = 0

    def set_telemetry(self, telemetry):
        return None

    def build_llm_payload(self, messages=None, json_mode=None, model_name=None, temperature=None):
        self.last_messages = messages
        return {"messages": messages}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None, timeout_seconds=None):
        self.llm_attempts += 1
        return self.response


def _ops(response='{"confidence": 0.8, "reason": "on track"}'):
    return GoTOperations(
        settings={"got_step_confidence_judge_enabled": True},
        io=_CaptureIO(response),
        memory_manager=None,
    )


def _graph_with_leaf(action: str, result: dict):
    graph = IdeaDag(root_title="root")
    graph.get_node(graph.root_id()).details["mandate"] = "Find the poet's birthplace."
    leaf = graph.add_child(
        graph.root_id(),
        "Resolve a sub-fact",
        details={
            DetailKey.ACTION.value: action,
            DetailKey.IS_LEAF.value: True,
            DetailKey.ACTION_RESULT.value: {ActionResultKey.SUCCESS.value: True, **result},
        },
    )
    return graph, leaf


def _resolved(io) -> dict:
    """The judge's user payload, parsed — the object the LLM was actually shown."""
    return json.loads(io.last_messages[1]["content"])


#: (action, result-as-emitted, a substring of that kind's real output)
_KIND_RESULTS = [
    (
        IdeaActionType.MERGE.value,
        {"synthesized": {"summary": "The poet was born in Parral."}, "raw_response": "{...}"},
        "born in Parral",
    ),
    (
        IdeaActionType.THINK.value,
        {"thinking_content": "Reasoning about the candidates..."},
        "Reasoning about the candidates",
    ),
    (
        IdeaActionType.VERIFY.value,
        {"verdict": "TRUE", "quote": "born in Parral", "reasoning": "the page states it"},
        "the page states it",
    ),
    (IdeaActionType.SAVE.value, {"count": 3}, "3"),
]
_KIND_IDS = ["merge", "think", "verify", "save"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action,result,needle", _KIND_RESULTS, ids=_KIND_IDS)
async def test_judge_sees_each_kinds_own_output(action, result, needle):
    ops = _ops()
    graph, leaf = _graph_with_leaf(action, result)

    verdict = await ops.judge_step_confidence(graph, leaf.node_id)

    assert verdict is not None and verdict["confidence"] == pytest.approx(0.8)
    assert ops.io.llm_attempts == 1
    payload = _resolved(ops.io)
    assert payload["resolved_content"], f"{action} was judged on an empty payload"
    assert needle in payload["resolved_content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action,result,_needle", _KIND_RESULTS, ids=_KIND_IDS)
async def test_judge_prompt_stays_leak_free_for_every_kind(action, result, _needle):
    ops = _ops()
    graph, leaf = _graph_with_leaf(action, result)

    await ops.judge_step_confidence(graph, leaf.node_id)

    assert ops.io.last_messages is not None, f"{action} never built a judge payload"
    blob = json.dumps(ops.io.last_messages)
    for leaked in ("grep_validations", "overall_passed", "overall_score", "ground_truth"):
        assert leaked not in blob, f"judge prompt leaked {leaked!r}"


@pytest.mark.asyncio
async def test_result_with_no_output_at_all_is_still_declined():
    ops = _ops()
    graph, leaf = _graph_with_leaf(IdeaActionType.MERGE.value, {"duration_ms": 12})

    assert await ops.judge_step_confidence(graph, leaf.node_id) is None
    assert ops.io.llm_attempts == 0
    assert ops.io.last_messages is None


@pytest.mark.asyncio
async def test_visit_content_is_preferred_over_the_fallback_rendering():
    ops = _ops()
    graph, leaf = _graph_with_leaf(
        IdeaActionType.VISIT.value, {"content": "page text", "synthesized": {"summary": "nope"}}
    )

    await ops.judge_step_confidence(graph, leaf.node_id)

    assert _resolved(ops.io)["resolved_content"] == "page text"


@pytest.mark.asyncio
async def test_long_fallback_output_is_truncated_like_page_content():
    ops = _ops()
    graph, leaf = _graph_with_leaf(
        IdeaActionType.THINK.value, {"thinking_content": "x" * 9000}
    )

    await ops.judge_step_confidence(graph, leaf.node_id)

    content = _resolved(ops.io)["resolved_content"]
    assert content.endswith("... [truncated]")
    assert len(content) < 9000
