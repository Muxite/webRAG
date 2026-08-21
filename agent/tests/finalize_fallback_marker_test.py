"""D10 (ENGINE_DESIGN_REVIEW): the degraded fallback deliverable is flagged in the schema.

When the finalize LLM call returns nothing, ``build_final_payload`` stitches an answer out of
the graph's leaf results. That payload used to be distinguishable from a real model answer
only by string-matching ``action_summary`` ("Fallback: LLM finalize call failed"), which no
grader should have to do. ``is_fallback_deliverable: True`` marks it structurally.

Emitted on the fallback path ONLY — absence means a normal finalize, matching the other
optional detail keys here (``truncated``, ``unverified_citations``).

Offline: the fake IO returns a scripted (or empty) finalize response, no network.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus


_MANDATE = "Summarize the pasted notes into one sentence."


def _graph() -> IdeaDag:
    g = IdeaDag(root_title="root")
    g.get_node(g.root_id()).details["mandate"] = _MANDATE
    g.add_child(
        g.root_id(), "read the notes",
        details={
            DetailKey.ACTION.value: IdeaActionType.THINK.value,
            DetailKey.ACTION_RESULT.value: {"success": True, "content": "The notes say 42."},
        },
        status=IdeaNodeStatus.DONE,
    )
    return g


class _FakeIO:
    def __init__(self, response):
        self._response = response

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        return self._response


def _run(response):
    return asyncio.run(
        build_final_payload(_FakeIO(response), load_idea_dag_settings(), _graph(), _MANDATE, "m")
    )


def test_empty_response_fallback_payload_is_flagged():
    payload = _run("")
    assert payload["is_fallback_deliverable"] is True
    assert payload["action_summary"] == "Fallback: LLM finalize call failed"


def test_normal_payload_carries_no_marker():
    payload = _run(json.dumps({"deliverable": "The notes say 42.", "summary": "read them"}))
    assert "is_fallback_deliverable" not in payload
    assert payload["final_deliverable"] == "The notes say 42."
