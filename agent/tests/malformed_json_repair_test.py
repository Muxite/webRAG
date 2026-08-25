"""Three action sites used to fail OPEN the instant ``json.loads`` raised.

``VisitLeafAction._select_links_with_llm`` fell straight to ``candidate_urls[:link_count]``,
``MergeLeafAction`` to a "Failed to parse LLM response"/not-achieved stub, and
``VerifyLeafAction`` to ``UNVERIFIABLE`` -- the model's answer was discarded without ever
being asked to fix its own formatting.

All three now make ONE bounded repair call (shared ``LeafAction._repair_malformed_json``)
carrying the original instruction, the malformed text and the parse error, and fall back
exactly as before when that call also fails. The happy path is untouched: no repair call,
no telemetry event.

No network: every LLM response is scripted.
"""
from __future__ import annotations

import asyncio
import json

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction, VerifyLeafAction, VisitLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.telemetry import TelemetrySession


MALFORMED = 'Sure! Here is the JSON you asked for: {"selected": ['

SETTINGS = load_idea_dag_settings()

CANDIDATES = [
    "https://en.wikipedia.org/wiki/Brooklyn_Bridge",
    "https://en.wikipedia.org/wiki/Brooklyn_Bridge_Park",
    "https://en.wikipedia.org/wiki/Brooklyn_Bridge_history",
]

MERGE_OK = json.dumps({
    "summary": "repaired summary", "key_findings": ["k"], "goal_achieved": False,
    "goal_evaluation": "partial", "missing_requirements": ["more"],
})
VERIFY_OK = json.dumps({
    "verdict": "TRUE", "confidence": 0.9, "supporting_url": "https://e.example",
    "contradicting_url": "", "quote": "q", "reasoning": "r",
})


class _ScriptedIO:
    """Returns the scripted replies in order, recording every prompt it was handed."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = []
        self.collection_name = "agent_memory"
        self.telemetry = TelemetrySession(enabled=True)

    def build_llm_payload(self, messages=None, **kw):
        return {"messages": messages, **kw}

    async def query_llm_with_fallback(self, payload, model_name=None, fallback_model=None,
                                      timeout_seconds=None):
        self.calls.append(payload)
        if not self._responses:
            raise AssertionError("action made more LLM calls than the test scripted")
        return self._responses.pop(0)

    def events(self, name):
        return [e for e in self.telemetry.events if e["event"] == name]

    def repair_prompt(self):
        return "\n".join(m["content"] for m in self.calls[-1]["messages"])


def _merge_graph():
    g = IdeaDag(root_title="root")
    node = g.add_child(g.root_id(), "merge me", status=IdeaNodeStatus.PENDING)
    node.details[DetailKey.MERGED_RESULTS.value] = [
        {"title": "a", "content": "alpha"}, {"title": "b", "content": "beta"},
    ]
    return g, node.node_id


def _run_merge(io):
    g, node_id = _merge_graph()
    return asyncio.run(MergeLeafAction(settings=load_idea_dag_settings()).execute(g, node_id, io))


def _verify_graph():
    g = IdeaDag(root_title="root")
    g.add_child(
        g.root_id(), "visit the source",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "success": True, "action": IdeaActionType.VISIT.value,
                "url": "https://e.example", "content": "The claim is stated here.",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    node = g.add_child(
        g.root_id(), "verify me",
        details={DetailKey.ACTION.value: IdeaActionType.VERIFY.value, "claim": "the claim"},
        status=IdeaNodeStatus.PENDING,
    )
    return g, node.node_id


def _run_verify(io):
    g, node_id = _verify_graph()
    return asyncio.run(VerifyLeafAction(settings=load_idea_dag_settings()).execute(g, node_id, io))


def _run_selection(io, link_count=2):
    action = VisitLeafAction(settings=load_idea_dag_settings())
    return asyncio.run(action._select_links_with_llm("the bridge", CANDIDATES, link_count, io))


# ---------------------------------------------------------------------------
# Happy path: byte-identical to before -- no repair call, no telemetry event
# ---------------------------------------------------------------------------

def test_link_selection_happy_path_makes_no_repair_call():
    io = _ScriptedIO(json.dumps({"selected": [CANDIDATES[2]]}))
    assert _run_selection(io, link_count=1) == [CANDIDATES[2]]
    assert len(io.calls) == 1
    assert io.events("malformed_llm_action") == []
    assert io.events("malformed_llm_action_repaired") == []


def test_merge_happy_path_makes_no_repair_call():
    io = _ScriptedIO(MERGE_OK)
    result = _run_merge(io)
    assert result["synthesized"]["summary"] == "repaired summary"
    assert len(io.calls) == 1
    assert io.events("malformed_llm_action") == []


def test_verify_happy_path_makes_no_repair_call():
    io = _ScriptedIO(VERIFY_OK)
    assert _run_verify(io)["verdict"] == "TRUE"
    assert len(io.calls) == 1
    assert io.events("malformed_llm_action") == []


# ---------------------------------------------------------------------------
# Malformed once -> exactly ONE repair call, counter increments, repair used
# ---------------------------------------------------------------------------

def test_link_selection_repairs_a_malformed_response():
    io = _ScriptedIO(MALFORMED, json.dumps({"selected": [CANDIDATES[1]]}))
    assert _run_selection(io, link_count=1) == [CANDIDATES[1]]
    assert len(io.calls) == 2
    assert len(io.events("malformed_llm_action")) == 1
    assert io.events("malformed_llm_action")[0]["payload"]["site"] == "visit_link_selection"
    assert io.events("malformed_llm_action_repaired")[0]["payload"]["repaired"] is True


def test_merge_repairs_a_malformed_response():
    io = _ScriptedIO("not json at all {", MERGE_OK)
    result = _run_merge(io)
    assert result["synthesized"]["summary"] == "repaired summary"
    assert len(io.calls) == 2
    assert len(io.events("malformed_llm_action")) == 1
    assert io.events("malformed_llm_action")[0]["payload"]["site"] == "merge_synthesis"
    assert io.events("malformed_llm_action_repaired")[0]["payload"]["repaired"] is True


def test_verify_repairs_a_malformed_response():
    io = _ScriptedIO("VERDICT: TRUE (sorry, no JSON)", VERIFY_OK)
    result = _run_verify(io)
    assert result["verdict"] == "TRUE"
    assert result["confidence"] == 0.9
    assert len(io.calls) == 2
    assert len(io.events("malformed_llm_action")) == 1
    assert io.events("malformed_llm_action")[0]["payload"]["site"] == "verify_verdict"


def test_repair_prompt_carries_the_instruction_the_bad_text_and_the_error():
    io = _ScriptedIO("VERDICT: TRUE (sorry, no JSON)", VERIFY_OK)
    _run_verify(io)
    prompt = io.repair_prompt()
    assert "VERDICT: TRUE (sorry, no JSON)" in prompt          # the malformed text
    assert "the claim" in prompt                               # the original instruction
    assert "JSON PARSE ERROR" in prompt and "line 1" in prompt  # the parse error


# ---------------------------------------------------------------------------
# Repair also fails -> the pre-existing fallback, unchanged
# ---------------------------------------------------------------------------

def test_link_selection_falls_back_when_repair_also_fails():
    io = _ScriptedIO(MALFORMED, "still not json")
    assert _run_selection(io) == CANDIDATES[:2]
    assert len(io.calls) == 2
    assert len(io.events("malformed_llm_action")) == 1
    assert io.events("malformed_llm_action_repaired")[0]["payload"]["repaired"] is False


def test_merge_falls_back_when_repair_also_fails():
    io = _ScriptedIO("not json at all {", "still not json")
    result = _run_merge(io)
    assert result["synthesized"]["goal_evaluation"] == "Failed to parse LLM response"
    assert result["synthesized"]["summary"] == "not json at all {"
    assert result["goal_achieved"] is False
    assert len(io.calls) == 2


def test_verify_falls_back_when_repair_also_fails():
    io = _ScriptedIO("VERDICT: TRUE (sorry, no JSON)", "still not json")
    result = _run_verify(io)
    assert result["verdict"] == "UNVERIFIABLE"
    assert result["confidence"] == 0.0
    assert result["reasoning"] == "Failed to parse verify response"
    assert len(io.calls) == 2


def test_repair_returning_a_non_object_falls_back_too():
    """Valid JSON of the wrong shape (a bare list) is not a usable repair."""
    io = _ScriptedIO("VERDICT: TRUE (sorry, no JSON)", json.dumps(["TRUE"]))
    result = _run_verify(io)
    assert result["verdict"] == "UNVERIFIABLE"
    assert io.events("malformed_llm_action_repaired")[0]["payload"]["repaired"] is False


def test_an_empty_repair_response_falls_back_without_raising():
    io = _ScriptedIO(MALFORMED, "")
    assert _run_selection(io) == CANDIDATES[:2]
    assert io.events("malformed_llm_action_repaired")[0]["payload"]["repaired"] is False


def test_repair_works_without_any_telemetry_session_attached():
    """Telemetry is optional on the IO stand-ins the engine hands leaves; the repair
    path must never depend on it."""
    io = _ScriptedIO(MALFORMED, json.dumps({"selected": [CANDIDATES[0]]}))
    io.telemetry = None
    assert _run_selection(io, link_count=1) == [CANDIDATES[0]]
    assert len(io.calls) == 2
