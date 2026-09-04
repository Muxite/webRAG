"""Bug 3: langgraph_solver's native transport had no per-turn tool-call cap or dedup.

Live evidence: mint01_q7b_220_..._langgraph_react_..._r1 -- ONE completion returned 4,925
tool_calls (~493 identical repeats of each of 5 URLs), and the host actually executed
thousands of them (2,461 http requests, 28 min wall clock) because `create_react_agent`'s
built-in ToolNode dispatches every `tool_call` on the AIMessage with no cap or dedup.

The fix hooks `create_react_agent`'s `post_model_hook` (runs after the model produces the
AIMessage, before the "tools" node dispatches it) to:
  (a) collapse repeated (name, canonical-args) calls within one turn to a single execution,
  (b) hard-cap the calls actually sent to "tools" per turn at `MAX_TOOL_CALLS_PER_TURN`,
  (c) record a telemetry marker whenever either kicks in,
  (d) leave a normal turn (a handful of distinct calls, no duplicates, under the cap)
      byte-identical -- the hook returns `{}` (no state update) so the router sends the
      SAME tool_calls list it always did.

These tests exercise `_tool_call_guard_hook` directly against `AIMessage`/state shapes, the
same level `langgraph_solver_test.py` already tests `_msg_tool_calls`/`_trim_messages` at --
no live model or graph run needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.app.langgraph_solver import (
    MAX_TOOL_CALLS_PER_TURN,
    _make_tool_call_guard_hook,
)


def _call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id}


def _telemetry():
    t = MagicMock()
    t.record_event = MagicMock()
    return t


def test_normal_turn_with_a_handful_of_distinct_calls_is_untouched():
    telemetry = _telemetry()
    hook = _make_tool_call_guard_hook(telemetry)
    calls = [_call("search", {"query": "a"}, "1"),
             _call("search", {"query": "b"}, "2"),
             _call("visit", {"url": "https://example.com/x"}, "3")]
    ai = AIMessage(content="", tool_calls=calls)
    state = {"messages": [HumanMessage(content="task"), ai]}
    update = hook(state)
    assert update == {}
    telemetry.record_event.assert_not_called()


def test_dedup_collapses_identical_calls_to_one_execution():
    telemetry = _telemetry()
    hook = _make_tool_call_guard_hook(telemetry)
    calls = [_call("visit", {"url": "https://example.com/x"}, "1"),
             _call("visit", {"url": "https://example.com/x"}, "2"),
             _call("visit", {"url": "https://example.com/x"}, "3")]
    ai = AIMessage(content="", tool_calls=calls)
    state = {"messages": [HumanMessage(content="task"), ai]}
    update = hook(state)
    assert update != {}
    new_messages = update["messages"]
    new_ai = next(m for m in new_messages if isinstance(m, AIMessage))
    assert new_ai.id == ai.id  # same id -> replaces the original in state via add_messages
    assert len(new_ai.tool_calls) == 1
    assert new_ai.tool_calls[0]["id"] == "1"
    # The two duplicates get an immediate synthetic ToolMessage instead of reaching "tools".
    synthetic = [m for m in new_messages if isinstance(m, ToolMessage)]
    assert {m.tool_call_id for m in synthetic} == {"2", "3"}
    telemetry.record_event.assert_called_once()
    event_name, payload = telemetry.record_event.call_args.args
    assert payload["dropped_duplicate"] == 2
    assert payload["dropped_cap"] == 0


def test_cap_enforced_and_marker_recorded():
    telemetry = _telemetry()
    hook = _make_tool_call_guard_hook(telemetry)
    calls = [_call("visit", {"url": f"https://example.com/{i}"}, str(i))
             for i in range(MAX_TOOL_CALLS_PER_TURN + 5)]
    ai = AIMessage(content="", tool_calls=calls)
    state = {"messages": [HumanMessage(content="task"), ai]}
    update = hook(state)
    assert update != {}
    new_ai = next(m for m in update["messages"] if isinstance(m, AIMessage))
    assert len(new_ai.tool_calls) == MAX_TOOL_CALLS_PER_TURN
    dropped_ids = {m.tool_call_id for m in update["messages"] if isinstance(m, ToolMessage)}
    assert len(dropped_ids) == 5
    telemetry.record_event.assert_called_once()
    _, payload = telemetry.record_event.call_args.args
    assert payload["dropped_cap"] == 5
    assert payload["cap"] == MAX_TOOL_CALLS_PER_TURN


def test_massive_duplicate_burst_like_live_evidence_collapses_to_one_kept_call():
    # Mirrors the live incident shape: ~493 identical repeats each of 5 distinct URLs.
    telemetry = _telemetry()
    hook = _make_tool_call_guard_hook(telemetry)
    urls = [f"https://example.com/page{i}" for i in range(5)]
    calls = []
    call_id = 0
    for _ in range(493):
        for url in urls:
            calls.append(_call("visit", {"url": url}, str(call_id)))
            call_id += 1
    ai = AIMessage(content="", tool_calls=calls)
    state = {"messages": [HumanMessage(content="task"), ai]}
    update = hook(state)
    new_ai = next(m for m in update["messages"] if isinstance(m, AIMessage))
    # Only the 5 unique URLs are ever sent to "tools" -- capped further to MAX_TOOL_CALLS_PER_TURN.
    assert len(new_ai.tool_calls) == min(5, MAX_TOOL_CALLS_PER_TURN)


def test_hook_is_a_noop_when_last_message_has_no_tool_calls():
    telemetry = _telemetry()
    hook = _make_tool_call_guard_hook(telemetry)
    state = {"messages": [HumanMessage(content="task"), AIMessage(content="final answer")]}
    assert hook(state) == {}
    telemetry.record_event.assert_not_called()


def test_hook_tolerates_missing_telemetry():
    hook = _make_tool_call_guard_hook(None)
    calls = [_call("visit", {"url": "https://example.com/x"}, "1"),
             _call("visit", {"url": "https://example.com/x"}, "2")]
    ai = AIMessage(content="", tool_calls=calls)
    state = {"messages": [HumanMessage(content="task"), ai]}
    update = hook(state)  # must not raise
    assert len(next(m for m in update["messages"] if isinstance(m, AIMessage)).tool_calls) == 1
