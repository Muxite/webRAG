"""Offline unit tests for agent.app.interactive.thoughts.ThoughtBuffer.

No network, no live LLM: telemetry is a plain fake object with .decisions /
.events lists that the tests mutate directly to simulate the engine running.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "services"))

from agent.app.interactive.thoughts import Thought, ThoughtBuffer


class FakeTelemetry:
    """Minimal duck-type of TelemetrySession: just the two lists ThoughtBuffer reads."""

    def __init__(self):
        self.decisions = []
        self.events = []


def _connector_io_event(direction, operation, payload, error=None):
    entry = {
        "event": "connector_io",
        "payload": {
            "connector": "ConnectorLLM",
            "direction": direction,
            "operation": operation,
            "payload": payload,
        },
    }
    if error:
        entry["payload"]["error"] = error
    return entry


def test_watermark_isolation_across_two_steps():
    telem = FakeTelemetry()
    buf = ThoughtBuffer(telem, max_steps=20)

    buf.capture(step_index=0, node_id="n0")
    telem.decisions.append({"stage": "expansion", "node_id": "n0"})
    telem.events.append({"event": "misc", "payload": {}})
    t0 = buf.collect(step_index=0, node_id="n0")

    buf.capture(step_index=1, node_id="n1")
    telem.decisions.append({"stage": "evaluation", "node_id": "n1"})
    telem.events.append({"event": "misc2", "payload": {}})
    t1 = buf.collect(step_index=1, node_id="n1")

    assert len(t0.decisions) == 1
    assert t0.decisions[0]["stage"] == "expansion"
    assert len(t0.events) == 1

    # Step 1's collect must only see what happened after step 1's watermark.
    assert len(t1.decisions) == 1
    assert t1.decisions[0]["stage"] == "evaluation"
    assert len(t1.events) == 1
    assert t1.events[0]["event"] == "misc2"


def test_ring_eviction_bounds_memory():
    telem = FakeTelemetry()
    buf = ThoughtBuffer(telem, max_steps=3)

    for i in range(5):
        buf.capture(step_index=i, node_id=f"n{i}")
        telem.decisions.append({"stage": "action", "node_id": f"n{i}"})
        buf.collect(step_index=i, node_id=f"n{i}")

    # Only the last 3 should remain.
    assert buf.get(0) is None
    assert buf.get(1) is None
    assert buf.get(2) is not None
    assert buf.get(3) is not None
    assert buf.get(4) is not None
    assert buf.latest().step_index == 4


def test_summarized_payload_degrades_to_no_raw_text_and_sets_truncated():
    telem = FakeTelemetry()
    buf = ThoughtBuffer(telem, max_steps=20)

    buf.capture(step_index=0, node_id="n0")
    # Shape produced by ConnectorBase._summarize_payload when full_capture is off:
    # string values become {"chars": N}.
    telem.events.append(_connector_io_event(
        "in", "llm_query", {"model": {"chars": 8}, "prompt_chars": 42},
    ))
    telem.events.append(_connector_io_event(
        "out", "llm_query", {"model": {"chars": 8}, "completion_chars": 10},
    ))
    thought = buf.collect(step_index=0, node_id="n0")

    assert thought.truncated is True
    assert len(thought.llm_calls) == 1
    call = thought.llm_calls[0]
    assert call["prompt_text"] is None
    assert call["completion_text"] is None


def test_full_capture_payload_extracts_llm_call_when_text_present():
    telem = FakeTelemetry()
    buf = ThoughtBuffer(telem, max_steps=20)

    buf.capture(step_index=0, node_id="n0")
    telem.events.append(_connector_io_event(
        "in", "llm_query", {"model": "gpt-5-mini", "prompt_text": "hello world"},
    ))
    telem.events.append(_connector_io_event(
        "out", "llm_query", {"model": "gpt-5-mini", "completion_text": "hi there"},
    ))
    thought = buf.collect(step_index=0, node_id="n0")

    assert thought.truncated is False
    assert len(thought.llm_calls) == 1
    call = thought.llm_calls[0]
    assert call["model"] == "gpt-5-mini"
    assert call["prompt_text"] == "hello world"
    assert call["completion_text"] == "hi there"


def test_full_capture_payload_carries_messages_when_present():
    telem = FakeTelemetry()
    buf = ThoughtBuffer(telem, max_steps=20)

    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hello world"},
    ]
    buf.capture(step_index=0, node_id="n0")
    telem.events.append(_connector_io_event(
        "in", "llm_query", {"model": "gpt-5-mini", "prompt_text": "hello world", "messages": messages},
    ))
    telem.events.append(_connector_io_event(
        "out", "llm_query", {"model": "gpt-5-mini", "completion_text": "hi there"},
    ))
    thought = buf.collect(step_index=0, node_id="n0")

    call = thought.llm_calls[0]
    assert call["messages"] == messages


def test_messages_absent_does_not_add_key():
    telem = FakeTelemetry()
    buf = ThoughtBuffer(telem, max_steps=20)

    buf.capture(step_index=0, node_id="n0")
    telem.events.append(_connector_io_event(
        "in", "llm_query", {"model": "gpt-5-mini", "prompt_text": "hello world"},
    ))
    telem.events.append(_connector_io_event(
        "out", "llm_query", {"model": "gpt-5-mini", "completion_text": "hi there"},
    ))
    thought = buf.collect(step_index=0, node_id="n0")

    call = thought.llm_calls[0]
    assert "messages" not in call


def test_telemetry_none_never_raises():
    buf = ThoughtBuffer(None, max_steps=5)
    buf.capture(step_index=0, node_id="n0")
    thought = buf.collect(step_index=0, node_id="n0")

    assert isinstance(thought, Thought)
    assert thought.decisions == []
    assert thought.events == []
    assert thought.llm_calls == []
    assert thought.truncated is False
    assert buf.latest() is thought
    assert buf.get(0) is thought
    assert buf.get(99) is None


def test_malformed_records_are_skipped_defensively():
    telem = FakeTelemetry()
    buf = ThoughtBuffer(telem, max_steps=20)

    buf.capture(step_index=0, node_id="n0")
    # Non-dict entries, missing keys, wrong types: none of this should raise.
    telem.decisions.append("not-a-dict")
    telem.decisions.append(None)
    telem.decisions.append({"stage": "action"})
    telem.events.append("also-not-a-dict")
    telem.events.append({"event": "connector_io"})  # missing "payload"
    telem.events.append({"event": "connector_io", "payload": "not-a-dict-either"})
    telem.events.append({"event": "connector_io", "payload": {"operation": "llm_query"}})  # no direction
    telem.events.append({"event": "connector_io", "payload": {
        "direction": "in", "operation": "llm_query", "payload": "not-a-dict",
    }})

    thought = buf.collect(step_index=0, node_id="n0")

    # Non-dict decision entries are dropped defensively; only the well-formed one survives.
    assert thought.decisions == [{"stage": "action"}]
    # Malformed connector_io events (missing/bad payload or direction) yield no llm_call.
    assert thought.llm_calls == [{"model": None, "prompt_text": None, "completion_text": None, "usage": None}]
    assert thought.truncated is False


def test_collect_without_prior_capture_still_returns_empty_thought():
    telem = FakeTelemetry()
    telem.decisions.append({"stage": "action"})
    buf = ThoughtBuffer(telem, max_steps=5)

    # No matching capture() watermark for step 7: should degrade gracefully,
    # not crash, and not retroactively attribute pre-existing telemetry.
    thought = buf.collect(step_index=7, node_id="n7")
    assert isinstance(thought, Thought)
    assert thought.step_index == 7
    assert thought.node_id == "n7"
