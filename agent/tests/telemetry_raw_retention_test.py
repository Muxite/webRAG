"""``telemetry_raw`` was dropped wholesale from the persisted result below verbosity 3.

``idea_test_runner`` popped the entire block, and every execution variant separately deleted
its JSONL trace on success. Those two losses compound: at the default
``IDEA_TEST_REPORT_VERBOSITY=1`` a finished cell keeps neither the trace file NOR the in-JSON
copy, which is why the four-way baseline had 0 of 96 cells with any recoverable per-step
record and its forensics had to be inferred from graph topology.

The block was dropped for size, and that concern is real -- but the size lives almost entirely
in ``documents_seen`` / ``chroma_stored`` / ``chroma_retrieved`` (page bodies and embedding
payloads), not in the parts forensics needs. Slimming keeps ``timings`` (now carrying the
call intervals that make concurrency provable), ``decisions`` and ``events``, and discards the
bulk.

No network: sessions are built in-process.
"""
from __future__ import annotations

import time

from agent.app.telemetry import TelemetrySession
from agent.app.testing.utils import slim_telemetry_raw


def _populated_summary() -> dict:
    session = TelemetrySession(enabled=True, mandate="m", correlation_id="c")
    session.record_timing("llm_call", time.perf_counter(), True)
    session.record_decision("expansion", node_id="n1", chosen="visit")
    session.events.append({"event": "connector_io", "payload": {"chars": 12}})
    session.documents_seen.append({"url": "https://example.org", "content": "x" * 50_000})
    session.chroma_stored.append({"id": "d1", "embedding": [0.0] * 1536})
    session.chroma_retrieved.append({"id": "d1", "score": 0.9})
    return session.summary()


def test_forensic_fields_are_kept():
    slim = slim_telemetry_raw(_populated_summary())
    assert slim["timings"]
    assert slim["decisions"]
    assert slim["events"]


def test_call_intervals_survive_the_slimming():
    """The whole point of keeping ``timings``: proving concurrency after the fact."""
    slim = slim_telemetry_raw(_populated_summary())
    assert "t_start" in slim["timings"][0]
    assert "t_end" in slim["timings"][0]


def test_bulk_payloads_are_dropped():
    slim = slim_telemetry_raw(_populated_summary())
    for dropped in ("documents_seen", "chroma_stored", "chroma_retrieved"):
        assert dropped not in slim


def test_identifying_metadata_is_kept():
    slim = slim_telemetry_raw(_populated_summary())
    assert slim["correlation_id"] == "c"
    assert "duration" in slim


def test_slimming_shrinks_the_payload():
    import json
    full = _populated_summary()
    slim = slim_telemetry_raw(full)
    assert len(json.dumps(slim, default=str)) < len(json.dumps(full, default=str))


def test_none_and_empty_are_handled():
    assert slim_telemetry_raw(None) is None
    assert slim_telemetry_raw({}) == {}


def test_a_non_dict_is_returned_untouched():
    """Defensive: the block comes from variant code, not all of which is the same shape."""
    assert slim_telemetry_raw("unexpected") == "unexpected"


def _capture_event() -> dict:
    """A connector_io event as recorded when LLM I/O capture is ON."""
    return {
        "event": "connector_io",
        "payload": {
            "connector": "ConnectorLLM",
            "direction": "in",
            "payload": {
                "prompt_chars": 21500,
                "prompt_text": "x" * 21500,
                "messages": [{"role": "user", "content": "y" * 21500}],
            },
        },
    }


def test_captured_raw_text_is_stripped_from_the_result_json():
    """Raw text belongs in the JSONL trace; multiplying it into every result JSON is the
    bloat the old wholesale pop was defending against."""
    slim = slim_telemetry_raw({"events": [_capture_event()]})
    inner = slim["events"][0]["payload"]["payload"]
    assert "prompt_text" not in inner
    assert "messages" not in inner


def test_stripping_keeps_the_counts_that_observability_reads():
    """``llm.calls`` and the char/word totals are derived from these fields."""
    slim = slim_telemetry_raw({"events": [_capture_event()]})
    inner = slim["events"][0]["payload"]["payload"]
    assert inner["prompt_chars"] == 21500
    assert slim["events"][0]["payload"]["connector"] == "ConnectorLLM"


def test_stripping_does_not_mutate_the_input():
    """The live session still holds the text -- the trace writer reads it after this runs."""
    event = _capture_event()
    slim_telemetry_raw({"events": [event]})
    assert "prompt_text" in event["payload"]["payload"]


def test_events_without_captured_text_are_untouched():
    plain = {"event": "connector_io", "payload": {"payload": {"prompt_chars": 12}}}
    slim = slim_telemetry_raw({"events": [plain]})
    assert slim["events"][0] == plain


def test_malformed_events_do_not_raise():
    slim = slim_telemetry_raw({"events": ["nonsense", None, {}, {"payload": "flat"}]})
    assert len(slim["events"]) == 4
