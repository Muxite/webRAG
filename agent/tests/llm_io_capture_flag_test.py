"""Raw LLM I/O capture was welded to ``IDEA_TEST_REPORT_VERBOSITY >= 3``.

Full prompt/completion capture already existed in the connectors, but the only way to switch it
on also inflated every other artifact in the run. ``IDEA_TEST_CAPTURE_LLM_IO`` decouples the
two: the raw text goes to the JSONL trace, ``slim_telemetry_raw`` keeps it out of the result
JSON, and verbosity stays free to control reporting.

Also pinned here: the ``langgraph_react`` arm's ``connector_io`` events are SYNTHESIZED --
reconstructed by replaying the final message list at end-of-run, so all of them carry an
end-of-run timestamp rather than their real call time. Their counts are sound but their
ordering is meaningless, and the marker is what lets analysis refuse to draw a timing
conclusion from that arm instead of silently reading a column of identical timestamps as
"perfectly concurrent".

No network: the solver's event emitter is driven directly.
"""
from __future__ import annotations

import pytest

from agent.app.trace_recorder import llm_io_capture_enabled


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_CAPTURE_LLM_IO", raising=False)


def test_capture_is_off_at_default_verbosity():
    assert llm_io_capture_enabled(1) is False


def test_verbosity_three_still_enables_capture():
    """Existing verbosity-3 workflows must be unchanged by the decoupling."""
    assert llm_io_capture_enabled(3) is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
def test_the_flag_enables_capture_without_raising_verbosity(monkeypatch, value):
    monkeypatch.setenv("IDEA_TEST_CAPTURE_LLM_IO", value)
    assert llm_io_capture_enabled(1) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_falsey_flag_leaves_the_verbosity_rule_in_charge(monkeypatch, value):
    monkeypatch.setenv("IDEA_TEST_CAPTURE_LLM_IO", value)
    assert llm_io_capture_enabled(1) is False
    assert llm_io_capture_enabled(3) is True


def _synth_events(full_capture: bool):
    from langchain_core.messages import AIMessage, HumanMessage

    from agent.app.langgraph_solver import _record_io_parity
    from agent.app.telemetry import TelemetrySession

    session = TelemetrySession(enabled=True)
    _record_io_parity(
        session,
        [HumanMessage(content="which telescope has the largest dish"),
         AIMessage(content="FAST, 500 m")],
        full_capture=full_capture,
    )
    return [e for e in session.events if e.get("event") == "connector_io"]


def test_langgraph_io_events_are_marked_synthesized():
    events = _synth_events(full_capture=False)
    assert events, "expected an in/out pair for the assistant turn"
    for event in events:
        assert event["payload"]["synthesized"] is True


def test_the_marker_is_present_under_full_capture_too():
    events = _synth_events(full_capture=True)
    assert events
    for event in events:
        assert event["payload"]["synthesized"] is True


def test_counts_are_still_recorded_alongside_the_marker():
    """The marker must not disturb what ``summarize_observability`` derives from these."""
    events = _synth_events(full_capture=False)
    inbound = [e for e in events if e["payload"]["direction"] == "in"]
    assert inbound and inbound[0]["payload"]["payload"]["prompt_chars"] > 0
    assert all(e["payload"]["connector"] == "ConnectorLLM" for e in events)
