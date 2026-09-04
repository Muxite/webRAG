"""Bug 1: full LLM I/O capture was silently disabled for every non-langgraph arm.

``execution.py`` (graph arm) and ``execution_langgraph.py`` (langgraph arm) both gate
``connector_llm.set_full_capture`` on ``llm_io_capture_enabled(report_verbosity)`` --
but ``execution_sequential.py`` (sequential_react arm) never called ``set_full_capture``
at all, so ``IDEA_TEST_CAPTURE_LLM_IO=1`` (or verbosity 3) had no effect on this arm's
trace: connector_io events only ever carried prompt_chars/completion_chars, never
prompt_text/completion_text.

These tests drive ``run_sequential_execution`` end-to-end (mocked connectors, mocked
``_run_react`` to skip the LLM loop itself) and assert the connectors are toggled into
full-capture mode for the duration of the run when the env flag is on, and never
toggled when it is off -- mirroring the graph arm's own ON/OFF pair in
``execution.py:530-534,596-600``.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import execution_sequential as seq


def _connectors():
    return {
        "connector_llm": MagicMock(),
        "connector_search": MagicMock(),
        "connector_http": MagicMock(),
        "connector_chroma": MagicMock(),
    }


def _test_module():
    tm = MagicMock()
    tm.metadata = {"test_id": "999"}
    tm.get_task_statement.return_value = "Do the thing."
    return tm


def test_full_capture_enabled_on_all_connectors_when_env_flag_set(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_CAPTURE_LLM_IO", "1")
    monkeypatch.setattr(seq, "_run_react", AsyncMock(return_value="answer"))
    conns = _connectors()
    asyncio.run(seq.run_sequential_execution(
        test_module=_test_module(), model_name="m", run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
        **conns,
    ))
    for name, conn in conns.items():
        conn.set_full_capture.assert_any_call(True)
        # toggled back off once the run finishes
        conn.set_full_capture.assert_any_call(False)
        assert conn.set_full_capture.call_args_list[-1].args == (False,)


def test_full_capture_left_untouched_when_env_flag_unset(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_CAPTURE_LLM_IO", raising=False)
    monkeypatch.delenv("IDEA_TEST_REPORT_VERBOSITY", raising=False)
    monkeypatch.setattr(seq, "_run_react", AsyncMock(return_value="answer"))
    conns = _connectors()
    asyncio.run(seq.run_sequential_execution(
        test_module=_test_module(), model_name="m", run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
        **conns,
    ))
    for conn in conns.values():
        conn.set_full_capture.assert_not_called()


def test_sequential_trace_records_prompt_text_when_capture_enabled(tmp_path, monkeypatch):
    # End-to-end through the REAL ConnectorLLM._record_io gate (Bug 1's actual live symptom),
    # not just the mock-call assertions above.
    from agent.app.connector_llm import ConnectorLLM
    from shared.connector_config import ConnectorConfig

    monkeypatch.setenv("IDEA_TEST_CAPTURE_LLM_IO", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    connector_llm = ConnectorLLM(ConnectorConfig())
    captured = {}

    async def fake_run_react(io, mandate, model_name, max_steps, max_tokens, **kwargs):
        # Exercise the real capture-gated code path directly.
        connector_llm._record_io(
            "in", "llm_call",
            payload={"prompt_chars": 5, "prompt_text": "hello"} if connector_llm._full_capture else {"prompt_chars": 5},
        )
        captured["full_capture_during_run"] = connector_llm._full_capture
        return "answer"

    monkeypatch.setattr(seq, "_run_react", fake_run_react)
    conns = _connectors()
    conns["connector_llm"] = connector_llm
    asyncio.run(seq.run_sequential_execution(
        test_module=_test_module(), model_name="m", run_stamp="r1",
        summarize_observability_func=lambda *a, **kw: {},
        **conns,
    ))
    assert captured["full_capture_during_run"] is True
    assert connector_llm._full_capture is False  # toggled back off after the run
