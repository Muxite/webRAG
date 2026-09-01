"""Offline unit tests for `agent/app/prompted_tools.py` — the framework-free prompted
tool-calling shim extracted from `langgraph_solver.py`'s `_EmulatedToolCallTransport`.

No LLM, no network, no langgraph/langchain import anywhere in the module under test.
"""
from __future__ import annotations

import asyncio

import pytest

from agent.app.prompted_tools import (
    JsonExtraction, ToolCall, ToolLoopExhausted, ToolLoopStep, ToolSpec,
    build_protocol, extract_decision, invalid_action_message, repair_json_text, run_tool_loop,
)


# --------------------------------------------------------------------------- protocol prompt


def test_build_protocol_lists_each_tool_with_its_slots():
    protocol = build_protocol([
        ToolSpec(name="search", description="Search the web.", arg_names=("query",)),
        ToolSpec(name="visit", description="Open a URL.", arg_names=("url",)),
    ])
    assert "- search(query): Search the web." in protocol
    assert "- visit(url): Open a URL." in protocol
    assert "finish(answer)" in protocol


def test_build_protocol_skips_a_finish_spec_since_the_footer_already_documents_it():
    protocol = build_protocol([ToolSpec(name="finish", description="submit")])
    assert protocol.count("finish(answer)") == 1


def test_invalid_action_message_lists_known_tools_plus_finish():
    msg = invalid_action_message(["search", "visit"])
    assert "search" in msg and "visit" in msg and "finish" in msg


# --------------------------------------------------------------------------- JSON extraction


def test_extract_decision_reads_a_bare_object():
    result = extract_decision('{"action": "search"}')
    assert result.value == {"action": "search"}
    assert result.source == "direct"
    assert result.repaired is False


def test_extract_decision_reads_a_fenced_object():
    raw = 'Sure!\n```json\n{"action": "visit", "args": {"url": "https://x/y"}}\n```\n'
    result = extract_decision(raw)
    assert result.value == {"action": "visit", "args": {"url": "https://x/y"}}
    assert result.source == "fenced"


def test_extract_decision_takes_the_first_dict_of_a_list():
    result = extract_decision('[{"action": "search"}]')
    assert result.value == {"action": "search"}


def test_extract_decision_finds_a_span_inside_prose():
    result = extract_decision('I will do this: {"action": "search", "args": {}} now.')
    assert result.value == {"action": "search", "args": {}}
    assert result.source == "span"


def test_extract_decision_returns_failed_for_prose():
    result = extract_decision("I think the answer is 1786.")
    assert result.value is None
    assert result.source == "failed"


def test_extract_decision_returns_failed_for_empty_input():
    assert extract_decision("").value is None
    assert extract_decision(None).value is None


def test_extract_decision_repairs_a_trailing_comma():
    result = extract_decision('{"action": "search", "args": {"query": "q"},}')
    assert result.value == {"action": "search", "args": {"query": "q"}}
    assert result.repaired is True
    assert result.repair_attempts >= 1


def test_extract_decision_repairs_single_quotes():
    result = extract_decision("{'action': 'search', 'args': {'query': 'q'}}")
    assert result.value == {"action": "search", "args": {"query": "q"}}
    assert result.repaired is True


def test_extract_decision_repairs_an_unterminated_string():
    result = extract_decision('{"action": "search", "args": {"query": "q}')
    assert result.value == {"action": "search", "args": {"query": "q"}}
    assert result.repaired is True


def test_extract_decision_repairs_a_truncated_close():
    result = extract_decision('{"action": "finish", "args": {"answer": "42"}')
    assert result.value == {"action": "finish", "args": {"answer": "42"}}
    assert result.repaired is True


def test_extract_decision_repairs_a_bare_action_line():
    result = extract_decision("action: search")
    assert result.value == {"action": "search"}
    assert result.repaired is True


# --------------------------------------------------------------------------- repair idempotency


@pytest.mark.parametrize("valid_json", [
    '{"action": "search", "args": {"query": "mont blanc"}}',
    '{"action": "finish", "args": {"answer": "1786"}}',
    '{"thought": "ok", "action": "visit", "args": {"url": "https://example.com/a?x=1"}}',
    '{"action": "search", "args": {"query": "a, b, c"}}',
])
def test_repair_json_text_is_a_noop_on_well_formed_input(valid_json):
    """A valid call must never be rewritten into a different valid call — repair only ever fires
    on an actual textual defect, so on well-formed JSON it must report nothing to change."""
    assert repair_json_text(valid_json) is None


def test_repair_json_text_is_idempotent_on_its_own_output():
    broken = '{"action": "search", "args": {"query": "q"},}'
    once = repair_json_text(broken)
    assert once is not None
    twice = repair_json_text(once)
    assert twice is None  # nothing left to fix


def test_extract_decision_never_repairs_a_call_that_already_parses():
    """Repair must not run at all when the first pass already parses — a well-formed call is
    reported as direct/fenced/span, never as repaired, even though repair_json_text might
    technically be a no-op on it (this pins the ordering, not just the fixer safety)."""
    result = extract_decision('{"action": "search", "args": {"query": "q"}}')
    assert result.repaired is False
    assert result.repair_attempts == 0


# --------------------------------------------------------------------------- run_tool_loop


class _Recorder:
    def __init__(self):
        self.transcript = []  # list of ToolLoopStep
        self.timings = []

    def render_view(self):
        return "\n".join(f"{s.kind}:{s.raw_text}" for s in self.transcript)

    def on_step(self, step):
        self.transcript.append(step)

    def record_timing(self, *, name, started_at, success, payload=None, error=None):
        self.timings.append({"name": name, "success": success, "payload": payload, "error": error})


def _scripted_call_model(responses):
    responses = list(responses)
    prompts = []

    async def call_model(prompt_text):
        prompts.append(prompt_text)
        text = responses.pop(0) if responses else ""
        return text, {"input_tokens": 7, "output_tokens": 3}

    return call_model, prompts


def _run(responses, tools=None, dispatch_tool=None, turns=20, telemetry=None, **kw):
    tools = tools if tools is not None else [ToolSpec(name="search", arg_names=("query",))]
    rec = _Recorder()
    call_model, prompts = _scripted_call_model(responses)
    calls = []

    async def default_dispatch(name, args):
        calls.append((name, args))
        return f"observed {name}"

    dispatch_tool = dispatch_tool or default_dispatch

    error = None
    try:
        asyncio.run(run_tool_loop(
            tools=tools, render_view=rec.render_view, call_model=call_model,
            dispatch_tool=dispatch_tool, on_step=rec.on_step, turns=turns,
            telemetry=telemetry, **kw,
        ))
    except BaseException as exc:  # noqa: BLE001 — asserted on by the caller
        error = exc
    return rec, prompts, calls, error


def test_loop_dispatches_the_action_and_records_a_tool_call_step():
    rec, _prompts, calls, error = _run([
        '{"thought": "look", "action": "search", "args": {"query": "mont blanc"}}',
        '{"thought": "done", "action": "finish", "args": {"answer": "1786"}}',
    ])
    assert error is None
    assert calls == [("search", {"query": "mont blanc"})]
    kinds = [s.kind for s in rec.transcript]
    assert kinds == ["tool_call", "finish"]
    assert rec.transcript[0].observation == "observed search"
    assert rec.transcript[1].call.args == {"answer": "1786"}


def test_loop_never_dispatches_an_unknown_action():
    rec, _prompts, calls, error = _run([
        '{"thought": "hmm", "action": "teleport", "args": {}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ])
    assert error is None
    assert calls == []
    assert rec.transcript[0].kind == "invalid_action"


def test_loop_recovers_from_a_malformed_turn_via_a_nudge():
    rec, prompts, _calls, error = _run([
        "not json at all",
        '{"thought": "ok", "action": "finish", "args": {"answer": "recovered"}}',
    ])
    assert error is None
    assert rec.transcript[0].kind == "malformed_nudge"
    assert rec.transcript[1].kind == "finish"


def test_loop_gives_up_after_max_malformed_turns_without_raising():
    rec, _prompts, _calls, error = _run(["not json"] * 6, max_malformed_turns=3)
    assert error is None
    assert rec.transcript[-1].kind == "malformed_give_up"
    assert len(rec.transcript) == 3


def test_loop_raises_tool_loop_exhausted_when_turns_run_out():
    rec, _prompts, calls, error = _run(
        ['{"thought": "again", "action": "search", "args": {"query": "q%d"}}' % i
         for i in range(10)],
        turns=3,
    )
    assert isinstance(error, ToolLoopExhausted)
    assert len(calls) == 3
    assert len(rec.transcript) == 3


def test_loop_truncates_thought_to_thought_chars():
    rec, _prompts, _calls, error = _run([
        '{"thought": "%s", "action": "finish", "args": {"answer": "x"}}' % ("a" * 500),
    ], thought_chars=10)
    assert error is None
    assert len(rec.transcript[0].thought) == 10


def test_loop_coerces_non_dict_args_instead_of_crashing():
    rec, _prompts, calls, error = _run([
        '{"thought": "oops", "action": "search", "args": "mont blanc"}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ])
    assert error is None
    assert calls == [("search", {})]


def test_loop_recovers_a_repairable_malformed_turn_without_a_nudge():
    """A trailing-comma decision must parse via repair on the FIRST attempt — no nudge turn, no
    malformed step at all — since the whole point of the repair-before-nudge step is to avoid
    burning an extra model turn on a fixable defect."""
    rec, _prompts, calls, error = _run([
        '{"thought": "look", "action": "search", "args": {"query": "q"},}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ])
    assert error is None
    assert calls == [("search", {"query": "q"})]
    assert [s.kind for s in rec.transcript] == ["tool_call", "finish"]


# --------------------------------------------------------------------------- telemetry


def test_loop_records_one_timing_per_turn_with_the_expected_payload_shape():
    telemetry = _Recorder()
    rec, _prompts, _calls, error = _run([
        '{"thought": "look", "action": "search", "args": {"query": "q"}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], telemetry=telemetry)
    assert error is None
    assert len(telemetry.timings) == 2
    for t in telemetry.timings:
        assert t["name"] == "tool_call_emulation"
        payload = t["payload"]
        assert payload["transport"] == "prompted"
        assert "action" in payload
        assert payload["json_source"] in ("direct", "fenced", "span", "failed")
        assert "repaired" in payload
        assert "repair_attempts" in payload
        assert "invalid_action" in payload
    assert telemetry.timings[0]["payload"]["action"] == "search"
    assert telemetry.timings[1]["payload"]["action"] == "finish"


def test_loop_telemetry_flags_a_repaired_turn():
    telemetry = _Recorder()
    _rec, _prompts, _calls, error = _run([
        '{"thought": "look", "action": "search", "args": {"query": "q"},}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], telemetry=telemetry)
    assert error is None
    assert telemetry.timings[0]["payload"]["repaired"] is True
    assert telemetry.timings[0]["payload"]["repair_attempts"] >= 1


def test_loop_telemetry_flags_an_invalid_action():
    telemetry = _Recorder()
    _rec, _prompts, _calls, error = _run([
        '{"thought": "hmm", "action": "teleport", "args": {}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], telemetry=telemetry)
    assert error is None
    assert telemetry.timings[0]["payload"]["invalid_action"] is True
    assert telemetry.timings[0]["success"] is False


def test_loop_with_no_telemetry_records_nothing_and_does_not_raise():
    rec, _prompts, _calls, error = _run([
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], telemetry=None)
    assert error is None  # no telemetry object supplied -> silently skipped


def test_telemetry_record_timing_failure_does_not_break_the_run():
    class _Boom:
        def record_timing(self, **kw):
            raise RuntimeError("boom")

    rec, _prompts, _calls, error = _run([
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], telemetry=_Boom())
    assert error is None
