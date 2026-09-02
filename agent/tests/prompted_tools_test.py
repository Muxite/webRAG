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


# ------------------------------------------------------------------- kv-line form (Task 2.1)


def test_extract_decision_repairs_action_equals_args_equals_kv_line():
    result = extract_decision('action=search args={"query": "mont blanc"}')
    assert result.value == {"action": "search", "args": {"query": "mont blanc"}}
    assert result.repaired is True


def test_extract_decision_repairs_action_colon_prefix_with_kv_slot_lines():
    result = extract_decision("ACTION: visit\nurl: https://en.wikipedia.org/wiki/Mont_Blanc")
    assert result.value["action"] == "visit"
    assert result.value["args"]["url"] == "https://en.wikipedia.org/wiki/Mont_Blanc"
    assert result.repaired is True


def test_extract_decision_repairs_bare_action_equals_multiline_kv():
    result = extract_decision("action=search\nquery=mont blanc")
    assert result.value == {"action": "search", "args": {"query": "mont blanc"}}
    assert result.repaired is True


# --------------------------------------------------------------- curly quotes (Task 2.3)


def test_extract_decision_repairs_curly_quotes_in_a_full_object():
    raw = "{“action”: “search”, “args”: {“query”: “q”}}"
    result = extract_decision(raw)
    assert result.value == {"action": "search", "args": {"query": "q"}}


def test_extract_decision_repairs_curly_single_quotes():
    raw = "{‘action’: ‘search’, ‘args’: {‘query’: ‘q’}}"
    result = extract_decision(raw)
    assert result.value == {"action": "search", "args": {"query": "q"}}


def test_fix_curly_quotes_is_a_noop_when_ascii_quotes_are_already_present():
    """Regression pin: a completion that is properly ASCII-quoted but merely CONTAINS a curly
    quote as ordinary punctuation inside a string value (e.g. a quoted sentence using “smart
    quotes”) must be left untouched by the curly-quote fixer — translating those unconditionally
    would splice a legitimate open string, turning a recoverable unterminated-string defect into
    unrecoverable garbage. Caught live against corpus record 070/tinyllama during development;
    this pins it permanently."""
    text = '{"thought": "“Though” is used like “even though”"}'
    assert repair_json_text(text) is None  # already valid JSON — the curly quotes are content
    result = extract_decision(text)
    assert result.value == {"thought": "“Though” is used like “even though”"}
    assert result.repaired is False


def test_fix_curly_quotes_does_not_corrupt_an_unterminated_string_with_curly_content():
    """The failure mode that motivated the guard above: an ASCII-opened string that never closes
    (truncated completion) and happens to contain curly-quote punctuation. Unconditional
    translation would turn the curly quotes into fresh ASCII string delimiters and destroy the
    span the unterminated-string fixer would otherwise recover cleanly."""
    text = '{"thought": "“Though” is a preposition, and it doesn\'t add value. Use "'
    result = extract_decision(text)
    assert result.value is not None
    assert result.value.get("thought", "").startswith("“Though”")


# --------------------------------------------------------------- python literals (Task 2.4)


def test_extract_decision_repairs_python_true_false_none():
    result = extract_decision(
        '{"action": "search", "args": {"query": "q", "strict": True, "cursor": None}}'
    )
    assert result.value == {
        "action": "search", "args": {"query": "q", "strict": True, "cursor": None},
    }
    assert result.repaired is True


def test_extract_decision_python_literal_fix_does_not_touch_string_contents():
    """A literal string VALUE containing the word True must not be corrupted."""
    result = extract_decision('{"action": "search", "args": {"query": "True Grit movie"}}')
    assert result.value == {"action": "search", "args": {"query": "True Grit movie"}}
    assert result.repaired is False  # already valid JSON — nothing to repair


# --------------------------------------------------------------- tool_call wrapper (Task 2.5)


def test_extract_decision_strips_tool_call_wrapper_tags():
    raw = '<tool_call>{"action": "search", "args": {"query": "q"}}</tool_call>'
    result = extract_decision(raw)
    assert result.value == {"action": "search", "args": {"query": "q"}}


def test_extract_decision_strips_special_token_wrappers():
    raw = '<|start_header_id|>assistant<|end_header_id|>\n{"action": "search", "args": {}}'
    result = extract_decision(raw)
    assert result.value == {"action": "search", "args": {}}


def test_wrapper_tag_stripping_does_not_corrupt_a_legitimate_value_mentioning_the_tag():
    """Regression pin: a query that genuinely asks ABOUT `<tool_call>`/`<|...|>` syntax must
    survive completely intact — stripping is only ever safe OUTSIDE a quoted JSON string."""
    raw = '{"action": "search", "args": {"query": "what is a <tool_call> tag in llama.cpp"}}'
    result = extract_decision(raw)
    assert result.value == {
        "action": "search", "args": {"query": "what is a <tool_call> tag in llama.cpp"},
    }
    assert result.repaired is False


# --------------------------------------------------------------- unterminated fence (Task 2.6)


def test_extract_decision_recovers_an_unterminated_code_fence():
    raw = '```json\n{"action": "search", "args": {"query": "q"}}'
    result = extract_decision(raw)
    assert result.value == {"action": "search", "args": {"query": "q"}}
    assert result.source == "fenced"


# --------------------------------------------------------------- trailing comma string-safety


def test_fix_trailing_comma_does_not_touch_a_quoted_comma_brace_sequence():
    """A literal `", }"` inside a quoted string value must survive untouched — the old
    regex-based fixer was not string-aware and would have rewritten it."""
    text = '{"action": "search", "args": {"query": "wait, }"}}'
    assert repair_json_text(text) is None  # already valid JSON, nothing to fix
    result = extract_decision(text)
    assert result.value == {"action": "search", "args": {"query": "wait, }"}}
    assert result.repaired is False


# --------------------------------------------------------------------------- repair idempotency


@pytest.mark.parametrize("valid_json", [
    '{"action": "search", "args": {"query": "mont blanc"}}',
    '{"action": "finish", "args": {"answer": "1786"}}',
    '{"thought": "ok", "action": "visit", "args": {"url": "https://example.com/a?x=1"}}',
    '{"action": "search", "args": {"query": "a, b, c"}}',
    '{"action": "search", "args": {"strict": true, "cursor": null, "ok": false}}',
    '{"action": "search", "args": {"query": "True Grit movie"}}',
    '{"action": "search", "args": {"query": "wait, }"}}',
    # a value that itself looks like a kv-line ("action=...") must not trip the kv-line fixer —
    # it only ever inspects the FIRST line of the raw text, and this text's first line is `{`.
    '{"action": "search", "args": {"query": "action=search args={\\"x\\": 1}"}}',
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


# --------------------------------------------------------------------------- inline argument (Task 2.2)


def test_loop_splits_an_inline_argument_from_the_action_field():
    rec, _prompts, calls, error = _run([
        '{"thought": "x", "action": "visit https://en.wikipedia.org/wiki/X", "args": {}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], tools=[ToolSpec(name="visit", arg_names=("url",))])
    assert error is None
    assert calls == [("visit", {"url": "https://en.wikipedia.org/wiki/X"})]
    assert rec.transcript[0].kind == "tool_call"


def test_loop_does_not_split_inline_argument_when_tool_has_multiple_slots():
    """Conservative: only maps the remainder when the named tool has EXACTLY one arg slot — a
    multi-slot tool is too ambiguous to guess which slot the trailing text belongs to."""
    rec, _prompts, calls, error = _run([
        '{"thought": "x", "action": "search extra text", "args": {}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], tools=[ToolSpec(name="search", arg_names=("query", "limit"))])
    assert error is None
    assert calls == []
    assert rec.transcript[0].kind == "invalid_action"


def test_loop_inline_argument_split_does_not_override_an_explicit_arg():
    rec, _prompts, calls, error = _run([
        '{"thought": "x", "action": "visit ignored-text", "args": {"url": "https://real"}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], tools=[ToolSpec(name="visit", arg_names=("url",))])
    assert error is None
    assert calls == [("visit", {"url": "https://real"})]


# --------------------------------------------------------------------------- bounded invalid-action loop (Task 3)


def test_loop_gives_up_after_max_invalid_actions_without_raising():
    rec, _prompts, calls, error = _run(
        ['{"thought": "x", "action": "teleport", "args": {}}'] * 6,
        max_invalid_actions=3,
    )
    assert error is None
    assert calls == []
    assert rec.transcript[-1].kind == "invalid_action_give_up"
    assert len(rec.transcript) == 3


def test_loop_invalid_action_streak_resets_on_a_successful_tool_call():
    rec, _prompts, calls, error = _run([
        '{"thought": "x", "action": "teleport", "args": {}}',
        '{"thought": "x", "action": "teleport", "args": {}}',
        '{"thought": "look", "action": "search", "args": {"query": "q"}}',
        '{"thought": "x", "action": "teleport", "args": {}}',
        '{"thought": "x", "action": "teleport", "args": {}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], max_invalid_actions=3)
    assert error is None
    assert calls == [("search", {"query": "q"})]
    assert rec.transcript[-1].kind == "finish"


def test_loop_fuzzy_matches_a_plural_action_name():
    rec, _prompts, calls, error = _run([
        '{"thought": "x", "action": "searches", "args": {"query": "q"}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ])
    assert error is None
    assert calls == [("search", {"query": "q"})]
    assert rec.transcript[0].kind == "tool_call"


def test_loop_fuzzy_matches_a_web_prefixed_alias():
    rec, _prompts, calls, error = _run([
        '{"thought": "x", "action": "search_web", "args": {"query": "q"}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ])
    assert error is None
    assert calls == [("search", {"query": "q"})]
    assert rec.transcript[0].kind == "tool_call"


def test_loop_fuzzy_match_does_not_guess_when_ambiguous():
    """`search`-or-`visit` style ambiguity must never be silently resolved — an action name
    that doesn't clearly reduce to exactly one known tool still counts as invalid."""
    rec, _prompts, calls, error = _run([
        '{"thought": "x", "action": "lookup", "args": {}}',
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], tools=[ToolSpec(name="search", arg_names=("query",)), ToolSpec(name="visit", arg_names=("url",))])
    assert error is None
    assert calls == []
    assert rec.transcript[0].kind == "invalid_action"


# --------------------------------------------------------------------------- json_telemetry hook (Task 4)


def test_loop_calls_json_telemetry_hook_once_per_turn_with_raw_text_and_parsed_ok():
    calls = []

    def hook(raw_text, parsed_ok):
        calls.append((raw_text, parsed_ok))

    rec, _prompts, _calls, error = _run([
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], json_telemetry_hook=hook)
    assert error is None
    assert calls == [('{"thought": "ok", "action": "finish", "args": {"answer": "done"}}', True)]


def test_loop_json_telemetry_hook_sees_malformed_turns_as_not_parsed():
    calls = []

    def hook(raw_text, parsed_ok):
        calls.append((raw_text, parsed_ok))

    rec, _prompts, _calls, error = _run([
        "not json at all",
        '{"thought": "ok", "action": "finish", "args": {"answer": "recovered"}}',
    ], json_telemetry_hook=hook)
    assert error is None
    assert calls[0] == ("not json at all", False)
    assert calls[1][1] is True


def test_loop_json_telemetry_hook_failure_does_not_break_the_run():
    def hook(raw_text, parsed_ok):
        raise RuntimeError("boom")

    rec, _prompts, _calls, error = _run([
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ], json_telemetry_hook=hook)
    assert error is None


def test_loop_with_no_json_telemetry_hook_does_not_raise():
    rec, _prompts, _calls, error = _run([
        '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
    ])
    assert error is None


# ------------------------------------------------------- tolerant finish-argument reading


def _finish_loop(decision_json: str):
    """Run one turn that finishes, returning the ToolCall the loop produced."""
    steps = []

    async def call_model(_prompt):
        return decision_json, None

    async def dispatch(_name, _args):  # pragma: no cover - finish never dispatches
        raise AssertionError("finish must not dispatch a tool")

    asyncio.run(run_tool_loop(
        tools=[ToolSpec(name="finish", description="submit", arg_names=("answer",))],
        render_view=lambda: "view", call_model=call_model, dispatch_tool=dispatch,
        on_step=steps.append, turns=2,
    ))
    return steps[-1]


def test_a_finish_that_names_its_answer_slots_differently_is_not_silently_discarded():
    """Measured on phi3:mini: it emits 98% valid JSON, does the work, and finishes with
    ``args: {"answer1": ..., "answer2": ...}``. The loop read only ``args["answer"]``, so the
    submission was dropped and the cell scored zero -- the model was penalised for naming a slot,
    not for being wrong. Every plausible answer slot is read, in sorted key order, so a two-part
    answer survives intact."""
    step = _finish_loop('{"action": "finish", "args": {"answer1": "419.7 m", "answer2": "330 m"}}')

    assert step.kind == "finish"
    assert "419.7 m" in step.call.args["answer"]
    assert "330 m" in step.call.args["answer"]


def test_a_plain_answer_slot_still_wins_over_any_alias():
    """An explicit ``answer`` is the model's own choice and must never be diluted by joining it
    with other keys that happen to be present."""
    step = _finish_loop(
        '{"action": "finish", "args": {"answer": "the real one", "note": "ignore me"}}')

    assert step.call.args["answer"] == "the real one"


def test_a_finish_with_no_answerish_slot_at_all_still_finishes_empty():
    """Absent is not invented: a finish carrying nothing answer-shaped submits an empty answer
    rather than scraping an unrelated field into one."""
    step = _finish_loop('{"action": "finish", "args": {"confidence": "high"}}')

    assert step.kind == "finish"
    assert step.call.args["answer"] == ""
