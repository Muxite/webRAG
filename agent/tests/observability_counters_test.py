"""Tests for the observability-counter fixes: search/llm call counts derived from
``telemetry_raw.timings`` (the one-entry-per-real-operation ledger) instead of the miscounted
sources they used to read, and ``prompted_tools.run_tool_loop``'s new distinction between a
turn whose tool ACTUALLY executed and one that only parsed.

See the handoff report for the full before/after numbers on real stored cells
(``agent/idea_test_results/phi3_both_210_*.json``, ``agent/idea_test_results/ledgerfinal01_*.json``).
"""
from __future__ import annotations

import asyncio
import glob
import json
import os

import pytest

from unittest.mock import MagicMock

from agent.app.testing.utils import summarize_observability
from agent.app.prompted_tools import (
    TOOL_ERROR_PREFIX, ToolSpec, run_tool_loop, tool_error_observation,
)
from agent.app.agent_io import AgentIO

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "idea_test_results")


class _EmptyTelemetry:
    """Minimal telemetry with the empty collections summarize_observability reads — same shape
    as ``observability_test.py``'s fixture."""

    events = []
    llm_usage = []
    chroma_stored = []
    chroma_retrieved = []
    documents_seen = []
    timings = []
    decisions = []


class _Telemetry(_EmptyTelemetry):
    def __init__(self, *, timings=None, events=None, documents_seen=None):
        self.timings = timings or []
        self.events = events or []
        self.documents_seen = documents_seen or []


def _search_call_timing(success=True):
    return {"name": "search", "duration": 0.1, "success": success,
            "payload": {"query": "q", "result_count": 6}}


def _search_result_doc():
    return {"source": "search", "document": {"title": "t", "url": "u", "description": "d"}}


def _llm_call_timing(success=True):
    return {"name": "llm_call", "duration": 0.1, "success": success, "payload": {"model": "m"}}


def _connector_io(direction, connector="ConnectorLLM"):
    return {"event": "connector_io",
            "payload": {"connector": connector, "direction": direction, "payload": {}}}


# --------------------------------------------------------------------------- search.count


def test_search_count_is_the_real_call_count_not_documents_seen():
    """Bug: the old ``search["count"]`` was the number of RESULT DOCUMENTS seen across every
    search call (one per result item), not the number of calls. 2 real search calls, each
    returning 6 results, must report ``count == 2``, not 12."""
    telemetry = _Telemetry(
        timings=[_search_call_timing(), _search_call_timing()],
        documents_seen=[_search_result_doc() for _ in range(12)],
    )
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["search"]["count"] == 2


def test_search_documents_seen_preserves_the_old_value_under_its_own_honest_name():
    telemetry = _Telemetry(
        timings=[_search_call_timing(), _search_call_timing()],
        documents_seen=[_search_result_doc() for _ in range(12)],
    )
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["search"]["documents_seen"]["count"] == 12


def test_search_count_zero_when_no_search_timings_is_a_real_zero_not_unknown():
    """Absent must never silently become zero for something UNCOMPUTABLE -- but a run that truly
    performed no search calls (no "search" timings in the ground-truth ledger at all) is a
    legitimate, computable 0, not an unknown."""
    telemetry = _Telemetry(timings=[])
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["search"]["count"] == 0


def test_search_count_includes_a_failed_search_call():
    """A call that happened but errored is still a real call -- the count is calls made, not
    calls that returned results."""
    telemetry = _Telemetry(timings=[_search_call_timing(success=False)])
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["search"]["count"] == 1


def test_search_count_zero_when_dispatch_never_reached_the_connector():
    """The reported bug's third case: a schema/validation error inside ``_invoke_tool`` never
    reaches the search connector at all, so no "search" timing is ever recorded -- the honest
    count is 0, even though the model's turns all named ``action: search``."""
    telemetry = _Telemetry(timings=[], documents_seen=[])
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["search"]["count"] == 0
    assert obs["search"]["documents_seen"]["count"] == 0


# --------------------------------------------------------------------------- llm.calls


def test_llm_calls_counts_only_the_out_direction_not_both_of_each_matched_pair():
    """Bug: the old ``llm["calls"]`` counted every ``connector_io`` EVENT tagged
    ``connector == "ConnectorLLM"`` -- but a real call logs a matched (in, out) PAIR of those,
    so the old count was 2x the real number of calls. 3 real calls -> 3, not 6. Present but
    IRRELEVANT "llm_call" timings must not change the outcome -- ``llm.calls`` no longer reads
    ``telemetry.timings`` at all (see the comment above this field in ``summarize_observability``
    for why: LangGraph's native tool-calling transport records no such timing at all, which
    would read as a false 0 rather than the real 3)."""
    telemetry = _Telemetry(
        timings=[_llm_call_timing(), _llm_call_timing(), _llm_call_timing()],
        events=[_connector_io("in"), _connector_io("out")] * 3,
    )
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["llm"]["calls"] == 3


def test_llm_calls_is_correct_for_a_transport_with_no_llm_call_timing_at_all():
    """The gap the timings-based approach would have hit: LangGraph's NATIVE tool-calling
    transport never records an "llm_call" (that's AgentIO/ConnectorLLM-only) or a
    "tool_call_emulation" (that's the OTHER LangGraph transport) timing -- its only per-call
    observability signal is the synthesized connector_io parity pair. 4 real calls, zero
    timings of any kind, must still report 4, not 0."""
    telemetry = _Telemetry(
        timings=[],
        events=[_connector_io("in"), _connector_io("out")] * 4,
    )
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["llm"]["calls"] == 4


def test_llm_calls_counts_only_the_out_direction_of_each_matched_pair():
    """Both LangGraph transports (native tool-calling AND the emulated prompted-JSON loop) never
    touch ConnectorLLM directly -- their only per-call observability signal is
    ``langgraph_solver._record_io_parity``'s synthesized (in, out) pair, one pair per real call.
    5 real calls -> 5 "in" + 5 "out" events -> 5, not 10."""
    telemetry = _Telemetry(
        events=[_connector_io("in"), _connector_io("out")] * 5,
    )
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["llm"]["calls"] == 5


def test_llm_calls_is_zero_not_unknown_when_no_llm_calls_happened_at_all():
    telemetry = _Telemetry(events=[])
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["llm"]["calls"] == 0


def test_llm_calls_char_word_totals_are_unaffected_the_bug_was_only_the_count():
    """Each real call logs exactly one "in" event (prompt_chars/words) and one "out" event
    (completion_chars/words) -- summing those keys across every connector_io event was already
    additive, not doubled, so char/word totals must be unchanged by this fix."""
    events = []
    for _ in range(2):
        events.append({"event": "connector_io", "payload": {
            "connector": "ConnectorLLM", "direction": "in",
            "payload": {"prompt_chars": 100, "prompt_words": 20},
        }})
        events.append({"event": "connector_io", "payload": {
            "connector": "ConnectorLLM", "direction": "out",
            "payload": {"completion_chars": 50, "completion_words": 10},
        }})
    telemetry = _Telemetry(timings=[_llm_call_timing(), _llm_call_timing()], events=events)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["llm"]["prompt"]["chars"] == 200
    assert obs["llm"]["completion"]["chars"] == 100


# --------------------------------------------------------------------------- real stored cells


def _load_execution(path):
    d = json.load(open(path))
    return d.get("execution") or d


def _telemetry_from_raw(execution):
    raw = execution.get("telemetry_raw") or {}
    return _Telemetry(
        timings=raw.get("timings") or [],
        events=raw.get("events") or [],
        documents_seen=raw.get("documents_seen") or [],
    )


def _fixture(pattern):
    matches = glob.glob(os.path.join(_RESULTS_DIR, pattern))
    if not matches:
        pytest.skip(f"fixture not present: {pattern}")
    return matches[0]


def test_phi3_both_210_recomputes_to_the_real_35_llm_calls_not_70():
    """The bug report's primary fixture: 35 real ``tool_call_emulation`` turns, the OLD stored
    result reported ``llm.calls: 70`` (exactly 2x, from the synthesized connector_io in/out
    pair)."""
    path = _fixture("phi3_both_210_phi3:mini_langgraph_react_*.json")
    execution = _load_execution(path)
    old_calls = (execution.get("observability") or {}).get("llm", {}).get("calls")
    assert old_calls == 70, "fixture drifted from what this test documents"
    telemetry = _telemetry_from_raw(execution)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["llm"]["calls"] == 35


def test_phi3_both_210_recomputes_search_count_to_zero_no_search_ever_executed():
    """The same fixture's third bug case: every one of the 35 turns named ``action: search``,
    every one was recorded ``success: True``, but zero real search timings exist -- the honest
    call count is 0."""
    path = _fixture("phi3_both_210_phi3:mini_langgraph_react_*.json")
    execution = _load_execution(path)
    telemetry = _telemetry_from_raw(execution)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["search"]["count"] == 0


def test_ledgerfinal01_search_count_recomputes_off_the_old_six_x_inflation():
    """A ledgerfinal01 cell whose stored ``search["count"]`` (12) is exactly 6x its 2 real
    ``search`` timing entries (6 result documents per call) -- the pattern the report verified
    across every ledgerfinal01 cell checked."""
    path = _fixture("ledgerfinal01_230_qwen2.5:7b_sequential_react_extract_*.json")
    execution = _load_execution(path)
    old_count = (execution.get("observability") or {}).get("search", {}).get("count")
    assert old_count == 12, "fixture drifted from what this test documents"
    telemetry = _telemetry_from_raw(execution)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["search"]["count"] == 2


def test_ledgerfinal01_sequential_react_extract_llm_calls_matches_llm_usage_ground_truth():
    """This cell's ``AgentIO``/``ConnectorLLM`` path hit a SEPARATE bug (fixed at its source in
    ``agent_io.py``): ``AgentIO.query_llm`` used to record its own redundant ``"llm_call"``
    timing on top of ``ConnectorLLM``'s own, doubling the stored ``telemetry_raw.timings``
    "llm_call" count (9 real calls -> 18 timings) for cells captured before that fix. The
    ``connector_io`` "out" events this field actually reads are NOT affected by that bug
    (``ConnectorLLM`` logs those itself, once per call) so recomputing here must land on the
    real count regardless of when the cell was captured."""
    path = _fixture("ledgerfinal01_230_qwen2.5:7b_sequential_react_extract_*.json")
    execution = _load_execution(path)
    old_calls = (execution.get("observability") or {}).get("llm", {}).get("calls")
    assert old_calls == 18, "fixture drifted from what this test documents"
    telemetry = _telemetry_from_raw(execution)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["llm"]["calls"] == 9
    assert obs["llm"]["calls"] == len(json.load(open(path)).get("execution", {})
                                        .get("telemetry_raw", {}).get("llm_usage") or [])


def test_ledgerfinal01_langgraph_react_llm_calls_matches_llm_usage_ground_truth():
    """LangGraph's NATIVE tool-calling arm: no "llm_call"/"tool_call_emulation" timing exists
    for it at all, so this is the case that would have silently read 0 under a timings-only
    derivation instead of the real 12."""
    path = _fixture("ledgerfinal01_226_qwen2.5:7b_langgraph_react_*.json")
    execution = _load_execution(path)
    old_calls = (execution.get("observability") or {}).get("llm", {}).get("calls")
    assert old_calls == 24, "fixture drifted from what this test documents"
    telemetry = _telemetry_from_raw(execution)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["llm"]["calls"] == 12
    assert not any(t.get("name") in ("llm_call", "tool_call_emulation") for t in telemetry.timings)


# --------------------------------------------------------------------------- prompted_tools: B


def _scripted_call_model(responses):
    responses = list(responses)

    async def call_model(prompt_text):
        text = responses.pop(0) if responses else ""
        return text, {"input_tokens": 1, "output_tokens": 1}

    return call_model


class _Recorder:
    def __init__(self):
        self.transcript = []
        self.timings = []

    def render_view(self):
        return "view"

    def on_step(self, step):
        self.transcript.append(step)

    def record_timing(self, *, name, started_at, success, payload=None, error=None):
        self.timings.append({"name": name, "success": success, "payload": payload})


def _run(responses, dispatch_tool, turns=20, telemetry=None, **kw):
    tools = [ToolSpec(name="search", arg_names=("query",))]
    rec = _Recorder()
    call_model = _scripted_call_model(responses)
    error = None
    try:
        asyncio.run(run_tool_loop(
            tools=tools, render_view=rec.render_view, call_model=call_model,
            dispatch_tool=dispatch_tool, on_step=rec.on_step, turns=turns,
            telemetry=telemetry, **kw,
        ))
    except BaseException as exc:  # noqa: BLE001
        error = exc
    return rec, error


def test_a_string_observation_starting_with_the_tool_error_sentinel_is_not_a_success():
    """Bug: a dispatch that raised inside the tool (langgraph_solver's ``_invoke_tool`` turns
    ANY exception into a ``TOOL ERROR: ...`` string) was recorded ``success=True`` because an
    action merely PARSED. The legacy str-returning ``DispatchTool`` contract must be read for
    this sentinel."""
    async def dispatch(name, args):
        return f"{TOOL_ERROR_PREFIX} ValidationError: bad schema"

    telemetry = _Recorder()
    rec, error = _run(
        ['{"thought": "x", "action": "search", "args": {"query": "q"}}'],
        dispatch, telemetry=telemetry, max_tool_errors=5,
    )
    assert error is None
    assert rec.transcript[0].executed is False
    assert telemetry.timings[0]["success"] is False
    assert telemetry.timings[0]["payload"]["executed"] is False


def test_a_successful_string_observation_is_still_executed_true():
    async def dispatch(name, args):
        return "3 results found"

    telemetry = _Recorder()
    rec, error = _run(
        ['{"thought": "x", "action": "search", "args": {"query": "q"}}',
         '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}'],
        dispatch, telemetry=telemetry,
    )
    assert error is None
    assert rec.transcript[0].executed is True
    assert telemetry.timings[0]["success"] is True
    assert telemetry.timings[0]["payload"]["executed"] is True


def test_the_explicit_tuple_contract_is_honored_over_sentinel_sniffing():
    """The preferred seam: a dispatcher that can tell success from failure itself returns
    ``(text, executed)`` directly, with no reliance on prefix text."""
    async def dispatch(name, args):
        return "looks fine but actually failed", False

    rec, error = _run(
        ['{"thought": "x", "action": "search", "args": {"query": "q"}}'],
        dispatch, max_tool_errors=5,
    )
    assert error is None
    assert rec.transcript[0].executed is False
    assert rec.transcript[0].kind == "tool_call"  # under the streak limit -> caller keeps looping


def test_not_parsed_vs_executed_vs_tool_errored_are_three_distinguishable_cases():
    """The three cases the task requires be distinguishable: not-parsed (malformed),
    parsed-but-tool-errored, parsed-and-executed."""
    async def dispatch(name, args):
        if args.get("query") == "bad":
            return f"{TOOL_ERROR_PREFIX} boom", False
        return "ok", True

    telemetry = _Recorder()
    rec, error = _run(
        [
            "not json at all",
            '{"thought": "x", "action": "search", "args": {"query": "bad"}}',
            '{"thought": "x", "action": "search", "args": {"query": "good"}}',
            '{"thought": "ok", "action": "finish", "args": {"answer": "done"}}',
        ],
        dispatch, telemetry=telemetry, max_tool_errors=5,
    )
    assert error is None
    kinds = [(s.kind, s.executed) for s in rec.transcript]
    assert kinds == [
        ("malformed_nudge", None),
        ("tool_call", False),
        ("tool_call", True),
        ("finish", None),
    ]


# --------------------------------------------------------------------------- prompted_tools: C


def test_repeated_tool_errors_are_bounded_and_the_loop_does_not_raise():
    async def dispatch(name, args):
        return f"{TOOL_ERROR_PREFIX} same schema error every time", False

    rec, error = _run(
        ['{"thought": "x", "action": "search", "args": {"query": "q"}}'] * 10,
        dispatch, max_tool_errors=3,
    )
    assert error is None  # gives up cleanly, does not raise ToolLoopExhausted
    assert len(rec.transcript) == 3
    assert rec.transcript[-1].kind == "tool_error_give_up"
    assert rec.transcript[-1].tool_error_count == 3


def test_tool_error_streak_resets_on_an_executed_call_between_failures():
    calls = ["fail", "ok", "fail", "fail", "fail"]

    async def dispatch(name, args):
        outcome = calls.pop(0)
        if outcome == "fail":
            return f"{TOOL_ERROR_PREFIX} boom", False
        return "fine", True

    rec, error = _run(
        ['{"thought": "x", "action": "search", "args": {"query": "q"}}'] * 5,
        dispatch, max_tool_errors=3,
    )
    assert error is None
    # fail(1), ok(reset), fail(1), fail(2), fail(3)->give up: never reaches streak 3 on the
    # first run because the executed call in between reset the counter.
    assert [s.tool_error_count for s in rec.transcript if s.executed is False] == [1, 1, 2, 3]
    assert rec.transcript[-1].kind == "tool_error_give_up"


def test_tool_error_observation_names_the_tool_the_failure_and_the_streak_not_just_raw_error():
    msg = tool_error_observation("search", "TOOL ERROR: ValidationError: bad schema", 2)
    assert "search" in msg
    assert "2" in msg
    assert "ValidationError" in msg  # the concrete detail survives, not swallowed


def test_the_model_sees_a_concrete_observation_not_the_bare_repeated_exception_text():
    async def dispatch(name, args):
        return f"{TOOL_ERROR_PREFIX} ValidationError: bad schema", False

    rec, error = _run(
        ['{"thought": "x", "action": "search", "args": {"query": "q"}}'],
        dispatch, max_tool_errors=5,
    )
    assert error is None
    observation = rec.transcript[0].observation
    assert observation != f"{TOOL_ERROR_PREFIX} ValidationError: bad schema"
    assert "did NOT run" in observation
    assert "attempt 1" in observation


# --------------------------------------------------------------------------- agent_io.py: root
# cause of the SEPARATE "llm_call" timing-doubling bug (does not affect llm.calls itself, which
# reads connector_io -- see summarize_observability's comment -- but does affect
# obs["timings"]["llm_call"]["count"] and anything else keyed off telemetry.timings by that name).


class _RecordingTelemetry:
    def __init__(self):
        self.timings = []

    def record_timing(self, *, name, started_at, success, payload=None, error=None):
        self.timings.append({"name": name, "success": success, "payload": payload, "error": error})


class _FakeConnectorLLM:
    """Stands in for ``ConnectorLLM.query_llm``'s own instrumentation contract: it records
    exactly ONE ``"llm_call"`` timing itself for every call it completes -- a real success, or an
    exception it catches internally (in which case it returns ``None`` rather than raising, the
    real connector's contract) -- and it is never told about a timeout `AgentIO` enforces
    externally via `asyncio.wait_for` (a `CancelledError` thrown into an in-flight call skips
    right past this, never reaching its own ``record_timing`` call, same as the real connector)."""

    def __init__(self, *, result="answer", hang=False):
        self._result = result
        self._hang = hang
        self._telemetry = None

    def set_telemetry(self, telemetry):
        self._telemetry = telemetry

    def get_model(self):
        return "m"

    async def query_llm(self, payload, model_name=None):
        if self._hang:
            await asyncio.sleep(10)
        if self._telemetry:
            self._telemetry.record_timing(
                name="llm_call", started_at=0.0, success=self._result is not None,
                payload={"model": "m"},
            )
        return self._result


def _make_agent_io(connector_llm, telemetry):
    io = AgentIO(
        connector_llm=connector_llm,
        connector_search=MagicMock(),
        connector_http=MagicMock(),
        connector_chroma=MagicMock(),
    )
    io.set_telemetry(telemetry)
    return io


def test_agent_io_query_llm_does_not_double_record_a_successful_call():
    telemetry = _RecordingTelemetry()
    io = _make_agent_io(_FakeConnectorLLM(result="answer"), telemetry)
    result = asyncio.run(io.query_llm({"model": "m"}))
    assert result == "answer"
    llm_call_timings = [t for t in telemetry.timings if t["name"] == "llm_call"]
    assert len(llm_call_timings) == 1  # only ConnectorLLM's own -- AgentIO must not add a 2nd


def test_agent_io_query_llm_does_not_double_record_a_connector_caught_failure():
    """``ConnectorLLM.query_llm`` returns ``None`` (not a raise) on a failure it catches itself,
    having already recorded its own timing -- AgentIO's wrapper must not add a second one just
    because the response was falsy."""
    telemetry = _RecordingTelemetry()
    io = _make_agent_io(_FakeConnectorLLM(result=None), telemetry)
    result = asyncio.run(io.query_llm({"model": "m"}))
    assert result is None
    llm_call_timings = [t for t in telemetry.timings if t["name"] == "llm_call"]
    assert len(llm_call_timings) == 1


def test_agent_io_query_llm_records_the_one_case_connector_llm_cannot_see():
    """``asyncio.wait_for``'s external timeout cancels the in-flight ``ConnectorLLM.query_llm``
    coroutine before it ever reaches its own ``record_timing`` call -- AgentIO is the only place
    that can observe this outcome, so it (and only it) must record here."""
    telemetry = _RecordingTelemetry()
    io = _make_agent_io(_FakeConnectorLLM(hang=True), telemetry)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(io.query_llm({"model": "m"}, timeout_seconds=0.01))
    llm_call_timings = [t for t in telemetry.timings if t["name"] == "llm_call"]
    assert len(llm_call_timings) == 1
    assert llm_call_timings[0]["success"] is False
