"""
Unit tests for scripts/trace_read.py -- the missing TraceRecorder JSONL reader.

Nothing in the repo read a trace back before this: agent/app/trace_recorder.py only writes,
and the result-JSON summarizer deliberately strips raw prompt/completion text on its way into
the per-cell result. These tests write synthetic trace lines with TraceRecorder itself (the
real writer this reader must stay compatible with) and assert calls_from_trace pairs "in"/"out"
connector_io events by call_id, not by file order.

No network.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from agent.app.trace_recorder import TraceRecorder

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "trace_read.py"


def _load_trace_read_module():
    """Import scripts/trace_read.py by path (scripts/ is not a package on sys.path)."""
    spec = importlib.util.spec_from_file_location("trace_read", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["trace_read"] = module
    spec.loader.exec_module(module)
    return module


trace_read = _load_trace_read_module()


def _write_llm_call(
    tracer: TraceRecorder,
    call_id: int,
    *,
    stage=None,
    node_id=None,
    prompt_text="what is the capital of France",
    completion_text="Paris",
    temperature=0.3,
    seed=None,
    attempts=1,
    error=None,
):
    in_payload = {
        "connector": "ConnectorLLM",
        "direction": "in",
        "operation": "llm_query",
        "call_id": call_id,
        "payload": {
            "model": "qwen2.5:7b",
            "prompt_chars": len(prompt_text),
            "prompt_text": prompt_text,
            "temperature": temperature,
            "seed": seed,
        },
    }
    if stage:
        in_payload["stage"] = stage
    if node_id:
        in_payload["node_id"] = node_id
    tracer.record("connector_io", in_payload)

    out_payload = {
        "connector": "ConnectorLLM",
        "direction": "out",
        "operation": "llm_query",
        "call_id": call_id,
        "payload": {
            "model": "qwen2.5:7b",
            "completion_chars": len(completion_text) if completion_text else 0,
            "completion_text": completion_text,
            "attempts": attempts,
        },
    }
    if error:
        out_payload["error"] = error
    if stage:
        out_payload["stage"] = stage
    if node_id:
        out_payload["node_id"] = node_id
    tracer.record("connector_io", out_payload)


class TestLoadTrace:
    def test_loads_events_in_file_order(self, tmp_path):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        tracer.record("decision", {"stage": "expansion"})
        tracer.record("timing", {"name": "llm_call"})
        tracer.close()

        events = trace_read.load_trace(path)
        assert [e.event for e in events] == ["decision", "timing"]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text('{"ts": 1.0, "event": "a", "payload": {}}\n\n', encoding="utf-8")
        events = trace_read.load_trace(path)
        assert len(events) == 1

    def test_skips_corrupt_lines_without_raising(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(
            '{"ts": 1.0, "event": "a", "payload": {}}\n'
            'not valid json at all\n'
            '{"ts": 2.0, "event": "b", "payload": {}}\n',
            encoding="utf-8",
        )
        events = trace_read.load_trace(path)
        assert [e.event for e in events] == ["a", "b"]


class TestCallsFromTrace:
    def test_pairs_in_and_out_by_call_id(self, tmp_path):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        _write_llm_call(tracer, call_id=1, prompt_text="prompt one", completion_text="answer one")
        tracer.close()

        calls = trace_read.calls_from_trace(path)
        assert len(calls) == 1
        call = calls[0]
        assert call.call_id == 1
        assert call.prompt_text == "prompt one"
        assert call.completion_text == "answer one"
        assert call.paired is True

    def test_pairing_is_by_call_id_not_file_order(self, tmp_path):
        """Interleave two calls' events (as concurrency would produce) -- the reader must not
        assume the Nth "in" pairs with the Nth "out" in file order."""
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        # call 2's "in" is written before call 1's "out" -- pure file-order pairing would
        # wrongly pair call 1's "in" with call 2's "out".
        tracer.record("connector_io", {
            "connector": "ConnectorLLM", "direction": "in", "operation": "llm_query",
            "call_id": 1, "payload": {"prompt_text": "first prompt"},
        })
        tracer.record("connector_io", {
            "connector": "ConnectorLLM", "direction": "in", "operation": "llm_query",
            "call_id": 2, "payload": {"prompt_text": "second prompt"},
        })
        tracer.record("connector_io", {
            "connector": "ConnectorLLM", "direction": "out", "operation": "llm_query",
            "call_id": 2, "payload": {"completion_text": "second answer"},
        })
        tracer.record("connector_io", {
            "connector": "ConnectorLLM", "direction": "out", "operation": "llm_query",
            "call_id": 1, "payload": {"completion_text": "first answer"},
        })
        tracer.close()

        calls = {c.call_id: c for c in trace_read.calls_from_trace(path)}
        assert calls[1].prompt_text == "first prompt"
        assert calls[1].completion_text == "first answer"
        assert calls[2].prompt_text == "second prompt"
        assert calls[2].completion_text == "second answer"

    def test_multiple_calls_stay_distinct(self, tmp_path):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        _write_llm_call(tracer, call_id=1, prompt_text="p1", completion_text="c1")
        _write_llm_call(tracer, call_id=2, prompt_text="p2", completion_text="c2")
        tracer.close()

        calls = trace_read.calls_from_trace(path)
        assert len(calls) == 2
        assert calls[0].prompt_text == "p1" and calls[0].completion_text == "c1"
        assert calls[1].prompt_text == "p2" and calls[1].completion_text == "c2"

    def test_stage_and_node_id_carried_onto_the_call(self, tmp_path):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        _write_llm_call(tracer, call_id=1, stage="expansion", node_id="n3")
        tracer.close()

        call = trace_read.calls_from_trace(path)[0]
        assert call.stage == "expansion"
        assert call.node_id == "n3"

    def test_attempts_and_seed_are_exposed(self, tmp_path):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        _write_llm_call(tracer, call_id=1, seed=42, attempts=3)
        tracer.close()

        call = trace_read.calls_from_trace(path)[0]
        assert call.seed == 42
        assert call.attempts == 3

    def test_error_only_call_is_unpaired_but_still_surfaced(self, tmp_path):
        """A terminal failure still writes an "in" + an "out" with an error -- paired() must
        stay True since both events exist, and the error text must be exposed."""
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        _write_llm_call(tracer, call_id=1, completion_text=None, error="status 402")
        tracer.close()

        call = trace_read.calls_from_trace(path)[0]
        assert call.error == "status 402"
        assert call.paired is True

    def test_non_connector_io_events_are_ignored(self, tmp_path):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        tracer.record("decision", {"stage": "expansion"})
        _write_llm_call(tracer, call_id=1)
        tracer.record("summary", {"mandate": "x"})
        tracer.close()

        calls = trace_read.calls_from_trace(path)
        assert len(calls) == 1

    def test_legacy_trace_without_call_id_degrades_to_unpaired_rows(self, tmp_path):
        """A trace written before call_id existed has no way to correlate in/out; each event
        must still surface as its own row instead of crashing or silently merging."""
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        tracer.record("connector_io", {
            "connector": "ConnectorLLM", "direction": "in", "operation": "llm_query",
            "payload": {"prompt_text": "legacy prompt"},
        })
        tracer.record("connector_io", {
            "connector": "ConnectorLLM", "direction": "out", "operation": "llm_query",
            "payload": {"completion_text": "legacy answer"},
        })
        tracer.close()

        calls = trace_read.calls_from_trace(path)
        assert len(calls) == 2
        assert not any(isinstance(c.call_id, int) for c in calls)

    def test_duration_is_none_when_unpaired(self, tmp_path):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        tracer.record("connector_io", {
            "connector": "ConnectorLLM", "direction": "in", "operation": "llm_query",
            "call_id": 1, "payload": {"prompt_text": "orphan"},
        })
        tracer.close()

        call = trace_read.calls_from_trace(path)[0]
        assert call.duration is None
        assert call.paired is False

    def test_no_calls_in_an_empty_trace(self, tmp_path):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        tracer.record("summary", {"mandate": "x"})
        tracer.close()

        assert trace_read.calls_from_trace(path) == []


class TestCLIMain:
    def test_main_prints_a_row_per_call(self, tmp_path, capsys):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        _write_llm_call(tracer, call_id=1)
        tracer.close()

        rc = trace_read.main([str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "call_id=1" in out

    def test_main_json_dumps_full_calls(self, tmp_path, capsys):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        _write_llm_call(tracer, call_id=1, prompt_text="dump me")
        tracer.close()

        rc = trace_read.main([str(path), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "dump me" in out

    def test_main_handles_empty_trace(self, tmp_path, capsys):
        path = tmp_path / "t.jsonl"
        tracer = TraceRecorder(path)
        tracer.record("summary", {})
        tracer.close()

        rc = trace_read.main([str(path)])
        assert rc == 0
        assert "No connector_io calls" in capsys.readouterr().out


@pytest.mark.skipif(not Path("agent/idea_test_results").exists(), reason="repo layout only")
def test_reader_survives_a_real_retained_trace_from_this_repo():
    """Smoke test against a real, already-on-disk trace (no LLM I/O capture assumed) -- the
    reader must not choke on production trace shapes even when call_id/stage/node_id are
    absent (every trace on disk predates this lane's change)."""
    results_dir = Path("agent/idea_test_results")
    sample = next(iter(results_dir.glob("*.jsonl")), None)
    if sample is None:
        pytest.skip("no retained trace files present")
    calls = trace_read.calls_from_trace(sample)
    events = trace_read.load_trace(sample)
    assert isinstance(events, list)
    assert isinstance(calls, list)
