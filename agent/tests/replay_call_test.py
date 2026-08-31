"""Tests for scripts/replay_call.py -- no network, no model, no GPU.

Uses fake async query functions / connectors instead of hitting a real backend, so this suite
runs offline. Live-replay behavior against a real local Ollama server is out of scope for these
unit tests (it needs a running server + honors R5's no-paid-runs rule at the integration level,
not here).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import replay_call as rc  # noqa: E402
from trace_read import Call  # noqa: E402


def _write_trace(tmp_path: Path, lines) -> Path:
    path = tmp_path / "trace.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def _in_event(call_id, messages, **overrides):
    payload = {
        "model": "qwen2.5:7b", "prompt_chars": 10, "prompt_words": 2,
        "temperature": 0.7, "top_p": None, "seed": None, "num_ctx": 32768,
        "response_format_type": None, "messages": messages,
    }
    payload.update(overrides)
    return {
        "ts": 1.0, "event": "connector_io",
        "payload": {
            "call_id": call_id, "connector": "ConnectorLLM", "operation": "llm_query",
            "direction": "in", "stage": "extract", "node_id": "n1", "payload": payload,
        },
    }


def _out_event(call_id, completion_text, **overrides):
    payload = {"model": "qwen2.5:7b", "completion_chars": len(completion_text),
               "completion_words": len(completion_text.split()), "attempts": 1,
               "completion_text": completion_text}
    payload.update(overrides)
    return {
        "ts": 2.0, "event": "connector_io",
        "payload": {
            "call_id": call_id, "connector": "ConnectorLLM", "operation": "llm_query",
            "direction": "out", "stage": "extract", "node_id": "n1", "payload": payload,
        },
    }


MESSAGES = [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Say hi."}]


def _full_trace(tmp_path, call_id=1, seed=None):
    return _write_trace(tmp_path, [
        _in_event(call_id, MESSAGES, seed=seed),
        _out_event(call_id, "hi"),
    ])


# --------------------------------------------------------------------------------------
# find_call
# --------------------------------------------------------------------------------------

def test_find_call_locates_by_call_id_matched_as_string_or_int(tmp_path):
    path = _full_trace(tmp_path, call_id=7)
    call = rc.find_call(path, 7)
    assert call.call_id == 7
    call2 = rc.find_call(path, "7")
    assert call2.call_id == 7


def test_find_call_raises_for_unknown_id(tmp_path):
    path = _full_trace(tmp_path, call_id=7)
    with pytest.raises(ValueError):
        rc.find_call(path, 999)


# --------------------------------------------------------------------------------------
# reconstruct_messages
# --------------------------------------------------------------------------------------

def test_reconstruct_messages_returns_the_captured_messages(tmp_path):
    path = _full_trace(tmp_path)
    call = rc.find_call(path, 1)
    messages = rc.reconstruct_messages(call)
    assert messages == MESSAGES
    # must be a copy, not the same list object
    messages.append({"role": "user", "content": "extra"})
    assert rc.reconstruct_messages(call) == MESSAGES


def test_reconstruct_messages_raises_a_clear_error_without_full_capture(tmp_path):
    path = _write_trace(tmp_path, [
        _in_event(1, messages=None),
        _out_event(1, "hi"),
    ])
    call = rc.find_call(path, 1)
    with pytest.raises(rc.ReplayNotCapturedError, match="IDEA_TEST_CAPTURE_LLM_IO"):
        rc.reconstruct_messages(call)


# --------------------------------------------------------------------------------------
# apply_replacements
# --------------------------------------------------------------------------------------

def test_apply_replacements_substitutes_within_string_content_only():
    messages = [{"role": "system", "content": "keep"}, {"role": "user", "content": "old text"}]
    out = rc.apply_replacements(messages, [("old", "new")])
    assert out[0]["content"] == "keep"
    assert out[1]["content"] == "new text"
    # original untouched
    assert messages[1]["content"] == "old text"


def test_apply_replacements_applies_pairs_in_order():
    messages = [{"role": "user", "content": "A"}]
    out = rc.apply_replacements(messages, [("A", "B"), ("B", "C")])
    assert out[0]["content"] == "C"


# --------------------------------------------------------------------------------------
# build_replay_payload
# --------------------------------------------------------------------------------------

def _call(**overrides):
    base = dict(call_id=1, model="qwen2.5:7b", temperature=0.7, top_p=0.9, seed=None,
                in_payload={"max_tokens": 256, "response_format_type": "json_object"})
    base.update(overrides)
    return Call(**base)


def test_build_replay_payload_defaults_to_the_original_sampling_params():
    call = _call()
    payload = rc.build_replay_payload(call, MESSAGES)
    assert payload["model"] == "qwen2.5:7b"
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 256
    assert payload["response_format"] == {"type": "json_object"}
    assert "seed" not in payload


def test_build_replay_payload_overrides_win_over_the_original():
    call = _call(seed=None)
    payload = rc.build_replay_payload(
        call, MESSAGES, temperature=0.0, top_p=1.0, seed=42, max_tokens=10, model="other-model",
    )
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["seed"] == 42
    assert payload["max_tokens"] == 10
    assert payload["model"] == "other-model"


# --------------------------------------------------------------------------------------
# classify_regime -- the honest-limit surface
# --------------------------------------------------------------------------------------

def test_anthropic_is_flagged_seed_unsupported_regardless_of_seed():
    result = rc.classify_regime(rc.KIND_ANTHROPIC, seed=42, repeat=1)
    assert result["regime"] == rc.REGIME_SEED_UNSUPPORTED_PROVIDER
    assert result["reproducible"] is False
    assert any("no seed parameter" in w for w in result["warnings"])


def test_no_seed_single_run_is_flagged_inconclusive():
    result = rc.classify_regime(rc.KIND_OPENAI_COMPATIBLE, seed=None, repeat=1)
    assert result["regime"] == rc.REGIME_UNSEEDED_SINGLE_RUN
    assert result["reproducible"] is False
    assert any("proves nothing" in w for w in result["warnings"])


def test_no_seed_with_repeats_is_flagged_indicative_not_proof():
    result = rc.classify_regime(rc.KIND_OPENAI_COMPATIBLE, seed=None, repeat=5)
    assert result["regime"] == rc.REGIME_UNSEEDED_MULTI_RUN
    assert result["reproducible"] == "indicative"
    assert any("not proof" in w for w in result["warnings"])


def test_seeded_ollama_carries_the_cold_start_caveat():
    result = rc.classify_regime(rc.KIND_OLLAMA_NATIVE, seed=7, repeat=1)
    assert result["regime"] == rc.REGIME_SEEDED_DETERMINISTIC
    assert result["reproducible"] is True
    assert any("cold-start" in w.lower() or "first ollama call" in w.lower()
               for w in result["warnings"])


def test_seeded_openai_compatible_notes_best_effort_not_guaranteed():
    result = rc.classify_regime(rc.KIND_OPENAI_COMPATIBLE, seed=7, repeat=1)
    assert result["regime"] == rc.REGIME_SEEDED_DETERMINISTIC
    assert result["reproducible"] is True
    assert any("best-effort" in w for w in result["warnings"])


# --------------------------------------------------------------------------------------
# run_replay
# --------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_replay_issues_repeat_calls_and_collects_completions():
    calls = []

    async def fake_query(payload):
        calls.append(payload)
        return f"resp-{len(calls)}"

    result = await rc.run_replay(fake_query, {"messages": MESSAGES}, repeat=3)
    assert result["completions"] == ["resp-1", "resp-2", "resp-3"]
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_run_replay_warmup_call_is_not_counted_in_completions():
    calls = []

    async def fake_query(payload):
        calls.append(payload)
        return f"resp-{len(calls)}"

    result = await rc.run_replay(fake_query, {"messages": MESSAGES}, repeat=2, warmup=True)
    assert len(calls) == 3  # 1 warmup + 2 reported
    assert result["completions"] == ["resp-2", "resp-3"]


# --------------------------------------------------------------------------------------
# format_report
# --------------------------------------------------------------------------------------

def test_format_report_shows_original_next_to_each_replay_and_a_match_summary():
    call = _call(completion_text="hi")
    replay = {"payload": {"messages": MESSAGES}, "completions": ["hi", "bye"]}
    regime = rc.classify_regime(rc.KIND_OPENAI_COMPATIBLE, seed=None, repeat=2)
    text = rc.format_report(call, replay, regime)
    assert "ORIGINAL COMPLETION" in text
    assert "REPLAY #1" in text and "REPLAY #2" in text
    assert "MATCH original: True" in text
    assert "MATCH original: False" in text
    assert "SUMMARY: 1/2 replay(s) matched" in text
    assert "not proof" in text


# --------------------------------------------------------------------------------------
# main() end to end, with a fake connector injected
# --------------------------------------------------------------------------------------

class _FakeBackend:
    pass


class _FakeConnector:
    def __init__(self):
        self._backend = _FakeBackend()
        self.calls = []

    async def query_llm(self, payload, stage=None, **kw):
        self.calls.append(payload)
        return "replayed answer"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def test_main_end_to_end_with_a_replacement_and_json_output(tmp_path, monkeypatch, capsys):
    path = _full_trace(tmp_path, call_id=1, seed=None)
    fake = _FakeConnector()
    monkeypatch.setattr(rc, "_build_connector", lambda: fake)

    rc_exit = rc.main([str(path), "1", "--replace", "hi", "hello", "--repeat", "2", "--json"])
    assert rc_exit == 0
    out = json.loads(capsys.readouterr().out)
    assert out["call_id"] == 1
    assert out["replay_completions"] == ["replayed answer", "replayed answer"]
    assert out["original_completion"] == "hi"
    assert out["regime"]["regime"] == rc.REGIME_UNSEEDED_MULTI_RUN
    assert len(fake.calls) == 2
    assert fake.calls[0]["messages"][1]["content"] == "Say hello."


def test_main_reports_a_clear_error_without_full_capture(tmp_path, monkeypatch, capsys):
    path = _write_trace(tmp_path, [_in_event(1, messages=None), _out_event(1, "hi")])
    monkeypatch.setattr(rc, "_build_connector", lambda: _FakeConnector())
    exit_code = rc.main([str(path), "1"])
    assert exit_code == 1
    assert "IDEA_TEST_CAPTURE_LLM_IO" in capsys.readouterr().err


def test_main_unknown_call_id_reports_a_clear_error(tmp_path, monkeypatch, capsys):
    path = _full_trace(tmp_path, call_id=1)
    monkeypatch.setattr(rc, "_build_connector", lambda: _FakeConnector())
    exit_code = rc.main([str(path), "999"])
    assert exit_code == 1
    assert "999" in capsys.readouterr().err
