"""Native-GoT expansion emits bad-model-lab JSON telemetry — offline, no LLM.

``expand()`` is where a weak model most often fails: it must emit a structured
``{"candidates": [...]}`` plan. Without a ``json_telemetry`` call site here a live
lab run through the native engine produced ZERO telemetry, so "why did the model
fail" (fenced / malformed / truncated / prose / refusal / empty) was unanswerable.

Pinned here: the block is a hard no-op when ``IDEA_TEST_JSON_TELEMETRY`` is unset,
and when set it records ``phase="native_expansion"`` with ``parsed_ok`` reflecting
the RAW completion's JSON validity — before ``_parse_candidates``' repair fallback.
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.expansion import LlmExpansionPolicy

_VALID = json.dumps(
    {"candidates": [{"title": "Search cats", "action": "search", "details": {"query": "cats"}}]}
)
# Right idea, markdown-fenced: json.loads fails, the repair fallback still salvages it.
_FENCED = "```json\n" + _VALID + "\n```"


class _StubIO:
    """Minimal AgentIO stand-in: canned completion, no connector, no network."""

    def __init__(self, response: str):
        self._response = response

    def build_llm_payload(self, **kwargs):
        return dict(kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs):
        return self._response


async def _expand(response: str):
    policy = LlmExpansionPolicy(io=_StubIO(response), model_name="tiny-local-model")
    graph = IdeaDag(root_title="Find facts about cats", root_details={"mandate": "Find facts about cats"})
    return await policy.expand(graph, graph.root_id())


def _entries(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_expansion_writes_no_telemetry_when_flag_unset(tmp_path, monkeypatch):
    out = tmp_path / "telemetry.jsonl"
    monkeypatch.delenv("IDEA_TEST_JSON_TELEMETRY", raising=False)
    monkeypatch.setenv("IDEA_TEST_JSON_TELEMETRY_PATH", str(out))

    candidates = await _expand(_VALID)

    assert len(candidates) == 1                 # normal path untouched
    assert not out.exists()


@pytest.mark.asyncio
async def test_expansion_records_valid_json_when_flag_set(tmp_path, monkeypatch):
    out = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("IDEA_TEST_JSON_TELEMETRY", "1")
    monkeypatch.setenv("IDEA_TEST_JSON_TELEMETRY_PATH", str(out))

    candidates = await _expand(_VALID)

    assert len(candidates) == 1
    entries = _entries(out)
    assert len(entries) == 1
    assert entries[0]["phase"] == "native_expansion"
    assert entries[0]["model"] == "tiny-local-model"
    assert entries[0]["parsed_ok"] is True
    assert entries[0]["class"] == "valid_json"


@pytest.mark.asyncio
async def test_expansion_records_malformed_json_before_repair(tmp_path, monkeypatch):
    out = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("IDEA_TEST_JSON_TELEMETRY", "1")
    monkeypatch.setenv("IDEA_TEST_JSON_TELEMETRY_PATH", str(out))

    candidates = await _expand(_FENCED)

    # The repair fallback still salvages the plan; telemetry reports the RAW failure.
    assert len(candidates) == 1
    entries = _entries(out)
    assert len(entries) == 1
    assert entries[0]["phase"] == "native_expansion"
    assert entries[0]["parsed_ok"] is False
    assert entries[0]["class"] == "fenced_json"


@pytest.mark.asyncio
async def test_expansion_records_empty_completion_without_crashing(tmp_path, monkeypatch):
    out = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv("IDEA_TEST_JSON_TELEMETRY", "1")
    monkeypatch.setenv("IDEA_TEST_JSON_TELEMETRY_PATH", str(out))

    await _expand("")

    entries = _entries(out)
    assert len(entries) == 1
    assert entries[0]["parsed_ok"] is False
    assert entries[0]["class"] == "empty"
