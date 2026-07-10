"""Native-GoT expansion drops strict json_schema for json_object — offline, no LLM.

OpenAI/Azure strict structured-output rejects the expansion schema because its
candidate ``details`` is a free-form, per-action object (no ``additionalProperties:
false`` possible without forbidding the action keys). The fix sends
``response_format: {"type": "json_object"}`` while folding the candidate shape into
the prompt as a text instruction so weak models still emit the right structure.
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, patch

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.expansion import LlmExpansionPolicy
from agent.app.llm_backends import LLMBackend


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "MODEL_API_URL",
        "OPENAI_BASE_URL",
        "OPENROUTER_BASE_URL",
        "MODEL_NAME",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_NAME", "gpt-4.1-nano")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")


def _make_connector_with_mock_backend():
    from shared.connector_config import ConnectorConfig

    cfg = ConnectorConfig()
    mock_backend = AsyncMock(spec=LLMBackend)
    mock_backend.normalize_payload.side_effect = lambda p, *_, **__: p
    mock_backend.simplify_payload.side_effect = lambda p: dict(p)
    with patch("agent.app.connector_llm.create_llm_backend", return_value=mock_backend):
        from agent.app.connector_llm import ConnectorLLM

        return ConnectorLLM(cfg)


class RecordingIO:
    """Captures build_llm_payload kwargs and returns a canned expansion response.

    build_llm_payload is delegated to a real ConnectorLLM (mock backend) so the
    produced ``response_format`` is exactly what the wire would carry.
    """

    def __init__(self, connector, response: str):
        self._connector = connector
        self._response = response
        self.payload_kwargs: dict = {}
        self.last_payload: dict = {}

    def set_telemetry(self, telemetry):
        return None

    def build_llm_payload(self, **kwargs):
        self.payload_kwargs = kwargs
        self.last_payload = self._connector.build_payload(**kwargs)
        return self.last_payload

    async def query_llm_with_fallback(self, payload, **kwargs):
        return self._response


@pytest.mark.asyncio
async def test_expansion_emits_json_object_and_conveys_shape_in_prompt():
    response = json.dumps(
        {"candidates": [{"title": "Search cats", "action": "search", "details": {"query": "cats"}}]}
    )
    connector = _make_connector_with_mock_backend()
    io = RecordingIO(connector, response)
    policy = LlmExpansionPolicy(io=io)

    graph = IdeaDag(root_title="Find facts about cats", root_details={"mandate": "Find facts about cats"})
    candidates = await policy.expand(graph, graph.root_id())

    # The canned response still parses into candidates (shape unbroken).
    assert len(candidates) == 1
    assert candidates[0]["details"].get("query") == "cats"

    # The expansion stage no longer sends a strict json_schema response_format.
    assert io.payload_kwargs.get("json_mode") is True
    assert io.payload_kwargs.get("json_schema") is None
    rf = io.last_payload.get("response_format")
    assert rf == {"type": "json_object"}, rf
    assert "json_schema" not in rf

    # ...but the candidate shape is still conveyed to the model as prompt text.
    prompt_text = "\n".join(str(m.get("content", "")) for m in io.payload_kwargs["messages"])
    assert "JSON Schema" in prompt_text
    for token in ("candidates", "title", "action", "details"):
        assert token in prompt_text, f"missing shape token {token!r} in prompt"


@pytest.mark.asyncio
async def test_schema_hint_keeps_message_count_stable():
    connector = _make_connector_with_mock_backend()
    io = RecordingIO(connector, json.dumps({"candidates": []}))
    policy = LlmExpansionPolicy(io=io)

    graph = IdeaDag(root_title="Research task", root_details={"mandate": "Research task"})
    await policy.expand(graph, graph.root_id())

    roles = [m.get("role") for m in io.payload_kwargs["messages"]]
    # The hint is folded into the existing system message, not added as a turn.
    assert roles == ["system", "user"], roles


def test_inject_schema_hint_prepends_system_when_absent():
    connector = _make_connector_with_mock_backend()
    policy = LlmExpansionPolicy(io=RecordingIO(connector, "{}"))
    out = policy._inject_schema_hint([{"role": "user", "content": "hi"}], "SHAPE-HINT")
    assert out[0] == {"role": "system", "content": "SHAPE-HINT"}
    assert out[1] == {"role": "user", "content": "hi"}
