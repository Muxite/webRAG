"""Typed decomposition contract: optional per-leaf ``expect`` field (opt-in,
``expansion_expect_contract_enabled``).

Ports the compiled path's measurable-output discipline into native expansion: a leaf
candidate may declare a one-line ``expect`` contract (exact value + source URL), which
threads candidate -> node detail (``DetailKey.EXPECT``) -> leaf execution (as an
extraction-target addendum to the intent) and is auto-serialized into evaluation.

Contracts pinned here:
  * flag OFF -> the expansion prompt + schema hint carry NO ``expect`` mention, and an
    ``expect`` in the model response is NOT parsed onto the node (default byte-identical);
  * flag ON -> the contract instruction + schema hint are injected, and a candidate's
    ``expect`` is parsed onto ``DetailKey.EXPECT``;
  * flag ON but a candidate WITHOUT ``expect`` still parses (optional field);
  * leaf execution folds a present ``expect`` into the effective intent, and is a no-op
    (intent unchanged) when absent.
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, patch

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.expansion import LlmExpansionPolicy, _EXPECT_CONTRACT_ADDENDUM
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.actions import SearchLeafAction
from agent.app.llm_backends import LLMBackend


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "LLM_PROVIDER", "LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY", "MODEL_API_URL", "OPENAI_BASE_URL", "OPENROUTER_BASE_URL",
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


def _prompt_text(io) -> str:
    return "\n".join(str(m.get("content", "")) for m in io.payload_kwargs["messages"])


_LEAF_WITH_EXPECT = {
    "candidates": [{
        "title": "Find the founding year of the observatory",
        "action": "search",
        "details": {"query": "observatory founding year"},
        "expect": "the exact founding year (e.g. 1861) AND the source URL",
    }]
}
_LEAF_WITHOUT_EXPECT = {
    "candidates": [{
        "title": "Find the founding year of the observatory",
        "action": "search",
        "details": {"query": "observatory founding year"},
    }]
}


@pytest.mark.asyncio
async def test_flag_off_prompt_has_no_expect_and_response_expect_not_parsed():
    connector = _make_connector_with_mock_backend()
    io = RecordingIO(connector, json.dumps(_LEAF_WITH_EXPECT))
    policy = LlmExpansionPolicy(io=io)  # default settings -> flag OFF

    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    candidates = await policy.expand(graph, graph.root_id())

    prompt = _prompt_text(io)
    assert _EXPECT_CONTRACT_ADDENDUM not in prompt, "contract addendum must be absent when flag off"
    assert "MEASURABLE OUTPUT CONTRACT" not in prompt
    # The schema hint must not advertise the optional ``expect`` field when off.
    assert '"expect"' not in prompt
    # Even though the response carried an ``expect``, the flag-off parse must ignore it.
    assert len(candidates) == 1
    assert DetailKey.EXPECT.value not in candidates[0]["details"]


@pytest.mark.asyncio
async def test_flag_on_injects_contract_and_parses_expect():
    connector = _make_connector_with_mock_backend()
    io = RecordingIO(connector, json.dumps(_LEAF_WITH_EXPECT))
    policy = LlmExpansionPolicy(io=io, settings={"expansion_expect_contract_enabled": True})

    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    candidates = await policy.expand(graph, graph.root_id())

    prompt = _prompt_text(io)
    assert _EXPECT_CONTRACT_ADDENDUM in prompt, "contract addendum must be injected when flag on"
    # The schema hint now advertises the optional ``expect`` field.
    assert '"expect"' in prompt or "expect" in prompt
    # The candidate's expect is parsed onto the node detail.
    assert len(candidates) == 1
    assert candidates[0]["details"][DetailKey.EXPECT.value] == (
        "the exact founding year (e.g. 1861) AND the source URL"
    )


@pytest.mark.asyncio
async def test_flag_on_candidate_without_expect_still_parses():
    connector = _make_connector_with_mock_backend()
    io = RecordingIO(connector, json.dumps(_LEAF_WITHOUT_EXPECT))
    policy = LlmExpansionPolicy(io=io, settings={"expansion_expect_contract_enabled": True})

    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    candidates = await policy.expand(graph, graph.root_id())

    assert len(candidates) == 1
    assert candidates[0]["details"].get("query") == "observatory founding year"
    assert DetailKey.EXPECT.value not in candidates[0]["details"], "expect is optional"


@pytest.mark.asyncio
async def test_flag_on_accepts_expect_nested_in_details():
    nested = {
        "candidates": [{
            "title": "Find the height",
            "action": "search",
            "details": {"query": "tower height", "expect": "the exact height in metres AND the URL"},
        }]
    }
    connector = _make_connector_with_mock_backend()
    io = RecordingIO(connector, json.dumps(nested))
    policy = LlmExpansionPolicy(io=io, settings={"expansion_expect_contract_enabled": True})

    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    candidates = await policy.expand(graph, graph.root_id())

    assert candidates[0]["details"][DetailKey.EXPECT.value] == "the exact height in metres AND the URL"


# --- leaf execution surfacing --------------------------------------------------------
def _search_node(graph: IdeaDag, *, intent=None, expect=None):
    details = {DetailKey.ACTION.value: IdeaActionType.SEARCH.value, DetailKey.QUERY.value: "q"}
    if intent is not None:
        details[DetailKey.INTENT.value] = intent
    if expect is not None:
        details[DetailKey.EXPECT.value] = expect
    return graph.add_child(graph.root_id(), "leaf", details=details)


def test_effective_intent_folds_expect_into_extraction_target():
    action = SearchLeafAction(settings={})
    graph = IdeaDag(root_title="root")
    node = _search_node(graph, intent="find the founding year", expect="exact year AND source URL")

    effective = action._effective_intent(node)
    assert "find the founding year" in effective
    assert "Report exactly: exact year AND source URL" in effective


def test_effective_intent_noop_when_expect_absent():
    action = SearchLeafAction(settings={})
    graph = IdeaDag(root_title="root")
    node = _search_node(graph, intent="find the founding year")

    # Byte-identical: returns the plain intent unchanged when no expect is present.
    assert action._effective_intent(node) == "find the founding year"


def test_effective_intent_expect_only_when_no_intent():
    action = SearchLeafAction(settings={})
    graph = IdeaDag(root_title="root")
    node = _search_node(graph, expect="exact value AND URL")

    assert action._effective_intent(node) == "Report exactly: exact value AND URL"
