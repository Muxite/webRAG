"""F7 — the degenerate single-node fallback is TAGGED and COUNTED, not silent.

When ``_parse_candidates`` returns nothing usable, ``_create_fallback_candidate`` emits ONE
guessed candidate (a URL regexed out of the title/mandate, a keyword-guessed search whose query
is ``title[:100]``, or a bare think node) and that single candidate becomes the parent's WHOLE
expansion. A subtree that should have fanned out collapses to one leaf, and until now nothing
distinguished it from a genuinely one-step plan.

This is pure instrumentation — no re-planning reaction (that is F6) — so what is pinned here is
exactly that:

* every fallback branch stamps ``DetailKey.FALLBACK_EXPANSION`` on the candidate it emits;
* a real (parsed) expansion never carries the tag;
* the tag survives the trip through ``graph.expand`` onto the created child;
* the run's final payload carries ``degenerate_fallback_count`` when any fired, and keeps its
  exact previous shape (no key at all) when none did;
* the collapse is logged at WARNING.

Offline: the LLM call is a scripted string, and the payload test mocks ``build_final_payload``
(``confidence_early_exit_run_loop_test``'s pattern).
"""
from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.app.idea_engine as engine_mod
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.expansion import LlmExpansionPolicy
from agent.app.llm_backends import LLMBackend


_TAG = DetailKey.FALLBACK_EXPANSION.value


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "LLM_PROVIDER", "LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY", "MODEL_API_URL", "OPENAI_BASE_URL", "OPENROUTER_BASE_URL",
        "MODEL_NAME", "IDEA_TEST_JSON_TELEMETRY",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_NAME", "gpt-4.1-nano")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")


def _connector():
    from shared.connector_config import ConnectorConfig

    backend = AsyncMock(spec=LLMBackend)
    backend.normalize_payload.side_effect = lambda p, *_, **__: p
    backend.simplify_payload.side_effect = lambda p: dict(p)
    with patch("agent.app.connector_llm.create_llm_backend", return_value=backend):
        from agent.app.connector_llm import ConnectorLLM

        return ConnectorLLM(ConnectorConfig())


class ScriptedIO:
    """One scripted completion, whatever the prompt (``expansion_input_output_framing_test``)."""

    def __init__(self, connector, response: str):
        self._connector = connector
        self._response = response
        self.calls: list = []

    def set_telemetry(self, telemetry):
        return None

    def build_llm_payload(self, **kwargs):
        self.calls.append(kwargs["messages"])
        return self._connector.build_payload(**kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs):
        return self._response


#: The live failure shape: valid JSON, wrong object — no ``candidates`` key at all.
_ECHOED_REPLY = json.dumps({"path": [{"node_id": "abc", "title": "t"}], "parent_id": "abc"})
_REAL_PLAN_REPLY = json.dumps({
    "candidates": [{
        "title": "Jengish Chokusu",
        "action": "search",
        "details": {"query": "Jengish Chokusu topographic prominence"},
    }]
})

_MANDATE = (
    "Search for and report the topographic prominence of each of the following six mountains, "
    "reading each page rather than guessing from memory."
)


def _node(title, mandate=""):
    graph = IdeaDag(root_title=title, root_details={"mandate": mandate})
    return graph.get_node(graph.root_id())


# --------------------------------------------------------------------------------------
# the tag, branch by branch
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,mandate,action",
    [
        ("Read https://example.org/peaks", "", IdeaActionType.VISIT.value),
        ("Find the prominence of Noshaq", "", IdeaActionType.SEARCH.value),
        ("Peaks", "", IdeaActionType.THINK.value),
    ],
    ids=["url_visit", "keyword_search", "generic_think"],
)
def test_every_fallback_branch_tags_the_candidate_it_emits(title, mandate, action):
    policy = LlmExpansionPolicy(io=AsyncMock(), model_name="m")

    candidate = policy._create_fallback_candidate(_node(title, mandate))

    assert candidate["details"][DetailKey.ACTION.value] == action
    assert candidate["details"][_TAG] is True


@pytest.mark.asyncio
async def test_an_unparseable_reply_yields_one_tagged_candidate_and_a_warning(caplog):
    io = ScriptedIO(_connector(), _ECHOED_REPLY)
    policy = LlmExpansionPolicy(io=io, model_name="m")
    graph = IdeaDag(root_title=_MANDATE, root_details={"mandate": _MANDATE})

    with caplog.at_level(logging.WARNING):
        candidates = await policy.expand(graph, graph.root_id())

    assert len(candidates) == 1, "the whole expansion is this one guessed node"
    assert candidates[0]["details"][_TAG] is True
    assert any("DEGENERATE FALLBACK" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_parsed_expansion_is_never_tagged():
    io = ScriptedIO(_connector(), _REAL_PLAN_REPLY)
    policy = LlmExpansionPolicy(io=io, model_name="m")
    graph = IdeaDag(root_title=_MANDATE, root_details={"mandate": _MANDATE})

    candidates = await policy.expand(graph, graph.root_id())

    assert len(candidates) == 1
    assert _TAG not in candidates[0]["details"]


def test_the_tag_rides_onto_the_created_child_through_graph_expand():
    policy = LlmExpansionPolicy(io=AsyncMock(), model_name="m")
    graph = IdeaDag(root_title=_MANDATE, root_details={"mandate": _MANDATE})
    candidate = policy._create_fallback_candidate(graph.get_node(graph.root_id()))

    children = graph.expand(graph.root_id(), [candidate])

    assert len(children) == 1
    assert children[0].details[_TAG] is True


# --------------------------------------------------------------------------------------
# the run-level counter
# --------------------------------------------------------------------------------------


async def _fixed_payload(*args, **kwargs):
    return {"final_deliverable": "answer", "success": True}


def _engine(monkeypatch):
    monkeypatch.setattr(engine_mod, "build_final_payload", _fixed_payload)
    io = MagicMock()
    io.connector_chroma = None
    io.telemetry = None
    return IdeaDagEngine(io=io, settings={}, model_name="m")


def _graph_with(tagged: int) -> IdeaDag:
    graph = IdeaDag(root_title=_MANDATE, root_details={"mandate": _MANDATE})
    for i in range(tagged):
        graph.add_child(graph.root_id(), f"degenerate {i}", details={_TAG: True})
    graph.add_child(graph.root_id(), "organic", details={})
    return graph


@pytest.mark.asyncio
async def test_the_final_payload_counts_every_degenerate_fallback(monkeypatch):
    engine = _engine(monkeypatch)
    graph = _graph_with(2)

    payload = await engine.finalize(graph, _MANDATE, pending_check=False)

    assert payload["degenerate_fallback_count"] == 2


@pytest.mark.asyncio
async def test_a_healthy_run_keeps_its_exact_payload_shape(monkeypatch):
    engine = _engine(monkeypatch)
    graph = _graph_with(0)

    payload = await engine.finalize(graph, _MANDATE, pending_check=False)

    assert "degenerate_fallback_count" not in payload
