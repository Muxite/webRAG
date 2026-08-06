"""Input/output framing for the expansion prompt (opt-in,
``expansion_input_output_framing_enabled``) and its bounded corrective retry (opt-in,
``expansion_echo_retry_enabled``).

The bug these fix, from live telemetry (2026-08-06, qwen2.5:0.5b, native ``graph`` variant,
``phase: native_expansion``): 6 of 8 syntactically-VALID expansion completions were the wrong
SHAPE — 5 echoed the user prompt's ``"path"`` context blob back, 1 echoed the schema hint's own
``{"name": "expansion_result", "schema": ...}`` envelope. The shipped user prompt reads
``Return your response as valid JSON. {"path": [...], ...}`` — an output imperative followed
immediately by an input blob — while the real output shape is stated once, far away, on the last
line of a long system prompt. A reply with no ``candidates`` key drops to
``_create_fallback_candidate``, which for the ROOT node (whose title IS the mandate) emits a
search query that is the mandate's first 100 characters, so the run makes zero page visits.

Mocked-LLM tests could never surface this — their scripted responses are well-formed by
construction — so ``test_echo_reply_without_the_retry_falls_back_to_a_truncated_mandate_query``
below stands in as the offline witness for the real-model failure.

Contracts pinned here:
  * flag OFF -> the rendered user message is BYTE-IDENTICAL (every benchmark/fixture run);
  * flag ON  -> the context blob is labeled read-only input, the misleading lead sentence is
    gone, the ``{candidates: [...]}`` shape is restated immediately after the blob, and the
    payload between them is preserved byte-for-byte;
  * flag ON  -> the schema hint loses its copyable ``{"name", "schema"}`` envelope;
  * the echo detector fires on THIS failure shape only (parsed, no candidates, input/schema key
    present) — never on malformed JSON, which ``_repair_json_object`` already owns;
  * the retry is hard-bounded at one extra call and degrades to the existing fallback.
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, patch

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.expansion import (
    _INPUT_FRAMING_FOOTER,
    _INPUT_FRAMING_HEADER,
    _USER_PROMPT_OUTPUT_LEAD,
    LlmExpansionPolicy,
    detect_input_echo,
    frame_expansion_user_prompt,
)
from agent.app.llm_backends import LLMBackend


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


def _make_connector_with_mock_backend():
    from shared.connector_config import ConnectorConfig

    cfg = ConnectorConfig()
    mock_backend = AsyncMock(spec=LLMBackend)
    mock_backend.normalize_payload.side_effect = lambda p, *_, **__: p
    mock_backend.simplify_payload.side_effect = lambda p: dict(p)
    with patch("agent.app.connector_llm.create_llm_backend", return_value=mock_backend):
        from agent.app.connector_llm import ConnectorLLM

        return ConnectorLLM(cfg)


class ScriptedIO:
    """Returns each scripted completion in turn and records every call's messages."""

    def __init__(self, connector, *responses: str):
        self._connector = connector
        self._responses = list(responses)
        self.calls: list = []

    def set_telemetry(self, telemetry):
        return None

    def build_llm_payload(self, **kwargs):
        self.calls.append(kwargs["messages"])
        return self._connector.build_payload(**kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs):
        if len(self.calls) <= len(self._responses):
            return self._responses[len(self.calls) - 1]
        return self._responses[-1]


# The real failure shapes, transcribed from the recorded telemetry.
_ECHOED_PATH_REPLY = json.dumps({
    "path": [{"node_id": "9179fb6b", "title": "You are given NO URLs - search to find the pages",
              "action": "visit:link_count=1", "details": {"optional_url": "<url>"}}],
    "parent_id": "9179fb6b",
})
_ECHOED_SCHEMA_REPLY = json.dumps({
    "name": "expansion_result",
    "schema": {"type": "object", "properties": {"candidates": [
        {"title": "Jengish Chokusu", "action": "visit",
         "details": {"optional_url": "<not provided>"}}]}},
})
_REAL_PLAN_REPLY = json.dumps({
    "candidates": [{
        "title": "Jengish Chokusu (Kyrgyzstan / China)",
        "action": "search",
        "details": {"query": "Jengish Chokusu topographic prominence"},
    }]
})

_MANDATE = (
    "You are given NO URLs - search to find the pages you need, then READ them (do not guess "
    "from memory). For each of the following six mountains, report the topographic prominence."
)


def _graph():
    return IdeaDag(root_title=_MANDATE, root_details={"mandate": _MANDATE})


def _user_message(policy, graph=None) -> str:
    graph = graph or _graph()
    return policy._build_messages(graph, graph.get_node(graph.root_id()))[1]["content"]


# --- the no-op guarantee -------------------------------------------------------------------


def test_flag_off_renders_the_template_byte_for_byte():
    """A template with no framing applied must reach the model EXACTLY as authored — pinned on a
    trivial custom template so the assertion is a literal, not a re-derivation of engine state."""
    graph = _graph()
    policy = LlmExpansionPolicy(
        io=AsyncMock(), model_name="m",
        settings={"expansion_user_prompt": "CTX for {parent_id} / {parent_title}"},
    )

    user = policy._build_messages(graph, graph.get_node(graph.root_id()))[1]["content"]

    assert user == f"CTX for {graph.root_id()} / {_MANDATE}"


def test_flag_off_shipped_prompt_keeps_its_lead_sentence_and_gains_no_framing():
    policy = LlmExpansionPolicy(io=AsyncMock(), model_name="m")  # default settings -> flag OFF

    user = _user_message(policy)

    assert user.startswith(_USER_PROMPT_OUTPUT_LEAD + '{"path":')
    assert _INPUT_FRAMING_HEADER not in user
    assert "END OF INPUT CONTEXT." not in user


# --- the framing itself --------------------------------------------------------------------


def test_flag_on_labels_the_input_and_restates_the_output_shape_after_it():
    policy = LlmExpansionPolicy(
        io=AsyncMock(), model_name="m",
        settings={"expansion_input_output_framing_enabled": True},
    )

    user = _user_message(policy)

    # The misleading imperative ("Return your response as valid JSON." straight into the input
    # blob) is gone — that sentence is what a weak model obeyed literally.
    assert _USER_PROMPT_OUTPUT_LEAD not in user
    # Input is labeled BEFORE the blob, output shape restated AFTER it (the recency position the
    # model was copying from).
    assert user.index(_INPUT_FRAMING_HEADER) < user.index('{"path":') < user.index("END OF INPUT")
    assert user.endswith(_INPUT_FRAMING_FOOTER)
    assert '{"candidates": [{"title":' in user
    # The observed wrong-shape keys are named explicitly as wrong.
    for wrong_key in ('"path"', '"node_id"', '"name"', '"schema"'):
        assert wrong_key in _INPUT_FRAMING_FOOTER


def test_flag_on_preserves_the_context_payload_byte_for_byte():
    """The framing only adds a label and a restatement; the run's actual context (path,
    event_log, memories, ...) must be unchanged, so nothing an operator tuned is lost."""
    graph = _graph()  # one graph: the node ids inside the context blob must line up
    off = LlmExpansionPolicy(io=AsyncMock(), model_name="m")
    on = LlmExpansionPolicy(
        io=AsyncMock(), model_name="m",
        settings={"expansion_input_output_framing_enabled": True},
    )

    payload = _user_message(off, graph)[len(_USER_PROMPT_OUTPUT_LEAD):]

    assert _user_message(on, graph) == (
        f"{_INPUT_FRAMING_HEADER}\n{payload}\n\n{_INPUT_FRAMING_FOOTER}"
    )


def test_custom_template_without_the_lead_sentence_is_wrapped_unchanged():
    """Alternate settings files and sequential mode carry their own user prompt; they must gain
    the framing without losing a single character of their own text."""
    framed = frame_expansion_user_prompt("SOME OTHER CONTEXT {}")

    assert framed == f"{_INPUT_FRAMING_HEADER}\nSOME OTHER CONTEXT {{}}\n\n{_INPUT_FRAMING_FOOTER}"


async def _expand(policy):
    graph = _graph()
    return await policy.expand(graph, graph.root_id())


async def _sent_prompt(**settings) -> str:
    """The full prompt an expansion CALL actually sends (the schema hint is attached inside
    ``expand``, not by ``_build_messages``)."""
    io = ScriptedIO(_make_connector_with_mock_backend(), _REAL_PLAN_REPLY)
    await _expand(LlmExpansionPolicy(io=io, settings=settings or None))
    return "\n".join(str(m.get("content", "")) for m in io.calls[0])


@pytest.mark.asyncio
async def test_flag_on_unwraps_the_copyable_schema_envelope():
    """``json_instruction_from_response_format`` dumps the whole {"name", "schema"} envelope into
    the system prompt, and one recorded completion returned that envelope verbatim. With the
    framing on, the hint carries the schema BODY only — same constraints, nothing to copy."""
    off = await _sent_prompt()

    # The default path still ships the envelope (byte-identical guarantee).
    assert '"name": "expansion_result"' in off

    on = await _sent_prompt(expansion_input_output_framing_enabled=True)

    assert '"name": "expansion_result"' not in on
    assert '"candidates"' in on and '"required"' in on


# --- the echo detector ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        (_ECHOED_PATH_REPLY, "path"),
        (_ECHOED_SCHEMA_REPLY, "name"),
        (json.dumps({"node_id": "x", "title": "t"}), "node_id"),
        (json.dumps({"event_log": []}), "event_log"),
        # not this failure shape:
        (_REAL_PLAN_REPLY, None),
        (json.dumps({"path": [], "candidates": [{"title": "t", "action": "search"}]}), None),
        (json.dumps({}), None),
        (json.dumps({"candidates": []}), None),
        (json.dumps({"answer": "42"}), None),
        ('{"path": [ broken', None),      # malformed JSON is a DIFFERENT class (json repair owns it)
        ("", None),
        (None, None),
    ],
)
def test_detect_input_echo_fires_on_this_shape_only(reply, expected):
    assert detect_input_echo(reply) == expected


def test_detect_input_echo_names_the_replys_own_first_key():
    """The corrective prompt quotes the key back at the model, so it must be the key the model
    actually led with, not whichever member of the watch-list happens to sort first."""
    assert detect_input_echo(json.dumps({"errors": "None", "path": []})) == "errors"


# --- the bounded retry ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_echo_reply_without_the_retry_falls_back_to_a_truncated_mandate_query():
    """The live failure, reproduced offline: an echoed reply yields no candidates, the fallback
    fires, and the ROOT node's fallback search query is the MANDATE's first 100 characters — an
    instruction preamble, not an entity — which is why those runs made zero page visits."""
    io = ScriptedIO(_make_connector_with_mock_backend(), _ECHOED_PATH_REPLY)
    policy = LlmExpansionPolicy(io=io)  # both flags OFF

    candidates = await _expand(policy)

    assert len(io.calls) == 1, "no retry when the flag is off"
    assert len(candidates) == 1
    details = candidates[0]["details"]
    assert details[DetailKey.ACTION.value] == IdeaActionType.SEARCH.value
    assert details[DetailKey.QUERY.value] == _MANDATE[:100]
    assert "Fallback candidate" in details[DetailKey.JUSTIFICATION.value]


@pytest.mark.asyncio
async def test_retry_recovers_a_real_plan_from_an_echoed_reply():
    io = ScriptedIO(_make_connector_with_mock_backend(), _ECHOED_PATH_REPLY, _REAL_PLAN_REPLY)
    policy = LlmExpansionPolicy(io=io, settings={"expansion_echo_retry_enabled": True})

    candidates = await _expand(policy)

    assert len(io.calls) == 2, "exactly one extra attempt"
    # The corrective directive rides on the existing user turn (role alternation preserved:
    # same message count as the first call) and names the offending key back at the model.
    retry_messages = io.calls[1]
    assert [m["role"] for m in retry_messages] == [m["role"] for m in io.calls[0]]
    assert retry_messages[-1]["role"] == "user"
    assert retry_messages[-1]["content"].startswith(io.calls[0][-1]["content"])
    assert '"path"' in retry_messages[-1]["content"]
    assert '{"candidates":' in retry_messages[-1]["content"]
    # And the plan is the model's, not the fallback's.
    assert len(candidates) == 1
    assert candidates[0]["title"] == "Jengish Chokusu (Kyrgyzstan / China)"
    assert candidates[0]["details"][DetailKey.QUERY.value] == (
        "Jengish Chokusu topographic prominence"
    )


@pytest.mark.asyncio
async def test_retry_is_bounded_at_one_attempt_and_still_falls_back():
    """A model that echoes twice costs exactly one extra call, then the pre-existing fallback
    path takes over unchanged."""
    io = ScriptedIO(
        _make_connector_with_mock_backend(), _ECHOED_PATH_REPLY, _ECHOED_SCHEMA_REPLY
    )
    policy = LlmExpansionPolicy(io=io, settings={"expansion_echo_retry_enabled": True})

    candidates = await _expand(policy)

    assert len(io.calls) == 2
    assert "Fallback candidate" in candidates[0]["details"][DetailKey.JUSTIFICATION.value]


@pytest.mark.asyncio
async def test_retry_does_not_fire_for_malformed_json():
    """Malformed JSON is the repair path's business; spending a second call on it would be a
    silent cost regression on a failure this lever does not address."""
    io = ScriptedIO(_make_connector_with_mock_backend(), '{"path": [ this is not json')
    policy = LlmExpansionPolicy(io=io, settings={"expansion_echo_retry_enabled": True})

    await _expand(policy)

    assert len(io.calls) == 1


@pytest.mark.asyncio
async def test_a_failing_retry_degrades_to_the_fallback_not_to_an_empty_plan():
    class _AngryIO(ScriptedIO):
        async def query_llm_with_fallback(self, payload, **kwargs):
            if len(self.calls) > 1:
                raise RuntimeError("boom")
            return _ECHOED_PATH_REPLY

    io = _AngryIO(_make_connector_with_mock_backend(), _ECHOED_PATH_REPLY)
    policy = LlmExpansionPolicy(io=io, settings={"expansion_echo_retry_enabled": True})

    candidates = await _expand(policy)

    assert len(candidates) == 1
    assert "Fallback candidate" in candidates[0]["details"][DetailKey.JUSTIFICATION.value]


@pytest.mark.asyncio
async def test_a_good_first_reply_never_triggers_the_retry():
    io = ScriptedIO(_make_connector_with_mock_backend(), _REAL_PLAN_REPLY)
    policy = LlmExpansionPolicy(
        io=io,
        settings={
            "expansion_echo_retry_enabled": True,
            "expansion_input_output_framing_enabled": True,
        },
    )

    candidates = await _expand(policy)

    assert len(io.calls) == 1
    assert len(candidates) == 1
