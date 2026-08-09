"""GoT schemas vs OpenAI/Azure strict structured-output rules — offline, no LLM.

OpenAI strict mode enforces two rules on every object in a json_schema:
  (a) ``additionalProperties: false`` is present, and
  (b) ``required`` enumerates *every* key in ``properties``.

EXPANSION (free-form ``details``) and MERGE (optional ``goal_evaluation`` /
``missing_requirements``) cannot satisfy both, so those two stages convey their
shape as a prompt instruction and use ``response_format: {"type":"json_object"}``.
The remaining strict-sent schemas (final / evaluation / evaluation_batch) MUST
stay clean — this guards against a regression that would trigger a live 400.
"""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, patch

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_schemas import DEFAULT_JSON_SCHEMAS
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.actions import MergeLeafAction
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.llm_backends import LLMBackend


def _strict_violations(schema, path="<root>"):
    """Walk a JSON schema; report every object that breaks an OpenAI strict rule."""
    out = []
    if not isinstance(schema, dict):
        return out
    if schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            out.append(f"{path}: missing additionalProperties:false")
        props = schema.get("properties")
        prop_keys = set(props.keys()) if isinstance(props, dict) else set()
        required = set(schema.get("required", []) or [])
        missing = prop_keys - required
        if missing:
            out.append(f"{path}: required omits {sorted(missing)}")
        if isinstance(props, dict):
            for key, sub in props.items():
                out += _strict_violations(sub, f"{path}.{key}")
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        out += _strict_violations(schema["items"], f"{path}[]")
    return out


# Schemas still sent on the wire as strict json_schema must satisfy BOTH rules.
STRICT_SENT = ("final_json_schema", "evaluation_json_schema", "evaluation_batch_json_schema")
# Schemas conveyed as a prompt hint + json_object (cannot satisfy strict mode).
JSON_OBJECT_SENT = ("expansion_json_schema", "merge_json_schema")


@pytest.mark.parametrize("key", STRICT_SENT)
def test_strict_sent_schemas_pass_both_rules(key):
    violations = _strict_violations(DEFAULT_JSON_SCHEMAS[key]["schema"])
    assert violations == [], f"{key} would be rejected by strict mode: {violations}"


@pytest.mark.parametrize("key", JSON_OBJECT_SENT)
def test_json_object_schemas_are_exempt_for_a_reason(key):
    # These intentionally break strict mode, which is exactly why their stages
    # drop to json_object; assert the violation exists so the exemption is documented.
    assert _strict_violations(DEFAULT_JSON_SCHEMAS[key]["schema"]), f"{key} now passes strict; revisit json_object choice"


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
    """Delegates build_llm_payload to a real connector so response_format is faithful."""

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
async def test_merge_emits_json_object_and_conveys_shape_in_prompt():
    response = json.dumps(
        {
            "summary": "combined",
            "key_findings": ["a", "b"],
            "goal_achieved": True,
            "goal_evaluation": "ok",
            "missing_requirements": [],
        }
    )
    connector = _make_connector_with_mock_backend()
    io = RecordingIO(connector, response)
    action = MergeLeafAction(settings=load_idea_dag_settings())

    graph = IdeaDag(root_title="root", root_details={"mandate": "m"})
    merge_node = graph.add_child(
        parent_id=graph.root_id(),
        title="Merge findings",
        details={
            DetailKey.ACTION.value: IdeaActionType.MERGE.value,
            DetailKey.GOAL.value: "combine the facts",
            DetailKey.MERGED_RESULTS.value: [{"summary": "fact A"}, {"summary": "fact B"}],
        },
    )

    result = await action.execute(graph, merge_node.node_id, io)
    assert result.get("success") is True

    # The merge stage no longer sends a strict json_schema response_format.
    assert io.payload_kwargs.get("json_mode") is True
    assert io.payload_kwargs.get("json_schema") is None
    assert io.last_payload.get("response_format") == {"type": "json_object"}

    # ...but the merge_result shape is still conveyed to the model as prompt text.
    prompt_text = "\n".join(str(m.get("content", "")) for m in io.payload_kwargs["messages"])
    assert "JSON Schema" in prompt_text
    for token in ("summary", "key_findings", "goal_achieved", "goal_evaluation", "missing_requirements"):
        assert token in prompt_text, f"missing shape token {token!r} in merge prompt"
