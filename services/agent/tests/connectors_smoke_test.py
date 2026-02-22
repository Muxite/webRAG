import os
import pytest

from agent.app.connector_http import ConnectorHttp
from agent.app.connector_search import ConnectorSearch
from agent.app.connector_chroma import ConnectorChroma
from agent.app.connector_llm import ConnectorLLM
from shared.connector_config import ConnectorConfig


def _real_enabled() -> bool:
    return os.environ.get("CONNECTOR_SMOKE_TESTS", "").lower() in ("1", "true", "yes", "on")


@pytest.mark.asyncio
async def test_connector_http_smoke():
    if not _real_enabled():
        pytest.skip("CONNECTOR_SMOKE_TESTS not enabled")
    connector = ConnectorHttp(ConnectorConfig())
    async with connector:
        result = await connector.request("GET", "https://example.com")
        assert result.error is False
        assert result.status == 200


@pytest.mark.asyncio
async def test_connector_search_smoke():
    if not _real_enabled():
        pytest.skip("CONNECTOR_SMOKE_TESTS not enabled")
    config = ConnectorConfig()
    if not config.search_api_key:
        pytest.skip("SEARCH_API_KEY not configured")
    connector = ConnectorSearch(config)
    async with connector:
        await connector.init_search_api()
        results = await connector.query_search("test query", count=1)
        assert results is not None


@pytest.mark.asyncio
async def test_connector_chroma_smoke():
    if not _real_enabled():
        pytest.skip("CONNECTOR_SMOKE_TESTS not enabled")
    config = ConnectorConfig()
    if not config.chroma_url:
        pytest.skip("CHROMA_URL not configured")
    connector = ConnectorChroma(config)
    await connector.init_chroma()
    collection = "test_smoke_collection"
    added = await connector.add_to_chroma(
        collection=collection,
        ids=["1"],
        metadatas=[{"tag": "smoke"}],
        documents=["Smoke test document"],
    )
    assert added is True
    result = await connector.query_chroma(collection, ["Smoke test"], n_results=1)
    assert result is not None


@pytest.mark.asyncio
async def test_connector_llm_smoke():
    if not _real_enabled():
        pytest.skip("CONNECTOR_SMOKE_TESTS not enabled")
    config = ConnectorConfig()
    if not config.llm_api_url or not config.openai_api_key:
        pytest.skip("MODEL_API_URL/OPENAI_API_KEY not configured")
    connector = ConnectorLLM(config)
    payload = connector.build_payload(
        messages=[{"role": "user", "content": "Reply with pong"}],
        json_mode=False,
    )
    response = await connector.query_llm(payload)
    assert response is not None


@pytest.mark.asyncio
async def test_connector_llm_api_structure():
    """Test that the LLM connector builds payloads with correct API structure."""
    config = ConnectorConfig()
    connector = ConnectorLLM(config)
    
    # Test with all new parameters
    test_schema = {
        "name": "test_schema",
        "schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            },
            "required": ["result"],
            "additionalProperties": False
        }
    }
    
    payload = connector.build_payload(
        messages=[{"role": "user", "content": "test"}],
        json_mode=True,
        model_name="gpt-5-mini",
        temperature=0.5,
        max_tokens=100,
        json_schema=test_schema,
        reasoning_effort="high",
        text_verbosity="medium",
    )
    
    # Verify structure
    assert "reasoning_effort" in payload, "reasoning_effort should be top-level"
    assert payload["reasoning_effort"] == "high", "reasoning_effort should be 'high'"
    
    assert "text" in payload, "text should be present"
    assert isinstance(payload["text"], dict), "text should be a dict"
    assert payload["text"]["verbosity"] == "medium", "text.verbosity should be 'medium'"
    
    assert "response_format" in payload, "response_format should be present"
    assert payload["response_format"]["type"] == "json_schema", "response_format.type should be 'json_schema'"
    assert "json_schema" in payload["response_format"], "response_format should contain json_schema"
    
    # For GPT-5 models, max_tokens should be converted to max_completion_tokens
    if payload.get("model", "").startswith(("gpt-5", "gpt-4o")):
        assert "max_completion_tokens" in payload, "max_completion_tokens should be present for GPT-5/GPT-4o"
        assert "max_tokens" not in payload, "max_tokens should be removed for GPT-5/GPT-4o"
    
    # Test without json_schema (should use json_object)
    payload2 = connector.build_payload(
        messages=[{"role": "user", "content": "test"}],
        json_mode=True,
        model_name="gpt-5-mini",
        json_schema=None,
    )
    assert payload2["response_format"]["type"] == "json_object", "Should use json_object when no schema provided"
    
    # Test without reasoning_effort and text_verbosity
    payload3 = connector.build_payload(
        messages=[{"role": "user", "content": "test"}],
        json_mode=False,
        model_name="gpt-5-mini",
    )
    assert "reasoning_effort" not in payload3, "reasoning_effort should not be present when not provided"
    assert "text" not in payload3, "text should not be present when not provided"
