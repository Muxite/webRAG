import pytest_asyncio
import pytest
from agent.app.connector_llm import ConnectorLLM
from shared.connector_config import ConnectorConfig
from agent.app.prompt_builder import PromptBuilder, build_payload

@pytest_asyncio.fixture
def connector_config():
    return ConnectorConfig()

@pytest_asyncio.fixture()
def connector(connector_config):
    return ConnectorLLM(connector_config)

@pytest.mark.asyncio
async def test_llm_query(connector: ConnectorLLM):
    """Test a basic LLM query returns a valid, non-empty response."""
    prompt = "Say 'pong' if you can read this."
    builder = PromptBuilder(observations=prompt)

    llm_response = await connector.query_llm(build_payload(builder.build_messages(), json_mode=False))

    assert llm_response is not None, "LLM query returned None"
    assert isinstance(llm_response, str)
    assert "pong" in llm_response.lower(), "LLM response did not contain 'pong'"
