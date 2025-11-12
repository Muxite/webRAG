import pytest_asyncio
import pytest
from app.connector_search import ConnectorSearch
from app.connector_config import ConnectorConfig

@pytest_asyncio.fixture
def connector_config():
    return ConnectorConfig()

@pytest_asyncio.fixture()
def connector(connector_config):
    return ConnectorSearch(connector_config)

@pytest.mark.asyncio
async def test_query_search(connector):
    """Test search query returns valid response."""
    async with connector as search_connector:
        query = "How many fish are there?"
        search_result = await search_connector.query_search(query)
        assert search_result is not None, "Search query failed"
        assert isinstance(search_result, list)
