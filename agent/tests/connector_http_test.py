import pytest_asyncio
import pytest
from app.connector_http import ConnectorHttp
from app.connector_config import ConnectorConfig

@pytest_asyncio.fixture
def connector_config():
    return ConnectorConfig()

@pytest_asyncio.fixture()
def connector(connector_config):
    return ConnectorHttp(connector_config)

@pytest.mark.asyncio
async def test_connector_http(connector):
    """Test the basic HTTP session works."""
    async with connector as http_connector:
        result = await http_connector.request("GET", "https://www.google.com/")
        assert result.error is False
        assert result.cache_retrieved is not None
        assert result.status == 200
        assert isinstance(result.cache_retrieved, str)
        assert "<!doctype html>" in result.cache_retrieved.lower()
