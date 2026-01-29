import pytest_asyncio
import pytest
from agent.app.connector_http import ConnectorHttp
from shared.connector_config import ConnectorConfig
import logging


@pytest_asyncio.fixture
def connector_config():
    return ConnectorConfig()

@pytest_asyncio.fixture()
def connector(connector_config):
    return ConnectorHttp(connector_config)

@pytest.mark.asyncio
async def test_connector_http(connector, caplog):
    """Test the basic HTTP session works."""
    caplog.set_level("INFO")
    async with connector as http_connector:
        result = await http_connector.request("GET", "https://www.google.com/")
        assert result.error is False
        assert result.status == 200
        logging.info(result.data[:32])
        assert isinstance(result.data, str)
        assert "<!doctype html>" in result.data.lower()

