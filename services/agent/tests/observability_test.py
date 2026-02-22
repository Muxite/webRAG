import pytest

from agent.app.connector_base import ConnectorBase
from agent.app.connector_search import ConnectorSearch
from shared.connector_config import ConnectorConfig
from shared.request_result import RequestResult


class FakeTelemetry:
    def __init__(self):
        self.events = []

    def record_event(self, event, payload):
        """
        Record a telemetry event.
        :param event: Event name.
        :param payload: Event payload.
        :returns: None.
        """
        self.events.append((event, payload))


def test_connector_base_record_io():
    config = ConnectorConfig()
    connector = ConnectorBase(config, name="TestConnector")
    telemetry = FakeTelemetry()
    connector.set_telemetry(telemetry)
    connector._record_io(
        direction="in",
        operation="unit_test",
        payload={"text": "hello", "items": [1, 2, 3], "meta": {"a": 1}},
    )
    assert telemetry.events
    event, payload = telemetry.events[0]
    assert event == "connector_io"
    assert payload["connector"] == "TestConnector"
    assert payload["direction"] == "in"
    assert payload["operation"] == "unit_test"
    summary = payload["payload"]
    assert summary["text"]["chars"] == 5
    assert summary["items"]["count"] == 3
    assert summary["meta"]["count"] == 1


class SearchConnectorStub(ConnectorSearch):
    def __init__(self, config: ConnectorConfig, telemetry: FakeTelemetry):
        super().__init__(config)
        self.search_api_key = "test-key"
        self.set_telemetry(telemetry)

    async def init_search_api(self) -> bool:
        """
        Pretend the search API is ready.
        :returns: True
        """
        return True

    async def request(self, method: str, url: str, retries: int = 4, **kwargs) -> RequestResult:
        """
        Return a fake HTTP response.
        :param method: HTTP method.
        :param url: Request URL.
        :param retries: Retry count.
        :returns: RequestResult instance.
        """
        data = {"web": {"results": [{"title": "A", "url": "https://a.example", "description": "a"}]}}
        return RequestResult(status=200, data=data, error=False)


@pytest.mark.asyncio
async def test_connector_search_records_io_events():
    config = ConnectorConfig()
    telemetry = FakeTelemetry()
    connector = SearchConnectorStub(config, telemetry)
    results = await connector.query_search("query", count=1)
    assert results is not None
    assert len(telemetry.events) >= 2
    directions = [payload["direction"] for _, payload in telemetry.events]
    operations = [payload["operation"] for _, payload in telemetry.events]
    assert "in" in directions
    assert "out" in directions
    assert operations.count("search_query") >= 2
