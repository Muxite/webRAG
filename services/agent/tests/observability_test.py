import pytest

from agent.app.connector_base import ConnectorBase
from agent.app.connector_search import ConnectorSearch
from agent.app.testing.utils import summarize_observability
from shared.connector_config import ConnectorConfig
from shared.request_result import RequestResult


class _EmptyTelemetry:
    """Minimal telemetry with the empty collections summarize_observability reads."""

    events = []
    llm_usage = []
    chroma_stored = []
    chroma_retrieved = []
    documents_seen = []
    timings = []
    decisions = []


def test_summarize_observability_omits_step_confidence_when_absent():
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, _EmptyTelemetry())
    assert obs["step_confidence"] is None


def test_summarize_observability_summarizes_step_confidence_sequence():
    output = {
        "final_deliverable": "x",
        "step_confidences": [
            {"step": 0, "node_id": "a", "kind": "search", "confidence": 0.2, "reason": "weak"},
            {"step": 1, "node_id": "b", "kind": "visit", "confidence": 0.8, "reason": "strong"},
            {"step": 2, "node_id": "c", "kind": "visit", "confidence": "bad", "reason": "skip"},
        ],
    }
    obs = summarize_observability({"output": output}, _EmptyTelemetry())
    sc = obs["step_confidence"]
    assert sc is not None
    # The unparseable third entry is dropped from the numeric sequence but kept in the trace.
    assert sc["sequence"] == [0.2, 0.8]
    assert sc["count"] == 2
    assert sc["mean"] == 0.5
    assert len(sc["trace"]) == 3


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
