import pytest

from agent.app.connector_base import ConnectorBase
from agent.app.connector_search import ConnectorSearch
from agent.app.testing.utils import _is_infra_timing, summarize_observability
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


# --- F17: infra-failure quarantine classification (testing.utils._is_infra_timing / "infra" block) ---

def _timing(name, success, status=None, infra_failed=None, error=None):
    payload = {}
    if status is not None or name in ("http_request", "search_query", "visit"):
        payload["status"] = status
    if infra_failed is not None:
        payload["infra_failed"] = infra_failed
    entry = {"name": name, "duration": 0.1, "success": success, "payload": payload}
    if error:
        entry["error"] = error
    return entry


def test_is_infra_timing_false_for_success():
    assert _is_infra_timing(_timing("http_request", success=True, status=200)) is False


def test_is_infra_timing_true_for_402_429_5xx():
    assert _is_infra_timing(_timing("http_request", success=False, status=402)) is True
    assert _is_infra_timing(_timing("search_query", success=False, status=422)) is True
    assert _is_infra_timing(_timing("search_query", success=False, status=429)) is True
    assert _is_infra_timing(_timing("http_request", success=False, status=503)) is True


def test_is_infra_timing_false_for_bot_block_status():
    """401/403/404 are the F18 bot-block problem, not infra — must stay a genuine failure so
    quarantine logic never masks a real 403-everywhere task."""
    assert _is_infra_timing(_timing("visit", success=False, status=403)) is False
    assert _is_infra_timing(_timing("http_request", success=False, status=404)) is False


def test_is_infra_timing_true_for_transport_failure_with_no_status():
    """status=None on an op that normally reports one means every attempt exhausted with no
    server response at all (DNS/connect/timeout) — a transport/infra symptom."""
    assert _is_infra_timing(_timing("visit", success=False, status=None)) is True
    assert _is_infra_timing(_timing("http_request", success=False, status=None)) is True


def test_is_infra_timing_honors_explicit_infra_flag_from_connector_llm():
    """llm_call timings carry no status code; connector_llm sets payload.infra_failed directly."""
    assert _is_infra_timing(_timing("llm_call", success=False, infra_failed=True)) is True
    assert _is_infra_timing(_timing("llm_call", success=False, infra_failed=False)) is False


class _TelemetryWithTimings(_EmptyTelemetry):
    def __init__(self, timings):
        self.timings = timings


def test_summarize_observability_reports_no_infra_failure_when_clean():
    telemetry = _TelemetryWithTimings([_timing("http_request", success=True, status=200)])
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["infra"] == {"failed": False, "failure_count": 0, "ops": [], "rates": {}}


def test_summarize_observability_flags_infra_failure_and_counts_ops():
    telemetry = _TelemetryWithTimings([
        _timing("http_request", success=True, status=200),
        _timing("search_query", success=False, status=422, error="Search API query failed: status=422"),
        _timing("llm_call", success=False, infra_failed=True, error="status 402"),
        _timing("visit", success=False, status=403, error="HTTP visit failed"),  # NOT infra
    ])
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    # Each of llm_call and search_query has a single attempt that failed outright (rate 1.0),
    # which is a total outage for that op regardless of the severity threshold.
    assert obs["infra"]["failed"] is True
    assert obs["infra"]["failure_count"] == 2
    assert obs["infra"]["ops"] == ["llm_call", "search_query"]


# --- Bug A: infra.failed severity threshold (material fraction / total outage, not any-op OR) ---


def test_summarize_observability_not_failed_on_partial_http_and_visit_failures():
    """14/16 http_request successes and 10/11 visit successes: occasional transient fetch
    hiccups are normal operating condition on a web-research run, not a corrupt cell."""
    timings = (
        [_timing("http_request", success=True, status=200) for _ in range(14)]
        + [_timing("http_request", success=False, status=None) for _ in range(2)]
        + [_timing("visit", success=True, status=200) for _ in range(10)]
        + [_timing("visit", success=False, status=None) for _ in range(1)]
    )
    telemetry = _TelemetryWithTimings(timings)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["infra"]["failed"] is False
    # Visibility is preserved even though the boolean gate no longer fires.
    assert obs["infra"]["failure_count"] == 3
    assert obs["infra"]["ops"] == ["http_request", "visit"]


def test_summarize_observability_failed_on_total_outage_for_an_op():
    """0/5 successes on an op is a total outage even though the failure count (5) alone
    wouldn't necessarily clear a naive rate threshold on a tiny sample elsewhere."""
    timings = [_timing("search_query", success=False, status=503) for _ in range(5)]
    telemetry = _TelemetryWithTimings(timings)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["infra"]["failed"] is True
    assert obs["infra"]["failure_count"] == 5


def test_summarize_observability_not_failed_on_exact_half_failure_rate():
    """8/16 http_request successes (rate == 0.5) is the worst observed rate in the
    2026-08-23 80-cell paid_wide_sweep replay among cells that still produced valid scores;
    it sits exactly at the threshold and must NOT flag (threshold is strictly-greater-than)."""
    timings = (
        [_timing("http_request", success=True, status=200) for _ in range(8)]
        + [_timing("http_request", success=False, status=500) for _ in range(8)]
    )
    telemetry = _TelemetryWithTimings(timings)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["infra"]["failed"] is False
    assert obs["infra"]["failure_count"] == 8


def test_summarize_observability_failed_just_above_half_failure_rate():
    """9/17 failures (rate > 0.5) crosses the threshold and must flag."""
    timings = (
        [_timing("http_request", success=True, status=200) for _ in range(8)]
        + [_timing("http_request", success=False, status=500) for _ in range(9)]
    )
    telemetry = _TelemetryWithTimings(timings)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["infra"]["failed"] is True


def test_summarize_observability_lone_infra_flag_among_many_successes_is_not_failed():
    """The per-timing classification for payload.infra_failed=True is unchanged (still infra,
    see test_is_infra_timing_honors_explicit_infra_flag_from_connector_llm) and is still
    counted in failure_count/ops. But the cell-level `failed` gate is now severity-based: one
    flagged llm_call out of 21 (rate ~0.048) is the same kind of transient hiccup a single
    failed http_request would be, so it must NOT alone flip the cell to infra-failed."""
    timings = (
        [_timing("llm_call", success=True) for _ in range(20)]
        + [_timing("llm_call", success=False, infra_failed=True, error="status 402")]
    )
    telemetry = _TelemetryWithTimings(timings)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["infra"]["failed"] is False
    assert obs["infra"]["failure_count"] == 1
    assert obs["infra"]["ops"] == ["llm_call"]


def test_summarize_observability_sole_infra_flagged_call_is_total_outage():
    """When the flagged call IS the only attempt at that op, it's a total outage (0
    successes) and still fails — this is the unchanged single-attempt case."""
    timings = [_timing("llm_call", success=False, infra_failed=True, error="status 402")]
    telemetry = _TelemetryWithTimings(timings)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["infra"]["failed"] is True
    assert obs["infra"]["failure_count"] == 1


def test_summarize_observability_no_failure_when_no_timings():
    telemetry = _TelemetryWithTimings([])
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert obs["infra"]["failed"] is False
    assert obs["infra"]["failure_count"] == 0
    assert obs["infra"]["ops"] == []


def test_summarize_observability_infra_exposes_failure_rate_for_consumer_judgement():
    """Full visibility: failure_count and ops report every classified failure unchanged;
    a rate field exposes the underlying severity so consumers can apply their own cutoff."""
    timings = (
        [_timing("http_request", success=True, status=200) for _ in range(14)]
        + [_timing("http_request", success=False, status=None) for _ in range(2)]
    )
    telemetry = _TelemetryWithTimings(timings)
    obs = summarize_observability({"output": {"final_deliverable": "x"}}, telemetry)
    assert "rates" in obs["infra"]
    assert obs["infra"]["rates"]["http_request"] == pytest.approx(2 / 16)


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
