"""Regression tests for the search-init infra-visibility gap.

``init_search_api()`` in all three search backends (Brave/``ConnectorSearch``, Serper, SearXNG)
used to emit NO timing at all on a failed health probe — the identical "chroma_init before
yesterday's fix" invisibility pattern. It was ALSO partially masked in a worse way: the probe
goes through ``ConnectorHttp.request``, which unconditionally emits a generic ``http_request``
timing, so a dead search backend showed up diluted into that bucket instead of being named.

This locks in the fix (see ``agent/app/connector_search.py``'s ``_probe_search_init`` /
``_record_search_init``):

1. A dead backend now emits a named ``search_init`` timing, stamped ``infra_failed`` and
   classified as infra by ``agent.app.testing.utils._is_infra_timing`` — for ALL three backends.
2. A healthy backend does not trip the infra classifier, AND still emits its own success-side
   ``search_init`` timing (chosen fix for the zero-success regression: a failure-only signal
   would make this op's ``success == 0`` permanently true in the severity gate, exactly the bug
   concurrently being fixed in ``connector_chroma.py``'s ``chroma_init``).
3. The double-counting decision — suppress the probe's own generic ``http_request`` timing for
   this one call, emitting ONLY the named ``search_init`` signal — is asserted directly: a dead
   backend must produce a ``search_init`` failure and ZERO ``http_request`` timings.

No live network is used; ``request()`` is monkeypatched per connector.
"""
import pytest

from agent.app.connector_search import ConnectorSearch
from agent.app.connector_search_serper import ConnectorSearchSerper
from agent.app.connector_search_searxng import ConnectorSearchXNG
from agent.app.testing.utils import _is_infra_timing
from shared.connector_config import ConnectorConfig
from shared.request_result import RequestResult


class _RecordingTelemetry:
    def __init__(self):
        self.timings = []

    def record_timing(self, **kwargs):
        self.timings.append(kwargs)

    def record_event(self, event, payload):
        pass

    def record_io(self, **kwargs):
        pass


def _timings_named(telemetry, name):
    return [t for t in telemetry.timings if t["name"] == name]


def _last_timing(telemetry, name):
    matches = _timings_named(telemetry, name)
    assert matches, f"no timing named {name!r} recorded; got {telemetry.timings}"
    return matches[-1]


def _make_brave() -> ConnectorSearch:
    cs = ConnectorSearch(ConnectorConfig())
    cs.search_api_key = "k"
    return cs


def _make_serper() -> ConnectorSearchSerper:
    cs = ConnectorSearchSerper(ConnectorConfig())
    cs.search_api_key = "k"
    return cs


def _make_searxng() -> ConnectorSearchXNG:
    return ConnectorSearchXNG(ConnectorConfig())


BACKENDS = [
    ("brave", _make_brave),
    ("serper", _make_serper),
    ("searxng", _make_searxng),
]


def _wire_dead_probe(monkeypatch, connector, status=503):
    """Stand in a failing health probe. Also asserts telemetry is genuinely detached during
    the probe call — proving `_probe_search_init` really suppresses the underlying
    `http_request` signal rather than merely not-emitting-it-in-this-test."""
    async def fake_request(method, url, retries=2, **kwargs):
        assert connector._telemetry is None, "probe call must run with telemetry detached"
        return RequestResult(status=status, error=True, data="boom")

    monkeypatch.setattr(connector, "request", fake_request)


def _wire_healthy_probe(monkeypatch, connector):
    async def fake_request(method, url, retries=2, **kwargs):
        assert connector._telemetry is None, "probe call must run with telemetry detached"
        return RequestResult(status=200, error=False, data={"web": {"results": []}, "organic": []})

    monkeypatch.setattr(connector, "request", fake_request)


# ---------------------------------------------------------------------------------------
# 1. Dead backend -> named, infra-tagged search_init signal (per backend).
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,make", BACKENDS)
@pytest.mark.asyncio
async def test_dead_backend_emits_named_infra_search_init(monkeypatch, name, make):
    connector = make()
    telemetry = _RecordingTelemetry()
    connector.set_telemetry(telemetry)
    _wire_dead_probe(monkeypatch, connector)

    ok = await connector.init_search_api()

    assert ok is False
    timing = _last_timing(telemetry, "search_init")
    assert timing["success"] is False
    assert timing["payload"]["provider"] == name
    assert timing["payload"]["infra_failed"] is True
    assert _is_infra_timing(timing) is True


# ---------------------------------------------------------------------------------------
# 2. Healthy backend -> no infra failure, but DOES emit a success-side search_init timing
#    (point 3 in the task: avoids the failure-only zero-success regression).
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,make", BACKENDS)
@pytest.mark.asyncio
async def test_healthy_backend_no_infra_signal(monkeypatch, name, make):
    connector = make()
    telemetry = _RecordingTelemetry()
    connector.set_telemetry(telemetry)
    _wire_healthy_probe(monkeypatch, connector)

    ok = await connector.init_search_api()

    assert ok is True
    assert not any(_is_infra_timing(t) for t in telemetry.timings)


@pytest.mark.parametrize("name,make", BACKENDS)
@pytest.mark.asyncio
async def test_healthy_backend_emits_success_side_search_init_timing(monkeypatch, name, make):
    """Point 3 (zero-success asymmetry): a success-only-emits-nothing design would leave the
    `search_init` bucket permanently at success==0, tripping the severity gate's "op produced
    zero successes outright" branch on a SINGLE transient failure. Success must be recorded."""
    connector = make()
    telemetry = _RecordingTelemetry()
    connector.set_telemetry(telemetry)
    _wire_healthy_probe(monkeypatch, connector)

    ok = await connector.init_search_api()

    assert ok is True
    timing = _last_timing(telemetry, "search_init")
    assert timing["success"] is True
    assert timing["payload"]["provider"] == name
    assert "infra_failed" not in timing["payload"]


@pytest.mark.parametrize("name,make", BACKENDS)
@pytest.mark.asyncio
async def test_one_transient_failure_then_recovery_avoids_permanent_zero_success(
    monkeypatch, name, make
):
    """Reproduces the exact scenario the chroma_init regression describes: a transient init
    failure followed by a successful retry-on-next-call. With a symmetric success timing, the
    per-op bucket ends with success=1/total=2 (rate 0.5, not flagged) instead of a permanently
    unreachable success==0."""
    connector = make()
    telemetry = _RecordingTelemetry()
    connector.set_telemetry(telemetry)

    _wire_dead_probe(monkeypatch, connector)
    first_ok = await connector.init_search_api()
    assert first_ok is False

    _wire_healthy_probe(monkeypatch, connector)
    second_ok = await connector.init_search_api()
    assert second_ok is True

    search_init_timings = _timings_named(telemetry, "search_init")
    assert len(search_init_timings) == 2
    successes = sum(1 for t in search_init_timings if t["success"])
    assert successes == 1, "success side must be represented, not permanently zero"


# ---------------------------------------------------------------------------------------
# 3. Double-counting decision: the probe's own http_request telemetry is suppressed — a dead
#    backend must produce the named search_init failure and NO http_request timing at all.
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,make", BACKENDS)
@pytest.mark.asyncio
async def test_dead_backend_does_not_double_count_http_request(monkeypatch, name, make):
    connector = make()
    telemetry = _RecordingTelemetry()
    connector.set_telemetry(telemetry)
    _wire_dead_probe(monkeypatch, connector)

    ok = await connector.init_search_api()

    assert ok is False
    assert _timings_named(telemetry, "search_init"), "expected the named signal to be recorded"
    assert not _timings_named(telemetry, "http_request"), (
        "probe's own http_request timing must be suppressed to avoid double-counting one "
        "outage as two independent severity-gate signals"
    )


# ---------------------------------------------------------------------------------------
# Caller/config error (missing API key) is NOT stamped infra — mirrors is_infra_llm_failure's
# transport-vs-caller-error distinction; a bad/missing key is not a transport outage.
# ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_api_key_emits_no_search_init_timing():
    """A missing key never reaches the probe at all — it's a caller/config error, loudly
    logged already, and must not be conflated with a genuine transport outage."""
    cs = ConnectorSearch(ConnectorConfig())
    cs.search_api_key = None
    telemetry = _RecordingTelemetry()
    cs.set_telemetry(telemetry)

    ok = await cs.init_search_api()

    assert ok is False
    assert not _timings_named(telemetry, "search_init")


@pytest.mark.asyncio
async def test_permanent_http_error_not_stamped_infra():
    """A 401 (bad/expired key) is a PERMANENT_ERROR_CODES caller error, not infra — reusing
    is_infra_llm_failure's classification distinguishes this from a genuine 5xx/timeout outage."""
    cs = ConnectorSearch(ConnectorConfig())
    cs.search_api_key = "k"
    telemetry = _RecordingTelemetry()
    cs.set_telemetry(telemetry)

    async def fake_request(method, url, retries=2, **kwargs):
        return RequestResult(status=401, error=True, data="unauthorized")

    cs.request = fake_request  # simple attribute swap; no monkeypatch fixture needed here

    ok = await cs.init_search_api()

    assert ok is False
    timing = _last_timing(telemetry, "search_init")
    assert timing["payload"]["infra_failed"] is False
    assert _is_infra_timing(timing) is False
