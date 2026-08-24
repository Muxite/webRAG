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

4. Concurrency safety: two sibling ``init_search_api()`` calls racing on the SAME connector
   instance (the live graph-engine fan-out shape) must not permanently wipe telemetry for the
   rest of the cell — see ``test_concurrent_init_search_api_does_not_permanently_wipe_telemetry``.

No live network is used; ``request()`` is monkeypatched per connector for tests 1-3. Test 4
drives real concurrency through ``ConnectorHttp.request``'s actual body via a fake aiohttp
session, since the race it targets only manifests across a genuine interleaved ``await``.
"""
import asyncio
import time

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
    """Stand in a failing health probe. Also asserts suppression is requested via the
    per-call ``suppress_timing`` kwarg — proving `_probe_search_init` asks
    `ConnectorHttp.request` to suppress the underlying `http_request` signal for this one
    call, rather than mutating shared connector state (the old, racy approach)."""
    async def fake_request(method, url, retries=2, **kwargs):
        assert kwargs.get("suppress_timing") is True, "probe call must request timing suppression"
        assert connector._telemetry is not None, "telemetry must stay attached during the probe"
        return RequestResult(status=status, error=True, data="boom")

    monkeypatch.setattr(connector, "request", fake_request)


def _wire_healthy_probe(monkeypatch, connector):
    async def fake_request(method, url, retries=2, **kwargs):
        assert kwargs.get("suppress_timing") is True, "probe call must request timing suppression"
        assert connector._telemetry is not None, "telemetry must stay attached during the probe"
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


# ---------------------------------------------------------------------------------------
# 4. Concurrency race: two sibling init_search_api() calls on the SAME connector instance
#    must not permanently disable telemetry. See module docstring point 4.
# ---------------------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status=200, json_data=None):
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self._json = json_data if json_data is not None else {}

    async def json(self):
        return self._json

    async def text(self):
        return ""


class _FakeRequestCM:
    """Stands in for aiohttp's ``session.request(...)`` async context manager. The
    ``await asyncio.sleep`` inside ``__aenter__`` is the actual interleave point: it yields
    control back to the event loop mid-``ConnectorHttp.request`` body, so two concurrent
    ``init_search_api()`` calls genuinely race inside ``_probe_search_init`` rather than
    running to completion one at a time."""

    def __init__(self, delay: float):
        self._delay = delay

    async def __aenter__(self):
        await asyncio.sleep(self._delay)
        return _FakeResponse(status=200, json_data={"web": {"results": []}, "organic": []})

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in wired directly onto the connector, so calls
    run through the REAL `ConnectorHttp.request` body (including its internal `await`)
    instead of a single-coroutine mock that can't reproduce an interleaving race."""

    def __init__(self, delay: float = 0.02):
        self.closed = False
        self._delay = delay

    def request(self, method, url, timeout=None, **kwargs):
        return _FakeRequestCM(self._delay)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_concurrent_init_search_api_does_not_permanently_wipe_telemetry():
    """Reproduces the live shape: sibling graph leaves share one search connector and both
    reach `init_search_api()` before `search_api_ready` flips True (it only flips AFTER the
    probe returns), so both enter `_probe_search_init` on the SAME instance concurrently.

    An earlier `_probe_search_init` implementation suppressed the probe's own `http_request`
    timing by save/clear/restore-ing `self._telemetry` — shared MUTABLE instance state across
    an `await`. Under real interleaving: A saves T, clears to None, awaits; B saves
    saved_telemetry_B = None (A's clear is visible), awaits; A's `finally` restores T
    correctly; B's `finally` then sets `self._telemetry = None` — permanently wiping telemetry
    for the rest of the cell, with no exception raised (ConnectorBase quietly no-ops on
    `_telemetry is None`).

    The fix threads a per-call `suppress_timing` kwarg through `ConnectorHttp.request` instead
    (see connector_http.py), which has no shared state to race.
    """
    cs = ConnectorSearch(ConnectorConfig())
    cs.search_api_key = "k"
    telemetry = _RecordingTelemetry()
    cs.set_telemetry(telemetry)
    cs.session = _FakeSession(delay=0.02)

    results = await asyncio.gather(cs.init_search_api(), cs.init_search_api())

    assert all(results), f"both concurrent inits should succeed: {results}"
    assert cs._telemetry is telemetry, "telemetry must not be wiped by a racing sibling call"

    # Functional check, not just attribute presence: telemetry recorded AFTER both concurrent
    # calls finish must actually land — proves it is live, not merely non-None by accident.
    before = len(telemetry.timings)
    cs._record_timing(name="post_race_probe", started_at=time.perf_counter(), success=True, payload={})
    assert len(telemetry.timings) == before + 1, "telemetry must still be functional after the race"
