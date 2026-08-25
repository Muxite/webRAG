"""Nothing persisted anywhere carried a call START time, so concurrency was unprovable.

``TelemetrySession.record_timing`` accepted ``started_at`` (a ``perf_counter`` reading) and
discarded it, keeping only ``duration``. ``timings_per_call`` in the result JSON was therefore
``{name, duration, success}`` -- three fields that cannot answer "did these two calls overlap?".

The forensics pass that produced the four-way baseline had to infer parallelism from
suspiciously-equal durations (two ``llm_call`` entries at 9.5504 / 9.5506), which is
circumstantial at best and silently wrong for the ``langgraph`` arm, whose ``connector_io``
events are all synthesised at end-of-run.

Recording a session-relative ``[t_start, t_end]`` pair makes overlap a direct read off the
already-persisted result JSON -- for every arm, including those that emit an empty graph, and
with no new artifact files. Offsets are relative to the session's own ``perf_counter`` anchor
rather than wall clock, so they stay monotonic and immune to clock skew.

No network: intervals are constructed from explicit ``perf_counter`` readings (one 50ms
sleep separates the deliberately-sequential pair, which needs real elapsed time to exist).
"""
from __future__ import annotations

import time

from agent.app.telemetry import TelemetrySession
from agent.app.testing.utils import summarize_observability


def _session(age_seconds: float = 0.0) -> TelemetrySession:
    """A session, optionally back-dated so calls can be placed inside its lifetime.

    ``record_timing`` floors offsets at the session origin, so a synthetic ``started_at``
    earlier than the session start would clamp to 0.0 and make interval assertions vacuous.
    Ageing the anchor gives the tests a real window to place calls in, without sleeping.
    """
    session = TelemetrySession(enabled=True, mandate="m", correlation_id="c")
    session._perf_start -= age_seconds
    return session


def test_record_timing_keeps_the_interval_not_just_the_duration():
    session = _session()
    started = time.perf_counter()
    session.record_timing("llm_call", started, True)

    entry = session.timings[0]
    assert "t_start" in entry and "t_end" in entry
    assert entry["t_end"] >= entry["t_start"] >= 0.0
    # The interval must agree with the duration it replaces.
    assert abs((entry["t_end"] - entry["t_start"]) - entry["duration"]) < 1e-6


def test_overlapping_calls_are_detectable_as_overlapping():
    """Two calls dispatched under one ``asyncio.gather`` -- the shape the DAG's fan-out uses."""
    session = _session(age_seconds=10.0)
    now = time.perf_counter()
    session.record_timing("llm_call", now - 0.5, True)
    session.record_timing("llm_call", now - 0.4, True)

    a, b = session.timings
    assert a["t_start"] < b["t_end"] and b["t_start"] < a["t_end"]


def test_sequential_calls_are_not_detectable_as_overlapping():
    """The negative control: a chain must not read as concurrent."""
    session = _session(age_seconds=10.0)
    # ``record_timing`` always stamps ``t_end`` at call time, so a genuinely sequential pair
    # needs real elapsed time between the two records -- two calls recorded in the same
    # microsecond share an end instant no matter what ``started_at`` claims.
    session.record_timing("llm_call", time.perf_counter() - 5.0, True)
    time.sleep(0.05)
    session.record_timing("llm_call", time.perf_counter() - 0.01, True)

    a, b = session.timings
    assert b["t_start"] > a["t_end"]


def test_intervals_reach_timings_per_call_in_the_result_json():
    """The whole point: the numbers must survive into the persisted observability block."""
    # Real width matters: an interval narrower than the 4-decimal rounding in
    # ``timings_per_call`` collapses to a point and makes the overlap check vacuous.
    session = _session(age_seconds=10.0)
    now = time.perf_counter()
    session.record_timing("search_query", now - 0.5, True)
    session.record_timing("search_query", now - 0.4, True)

    summary = summarize_observability({}, session)
    per_call = summary["timings_per_call"]
    assert len(per_call) == 2
    for call in per_call:
        assert "t_start" in call and "t_end" in call
    assert per_call[0]["t_start"] < per_call[1]["t_end"]
    assert per_call[1]["t_start"] < per_call[0]["t_end"]


def test_aggregate_timing_fields_are_unchanged():
    """The added fields must not disturb what existing analysis scripts already read."""
    session = _session()
    session.record_timing("llm_call", time.perf_counter(), True)
    summary = summarize_observability({}, session)

    call = summary["timings_per_call"][0]
    assert set(call) >= {"name", "duration", "success"}
    assert summary["timings"]["llm_call"]["count"] == 1
    assert summary["timings"]["llm_call"]["success_count"] == 1


def test_a_disabled_session_records_nothing():
    session = TelemetrySession(enabled=False)
    session.record_timing("llm_call", time.perf_counter(), True)
    assert session.timings == []
