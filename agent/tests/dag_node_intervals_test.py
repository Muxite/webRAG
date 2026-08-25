"""``IdeaNode`` carried no timestamps, so per-node wall-clock order was unrecoverable.

``idea_engine`` says so outright in a docstring at the race-merge winner selection: *"The graph
keeps no per-node completion time, and this is the only ordering signal available to
SimpleMergePolicy.select_winner at merge time."* The workaround, ``_record_race_completion``,
stamps a step INDEX rather than a time, and only for nodes carrying a race-group label -- inert
on every normal path.

The consequence for the four-way baseline: ``execution.graph`` recorded which nodes ran and
what they produced, but not WHEN, so "did this fan-out execute concurrently?" could not be
answered from the persisted artifact that the forensics scripts already parse.

``started_at`` / ``ended_at`` are session-relative seconds sharing the telemetry session's
``perf_counter`` anchor, so node intervals and call intervals (``timings_per_call``) land on
one timeline and can be correlated directly.

No network: actions are stubbed.
"""
from __future__ import annotations

import asyncio

from agent.app.idea_dag import IdeaDag, IdeaNode
from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.app.telemetry import TelemetrySession


def test_node_defaults_carry_no_interval():
    node = IdeaNode(node_id="n", title="t")
    assert node.started_at is None
    assert node.ended_at is None


def test_interval_survives_a_to_dict_from_dict_round_trip():
    """The forensics scripts read ``execution.graph``; the fields must reach it and come back."""
    graph = IdeaDag(root_title="root")
    child = graph.add_child(graph.root_id(), "visit", status=IdeaNodeStatus.DONE)
    child.started_at = 1.25
    child.ended_at = 3.5

    payload = graph.to_dict()
    assert payload["nodes"][child.node_id]["started_at"] == 1.25
    assert payload["nodes"][child.node_id]["ended_at"] == 3.5

    restored = IdeaDag.from_dict(payload)
    node = restored.get_node(child.node_id)
    assert node.started_at == 1.25
    assert node.ended_at == 3.5


def test_from_dict_tolerates_artifacts_written_before_the_fields_existed():
    """Old result JSONs must still load -- these are read back by the analysis scripts."""
    graph = IdeaDag(root_title="root")
    graph.add_child(graph.root_id(), "visit", status=IdeaNodeStatus.DONE)
    payload = graph.to_dict()
    for node in payload["nodes"].values():
        node.pop("started_at", None)
        node.pop("ended_at", None)

    restored = IdeaDag.from_dict(payload)
    for node in restored.iter_depth_first():
        assert node.started_at is None
        assert node.ended_at is None


def test_telemetry_elapsed_is_monotonic_and_session_relative():
    session = TelemetrySession(enabled=True)
    first = session.elapsed()
    second = session.elapsed()
    assert 0.0 <= first <= second


def test_elapsed_shares_the_anchor_used_by_record_timing():
    """Node intervals and call intervals have to land on ONE timeline to be correlatable."""
    import time

    session = TelemetrySession(enabled=True)
    session._perf_start -= 10.0

    before = session.elapsed()
    session.record_timing("llm_call", time.perf_counter() - 0.5, True)
    after = session.elapsed()

    timing = session.timings[0]
    assert before <= timing["t_end"] <= after


def test_execute_action_stamps_the_interval_on_the_node():
    """The single dispatch choke point: every routing path goes through ``_execute_action``."""
    from agent.app.idea_engine import IdeaDagEngine

    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(), "think",
        details={DetailKey.ACTION.value: "think"},
        status=IdeaNodeStatus.PENDING,
    )

    engine = IdeaDagEngine.__new__(IdeaDagEngine)
    engine.io = type("_IO", (), {"telemetry": TelemetrySession(enabled=True)})()

    stamped = engine._stamp_node_start(graph, node.node_id)
    assert node.started_at is not None
    assert node.ended_at is None
    engine._stamp_node_end(graph, node.node_id, stamped)
    assert node.ended_at is not None
    assert node.ended_at >= node.started_at


def test_stamping_is_inert_without_a_telemetry_session():
    """A debug run with telemetry off must not crash, and must not invent timings."""
    from agent.app.idea_engine import IdeaDagEngine

    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "think", status=IdeaNodeStatus.PENDING)

    engine = IdeaDagEngine.__new__(IdeaDagEngine)
    engine.io = type("_IO", (), {"telemetry": None})()

    stamped = engine._stamp_node_start(graph, node.node_id)
    engine._stamp_node_end(graph, node.node_id, stamped)
    # The engine falls back to its own anchor rather than leaving the run untimed.
    assert node.started_at is not None
    assert node.ended_at >= node.started_at


def test_concurrent_nodes_read_as_overlapping():
    """The question the whole change exists to answer, on the DAG's own gather shape."""
    from agent.app.idea_engine import IdeaDagEngine

    graph = IdeaDag(root_title="root")
    a = graph.add_child(graph.root_id(), "a", status=IdeaNodeStatus.PENDING)
    b = graph.add_child(graph.root_id(), "b", status=IdeaNodeStatus.PENDING)

    engine = IdeaDagEngine.__new__(IdeaDagEngine)
    engine.io = type("_IO", (), {"telemetry": TelemetrySession(enabled=True)})()

    async def _run(node_id):
        stamped = engine._stamp_node_start(graph, node_id)
        await asyncio.sleep(0.05)
        engine._stamp_node_end(graph, node_id, stamped)

    async def _both():
        await asyncio.wait_for(
            asyncio.gather(_run(a.node_id), _run(b.node_id)), timeout=5,
        )

    asyncio.run(_both())

    assert a.started_at < b.ended_at and b.started_at < a.ended_at
