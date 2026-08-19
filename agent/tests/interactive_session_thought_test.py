"""Tests for the ThoughtBuffer/ThoughtLog wiring in interactive/session.py.

Uses a stub engine (no LLM / connectors) and a stub telemetry object; no network,
no live LLM. Mirrors the fakes in interactive_session_test.py.
"""
from __future__ import annotations

from typing import List, Optional

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.interactive.controller import Action, Cmd
from agent.app.interactive.session import DebugSession


class _FakeTelemetry:
    """Minimal duck-type of TelemetrySession: the two lists ThoughtBuffer reads."""

    def __init__(self):
        self.decisions: List[dict] = []
        self.events: List[dict] = []


class _FakeIO:
    def __init__(self, telemetry=None):
        self.telemetry = telemetry


class _FakeEngine:
    """Scripted async step(); appends a decision to telemetry on each call."""

    def __init__(self, io=None, script: Optional[List[Optional[str]]] = None, echo: Optional[str] = None):
        self.io = io
        self._script = list(script or [])
        self._echo = echo
        self.calls: List[str] = []

    async def step(self, graph, node_id, step_index):
        self.calls.append(node_id)
        if self.io and self.io.telemetry:
            self.io.telemetry.decisions.append({"stage": "expansion", "chosen": node_id})
        if self._echo is not None:
            return self._echo
        if self._script:
            return self._script.pop(0)
        return None


class _ScriptedController:
    def __init__(self, cmds: List[Cmd]):
        self._cmds = list(cmds)

    def ask(self, label: str = "") -> Cmd:
        return self._cmds.pop(0) if self._cmds else Cmd(Action.QUIT)

    def read_multiline(self, prompt: str = "", cont: str = "") -> str:
        return ""

    def read_line(self, prompt: str = "") -> str:
        return ""


def _leaf_graph():
    """Returns (graph, leaf_id, leaf_node). ``add_child`` returns the IdeaNode itself;
    engine/buffer calls want the string node_id, while `_pause` wants the node."""
    g = IdeaDag(root_title="mandate", root_details={"mandate": "mandate"})
    leaf_node = g.add_child(
        g.root_id(),
        title="do a thing",
        details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
        status=IdeaNodeStatus.PENDING,
    )
    return g, leaf_node.node_id, leaf_node


def _session(graph, engine, ctrl, max_steps=10):
    sess = DebugSession(engine=engine, graph=graph, ctrl=ctrl, max_steps=max_steps)
    sess._out = lambda *_a, **_k: None
    return sess


# ---------------------------------------------------------------- buffer construction


def test_thought_buffer_built_from_engine_telemetry():
    g, leaf_id, leaf = _leaf_graph()
    telem = _FakeTelemetry()
    engine = _FakeEngine(io=_FakeIO(telemetry=telem))
    sess = _session(g, engine, _ScriptedController([]))
    assert sess._thoughts is not None


def test_thought_buffer_tolerates_missing_io():
    g, leaf_id, leaf = _leaf_graph()

    class _NoIoEngine:
        async def step(self, graph, node_id, step_index):
            return None

    sess = _session(g, _NoIoEngine(), _ScriptedController([]))
    assert sess._thoughts is not None
    # Should not raise even though there's no io/telemetry at all.
    latest = sess._thoughts.latest()
    assert latest is None


def test_thought_buffer_tolerates_none_telemetry():
    g, leaf_id, leaf = _leaf_graph()
    engine = _FakeEngine(io=_FakeIO(telemetry=None))
    sess = _session(g, engine, _ScriptedController([]))
    assert sess._thoughts is not None


# ---------------------------------------------------------------- capture/collect around a step


@pytest.mark.asyncio
async def test_engine_step_captures_and_collects_thought():
    g, leaf_id, leaf = _leaf_graph()
    telem = _FakeTelemetry()
    engine = _FakeEngine(io=_FakeIO(telemetry=telem), echo=None)
    sess = _session(g, engine, _ScriptedController([]))

    await sess._engine_step(leaf_id)

    latest = sess._thoughts.latest()
    assert latest is not None
    assert latest.node_id == leaf_id
    assert len(latest.decisions) == 1
    assert latest.decisions[0]["chosen"] == leaf_id


@pytest.mark.asyncio
async def test_engine_step_collects_thought_even_on_failure():
    g, leaf_id, leaf = _leaf_graph()

    class _FailingEngine:
        io = None

        async def step(self, graph, node_id, step_index):
            raise RuntimeError("boom")

    sess = _session(g, _FailingEngine(), _ScriptedController([]))
    result = await sess._engine_step(leaf_id)
    assert result is None
    # A Thought (possibly empty) must still have been collected for the failed step.
    latest = sess._thoughts.latest()
    assert latest is not None


@pytest.mark.asyncio
async def test_autorun_captures_and_collects_per_step():
    g, leaf_id, leaf = _leaf_graph()
    telem = _FakeTelemetry()
    engine = _FakeEngine(io=_FakeIO(telemetry=telem), script=[None])
    sess = _session(g, engine, _ScriptedController([]))

    await sess._autorun(leaf_id, depth=1)

    assert engine.calls == [leaf_id]
    latest = sess._thoughts.latest()
    assert latest is not None
    assert latest.node_id == leaf_id


# ---------------------------------------------------------------- THOUGHT / WHY commands


def test_thought_command_renders_without_raising_and_continues_loop():
    g, leaf_id, leaf = _leaf_graph()
    telem = _FakeTelemetry()
    engine = _FakeEngine(io=_FakeIO(telemetry=telem))
    ctrl = _ScriptedController([Cmd(Action.THOUGHT), Cmd(Action.STEP)])
    sess = _session(g, engine, ctrl)

    action = sess._pause(leaf)
    assert action is Action.STEP  # THOUGHT was consumed, loop continued to STEP


def test_thought_command_with_no_thought_yet_prints_explicit_line():
    g, leaf_id, leaf = _leaf_graph()
    engine = _FakeEngine(io=_FakeIO(telemetry=_FakeTelemetry()))
    ctrl = _ScriptedController([Cmd(Action.THOUGHT), Cmd(Action.QUIT)])
    sess = _session(g, engine, ctrl)

    printed = []
    sess._out = printed.append

    action = sess._pause(leaf)
    assert action is None
    assert any("no thought" in line for line in printed)


def test_why_command_renders_without_raising_and_continues_loop():
    g, leaf_id, leaf = _leaf_graph()
    telem = _FakeTelemetry()
    engine = _FakeEngine(io=_FakeIO(telemetry=telem))
    ctrl = _ScriptedController([Cmd(Action.WHY), Cmd(Action.STEP)])
    sess = _session(g, engine, ctrl)

    action = sess._pause(leaf)
    assert action is Action.STEP


def test_why_command_with_no_decisions_prints_explicit_line():
    g, leaf_id, leaf = _leaf_graph()
    engine = _FakeEngine(io=_FakeIO(telemetry=_FakeTelemetry()))
    ctrl = _ScriptedController([Cmd(Action.WHY), Cmd(Action.QUIT)])
    sess = _session(g, engine, ctrl)

    printed = []
    sess._out = printed.append

    action = sess._pause(leaf)
    assert action is None
    assert any("no decisions" in line for line in printed)


def test_why_command_after_step_shows_recorded_decision():
    g, leaf_id, leaf = _leaf_graph()
    telem = _FakeTelemetry()
    engine = _FakeEngine(io=_FakeIO(telemetry=telem))
    sess = _session(g, engine, _ScriptedController([]))

    # Simulate a completed step's thought already being in the buffer.
    sess._thoughts.capture(1, leaf_id)
    telem.decisions.append({"stage": "expansion", "chosen": leaf_id, "rationale": "best"})
    sess._thoughts.collect(1, leaf_id)

    printed = []
    sess._out = printed.append
    sess._handle_why()
    assert any("expansion" in line for line in printed)


# ---------------------------------------------------------------- thought_log wiring


@pytest.mark.asyncio
async def test_thought_log_receives_collected_thoughts(monkeypatch):
    g, leaf_id, leaf = _leaf_graph()
    telem = _FakeTelemetry()
    engine = _FakeEngine(io=_FakeIO(telemetry=telem))
    sess = _session(g, engine, _ScriptedController([]))

    written = []

    class _StubLog:
        def write(self, thought):
            written.append(thought)

    sess._thought_log = _StubLog()
    await sess._engine_step(leaf_id)

    assert len(written) == 1
    assert written[0].node_id == leaf_id


def test_no_thought_log_by_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("IDEA_THOUGHT_LOG", raising=False)
    g, leaf_id, leaf = _leaf_graph()
    engine = _FakeEngine(io=_FakeIO(telemetry=_FakeTelemetry()))
    sess = _session(g, engine, _ScriptedController([]))
    assert sess._thought_log is None
