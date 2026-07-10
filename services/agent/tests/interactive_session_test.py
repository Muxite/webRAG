"""Tests for the agent-debug stepping session (interactive/session.py).

Uses a stub engine (no LLM / connectors) and a scripted controller to drive the
pause command loop and auto-run termination deterministically.
"""
from __future__ import annotations

from typing import List, Optional

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.interactive.controller import Action, Cmd, Controller
from agent.app.interactive.session import DebugSession


class _FakeEngine:
    """Minimal engine: scripted async step(); records calls."""

    def __init__(self, script: Optional[List[Optional[str]]] = None, echo: Optional[str] = None):
        self._script = list(script or [])
        self._echo = echo  # if set, always return this id
        self.calls: List[str] = []

    async def step(self, graph, node_id, step_index):
        self.calls.append(node_id)
        if self._echo is not None:
            return self._echo
        if self._script:
            return self._script.pop(0)
        return None


class _ScriptedController:
    """Feeds pre-baked Cmd/edit/feedback responses to DebugSession."""

    def __init__(self, cmds: List[Cmd], multiline: Optional[List[str]] = None, lines: Optional[List[str]] = None):
        self._cmds = list(cmds)
        self._multiline = list(multiline or [])
        self._lines = list(lines or [])

    def ask(self, label: str = "") -> Cmd:
        return self._cmds.pop(0) if self._cmds else Cmd(Action.QUIT)

    def read_multiline(self, prompt: str = "", cont: str = "") -> str:
        return self._multiline.pop(0) if self._multiline else ""

    def read_line(self, prompt: str = "") -> str:
        return self._lines.pop(0) if self._lines else ""


def _leaf_graph():
    g = IdeaDag(root_title="mandate", root_details={"mandate": "mandate"})
    leaf = g.add_child(
        g.root_id(),
        title="do a thing",
        details={DetailKey.ACTION.value: IdeaActionType.THINK.value},
        status=IdeaNodeStatus.PENDING,
    )
    return g, leaf


def _session(graph, ctrl, max_steps=10):
    sess = DebugSession(engine=_FakeEngine(), graph=graph, ctrl=ctrl, max_steps=max_steps)
    sess._out = lambda *_a, **_k: None  # silence output
    return sess


# ---------------------------------------------------------------- pause / resume


def test_pause_returns_step():
    g, leaf = _leaf_graph()
    sess = _session(g, _ScriptedController([Cmd(Action.STEP)]))
    assert sess._pause(leaf) is Action.STEP


def test_pause_returns_next():
    g, leaf = _leaf_graph()
    sess = _session(g, _ScriptedController([Cmd(Action.NEXT)]))
    assert sess._pause(leaf) is Action.NEXT


def test_pause_quit_returns_none_and_sets_flag():
    g, leaf = _leaf_graph()
    sess = _session(g, _ScriptedController([Cmd(Action.QUIT)]))
    assert sess._pause(leaf) is None
    assert sess._quit is True


def test_pause_loops_on_observation_commands_then_steps():
    g, leaf = _leaf_graph()
    ctrl = _ScriptedController(
        [Cmd(Action.INFO), Cmd(Action.PRINT), Cmd(Action.LIST), Cmd(Action.GRAPH), Cmd(Action.HELP), Cmd(Action.STEP)]
    )
    sess = _session(g, ctrl)
    # None of the observation commands advance; the final STEP is returned.
    assert sess._pause(leaf) is Action.STEP


# ---------------------------------------------------------------------- edit


def test_edit_overrides_resolved_content_and_does_not_advance():
    g, leaf = _leaf_graph()
    ctrl = _ScriptedController([Cmd(Action.EDIT), Cmd(Action.STEP)], multiline=["hand written answer"])
    sess = _session(g, ctrl)
    # EDIT loops back (does not auto-advance) so STEP still gets returned.
    assert sess._pause(leaf) is Action.STEP
    result = g.get_node(leaf.node_id).details[DetailKey.ACTION_RESULT.value]
    assert result["content"] == "hand written answer"
    assert result["success"] is True
    assert result["human_edited"] is True


def test_edit_merges_into_existing_action_result():
    g, leaf = _leaf_graph()
    g.update_details(leaf.node_id, {DetailKey.ACTION_RESULT.value: {"success": False, "links": ["a"]}})
    ctrl = _ScriptedController([Cmd(Action.EDIT), Cmd(Action.STEP)], multiline=["new body"])
    sess = _session(g, ctrl)
    sess._pause(leaf)
    result = g.get_node(leaf.node_id).details[DetailKey.ACTION_RESULT.value]
    assert result["content"] == "new body"
    assert result["success"] is True
    assert result["links"] == ["a"]  # preserved


def test_edit_empty_cancels():
    g, leaf = _leaf_graph()
    ctrl = _ScriptedController([Cmd(Action.EDIT), Cmd(Action.STEP)], multiline=["   "])
    sess = _session(g, ctrl)
    sess._pause(leaf)
    assert DetailKey.ACTION_RESULT.value not in g.get_node(leaf.node_id).details


# -------------------------------------------------------------------- feedback


def test_feedback_from_argument_writes_detail_key():
    g, leaf = _leaf_graph()
    ctrl = _ScriptedController([Cmd(Action.FEEDBACK, "prefer official docs"), Cmd(Action.STEP)])
    sess = _session(g, ctrl)
    assert sess._pause(leaf) is Action.STEP
    assert g.get_node(leaf.node_id).details[DetailKey.HUMAN_FEEDBACK.value] == "prefer official docs"


def test_feedback_prompts_when_no_argument():
    g, leaf = _leaf_graph()
    ctrl = _ScriptedController([Cmd(Action.FEEDBACK, ""), Cmd(Action.STEP)], lines=["typed note"])
    sess = _session(g, ctrl)
    sess._pause(leaf)
    assert g.get_node(leaf.node_id).details[DetailKey.HUMAN_FEEDBACK.value] == "typed note"


def test_feedback_empty_cancels():
    g, leaf = _leaf_graph()
    ctrl = _ScriptedController([Cmd(Action.FEEDBACK, ""), Cmd(Action.STEP)], lines=["   "])
    sess = _session(g, ctrl)
    sess._pause(leaf)
    assert DetailKey.HUMAN_FEEDBACK.value not in g.get_node(leaf.node_id).details


# --------------------------------------------------------------------- autorun


@pytest.mark.asyncio
async def test_autorun_terminates_on_none():
    g, leaf = _leaf_graph()
    sess = DebugSession(engine=_FakeEngine(script=[None]), graph=g, ctrl=Controller(), max_steps=10)
    sess._out = lambda *_a, **_k: None
    await sess._autorun(leaf.node_id, depth=1)
    assert sess._step == 1  # single step, then engine returned None


@pytest.mark.asyncio
async def test_autorun_halts_at_max_steps():
    g, leaf = _leaf_graph()
    # echo a non-terminal, non-ancestor id forever -> only the max_steps guard stops it.
    sess = DebugSession(engine=_FakeEngine(echo="other"), graph=g, ctrl=Controller(), max_steps=3)
    sess._out = lambda *_a, **_k: None
    await sess._autorun(leaf.node_id, depth=1)
    assert sess._step == 3
    assert sess._halt() is True


@pytest.mark.asyncio
async def test_autorun_survives_engine_exception():
    class _BoomEngine:
        async def step(self, *a):
            raise RuntimeError("boom")

    g, leaf = _leaf_graph()
    sess = DebugSession(engine=_BoomEngine(), graph=g, ctrl=Controller(), max_steps=10)
    sess._out = lambda *_a, **_k: None
    await sess._autorun(leaf.node_id, depth=1)  # must not raise
    assert sess._step == 1
