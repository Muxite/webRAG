"""`_execute_action`'s dispatch gate used to coerce the node's action-type string through
`IdeaActionType(str(action_type))` before ever consulting the registry. That coercion raises
for any name that was never added to the enum, so `LeafActionRegistry.register()`/
`install_pack()` — despite being built and documented for exactly this — could never make a
new action reachable via the model's own action selection; it silently fell back to THINK
every time (see `idea_policies/extra_actions/pack.py`'s own docstring, written around this
exact limitation).

These pin the fix: a registered-but-non-enum action, once allowed, is genuinely dispatched.
"""
from __future__ import annotations

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.extra_actions.regex_extract import RegexExtractAction


class _StubIO:
    telemetry = None
    connector_chroma = None

    def set_telemetry(self, telemetry):
        return None


def _make_engine(*, allowed_actions=None):
    settings = {"allow_unscored_selection": True, "min_score_threshold": 0.0}
    if allowed_actions is not None:
        settings["allowed_actions"] = allowed_actions
    return IdeaDagEngine(io=_StubIO(), settings=settings)


def _node(graph, *, action_name, details=None):
    return graph.add_child(
        graph.root_id(),
        "leaf",
        details={DetailKey.ACTION.value: action_name, **(details or {})},
    )


@pytest.mark.asyncio
async def test_a_registered_non_enum_action_is_actually_reachable():
    """The fix: `regex_extract` was never an `IdeaActionType` member, but once it's registered
    AND allowed, `_execute_action` must run it for real — not silently default to THINK."""
    engine = _make_engine(allowed_actions=["think", "regex_extract"])
    engine.actions.register(RegexExtractAction)
    graph = IdeaDag(root_title="root")
    node = _node(
        graph,
        action_name="regex_extract",
        details={"pattern": r"\d+", "text": "3 apples, 12 oranges"},
    )

    result = await engine._execute_action(graph, graph.root_id(), node.node_id)

    assert result["action"] == "regex_extract"
    assert result["matches"] == ["3", "12"]
    assert result["count"] == 2


@pytest.mark.asyncio
async def test_registered_but_not_allowed_still_falls_back_to_think():
    """Registering an action makes it DISPATCHABLE, not automatically PERMITTED — the
    allowed_actions gate is unchanged and still wins."""
    engine = _make_engine(allowed_actions=["think"])
    engine.actions.register(RegexExtractAction)
    graph = IdeaDag(root_title="root")
    node = _node(
        graph,
        action_name="regex_extract",
        details={"pattern": r"\d+", "text": "3 apples"},
    )

    result = await engine._execute_action(graph, graph.root_id(), node.node_id)

    assert result["action"] == IdeaActionType.THINK.value


@pytest.mark.asyncio
async def test_allowed_but_never_registered_still_falls_back_to_think():
    """An action name that's permitted but nobody ever registered a class for — the
    pre-existing 'unknown action' fallback, unchanged."""
    engine = _make_engine(allowed_actions=["think", "some_future_tool"])
    graph = IdeaDag(root_title="root")
    node = _node(graph, action_name="some_future_tool")

    result = await engine._execute_action(graph, graph.root_id(), node.node_id)

    assert result["action"] == IdeaActionType.THINK.value


@pytest.mark.asyncio
async def test_existing_enum_actions_still_dispatch_byte_identically():
    """Regression guard: every pre-existing action_type (an IdeaActionType member) must keep
    resolving exactly as before the fix."""
    engine = _make_engine()
    graph = IdeaDag(root_title="root")
    node = _node(graph, action_name=IdeaActionType.THINK.value)

    result = await engine._execute_action(graph, graph.root_id(), node.node_id)

    assert result["action"] == IdeaActionType.THINK.value
    assert result["success"] is True


@pytest.mark.asyncio
async def test_extra_action_pack_bundle_is_reachable_via_install_pack():
    """The end-to-end claim `extra_actions/pack.py`'s own docstring was waiting on: its whole
    bundle, wired in through the registry's existing `install_pack`, is now genuinely
    selectable by the model — not just importable and callable directly."""
    from agent.app.idea_policies.extra_actions.pack import ExtraActionPack

    engine = _make_engine(allowed_actions=["think"])
    installed = engine.actions.install_pack(ExtraActionPack(settings=engine.settings))
    engine.settings["allowed_actions"] = engine.settings["allowed_actions"] + installed
    assert "regex_extract" in installed

    graph = IdeaDag(root_title="root")
    node = _node(
        graph,
        action_name="regex_extract",
        details={"pattern": r"[a-z]+", "text": "abc 123 def"},
    )

    result = await engine._execute_action(graph, graph.root_id(), node.node_id)

    assert result["action"] == "regex_extract"
    assert result["matches"] == ["abc", "def"]
