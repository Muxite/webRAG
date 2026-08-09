"""Tool availability as typed config (``ToolsConfig``) and its native-engine wiring.

Slice B.1 of the unified-agent plan: the action menu stops being a hard-coded literal at each
entry point and becomes a declarative config group, and ``install_action_pack`` — built and
unit-tested but never called in production — gets its first real call site.

The load-bearing assertion here is the regression guard: with default settings the engine's
``allowed_actions`` must still be exactly today's six core actions, in today's order. Everything
this group adds is opt-in on top of that.
"""

from __future__ import annotations

import pytest

from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.config import IdeaConfig, ToolsConfig
from agent.app.idea_policies.extra_actions import SandboxToolPack


#: The list ``idea_test_runner`` has always assembled by hand for a graph run.
CORE_ACTIONS = ["search", "visit", "save", "think", "merge", "verify"]


class _StubIO:
    telemetry = None
    connector_chroma = None
    connector_sandbox = None

    def set_telemetry(self, telemetry):
        return None


def _engine(**settings) -> IdeaDagEngine:
    return IdeaDagEngine(io=_StubIO(), settings=settings or None)


def test_tools_config_defaults_reproduce_todays_action_menu():
    empty = ToolsConfig.from_settings({})
    assert list(empty.core_actions) == CORE_ACTIONS
    assert empty.sandbox_pack_enabled is False
    assert empty.calculator_pack_enabled is False
    # The shipped JSON must agree with the dataclass default, both for the menu (via the legacy
    # `allowed_actions` key it still reads) and for the packs.
    prod = ToolsConfig.from_settings(load_idea_dag_settings())
    assert list(prod.core_actions) == CORE_ACTIONS
    assert (prod.sandbox_pack_enabled, prod.calculator_pack_enabled) == (False, False)
    # The narrowable subset defaults to everything SandboxToolPack ships.
    assert list(prod.sandbox_pack_actions) == SandboxToolPack().names()


def test_tools_config_reads_legacy_allowed_actions_then_typed_override():
    # No typed override -> the run's own (possibly narrowed) allowed_actions is carried through.
    carried = ToolsConfig.from_settings({"allowed_actions": ["think", "regex_extract"]})
    assert carried.core_actions == ("think", "regex_extract")
    # The typed key wins when set, and a JSON array is normalised to a tuple.
    override = ToolsConfig.from_settings(
        {"allowed_actions": ["think"], "tools_core_actions": ["search", "visit"]}
    )
    assert override.core_actions == ("search", "visit")
    assert IdeaConfig.from_settings({}).tools.core_actions == tuple(CORE_ACTIONS)


def test_default_engine_allowed_actions_are_byte_identical():
    """The regression guard: default construction must not move the action menu at all."""
    engine = _engine()
    assert engine.settings["allowed_actions"] == CORE_ACTIONS
    assert engine.settings["allowed_actions"] == load_idea_dag_settings()["allowed_actions"]
    # The expansion policy's own snapshot (the prompt's action menu) must agree.
    assert engine.expansion.settings["allowed_actions"] == CORE_ACTIONS
    # Nothing is installed beyond the built-ins.
    assert not engine.actions.has("read_file")
    assert not engine.actions.has("run_python")


def test_engine_respects_a_callers_narrowed_allowed_actions():
    """ToolsConfig must not clobber the per-run overrides callers already rely on."""
    engine = _engine(allowed_actions=["think", "regex_extract"])
    assert engine.settings["allowed_actions"] == ["think", "regex_extract"]
    engine = _engine(allowed_actions=CORE_ACTIONS, tools_core_actions=["search", "think"])
    assert engine.settings["allowed_actions"] == ["search", "think"]


def test_sandbox_pack_grants_only_the_configured_subset():
    engine = _engine(
        tools_sandbox_pack_enabled=True,
        tools_sandbox_pack_actions=["read_file"],
    )
    allowed = engine.settings["allowed_actions"]
    assert allowed == CORE_ACTIONS + ["read_file"]
    assert engine.expansion.settings["allowed_actions"] == allowed
    # The whole pack is INSTALLED (the registry knows every class) but only the configured
    # subset is PERMITTED — narrowing happens at the gate, not by editing ACTION_CLASSES.
    for name in SandboxToolPack().names():
        assert engine.actions.has(name)
        assert (name in allowed) is (name == "read_file")


def test_sandbox_pack_default_subset_grants_the_whole_pack():
    engine = _engine(tools_sandbox_pack_enabled=True)
    allowed = engine.settings["allowed_actions"]
    assert allowed == CORE_ACTIONS + SandboxToolPack().names()


def test_calculator_pack_is_independently_enableable():
    engine = _engine(tools_calculator_pack_enabled=True)
    assert engine.settings["allowed_actions"] == CORE_ACTIONS + ["run_python"]
    assert engine.actions.has("run_python")
    # "let the model compute" without "let the model read and write files".
    assert not engine.actions.has("write_file")


@pytest.mark.asyncio
async def test_an_installed_but_unpermitted_pack_action_stays_closed():
    """End-to-end: a narrowed subset is enforced by the dispatch gate, not just by the list."""
    from agent.app.idea_dag import IdeaDag
    from agent.app.idea_policies.base import DetailKey, IdeaActionType

    engine = _engine(
        tools_sandbox_pack_enabled=True,
        tools_sandbox_pack_actions=["read_file"],
    )
    graph = IdeaDag(root_title="root")
    node = graph.add_child(
        graph.root_id(),
        "leaf",
        details={DetailKey.ACTION.value: "write_file", "path": "x.txt", "content": "y"},
    )

    result = await engine._execute_action(graph, graph.root_id(), node.node_id)

    assert result["action"] == IdeaActionType.THINK.value
