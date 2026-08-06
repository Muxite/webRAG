"""Dispatch reachability is not PRACTICAL reachability.

`_execute_action` now resolves an action by name through `LeafActionRegistry`, so an action
registered via `register()`/`install_pack()` genuinely runs. But the expansion prompt only ever
injected the bare NAMES into one sentence (`{allowed_actions}`), hand-wrote descriptions for
search/visit/think/save alone, and then spent a whole SEARCH + VISIT PIPELINE section pushing the
model toward those four ("Both search and visit are required for any web information task"). A
model reading that prompt would never pick `wikipedia_summary` — not because dispatch was broken,
but because nothing told it what the action does or when it beats search+visit.

These pin the fix (the extra-actions menu) and, above all, its no-op guarantee: a run that allows
only core actions must produce a BYTE-IDENTICAL prompt, since every benchmark task and fixture is
such a run.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.actions import LeafAction, LeafActionRegistry
from agent.app.idea_policies.base import IdeaActionType
from agent.app.idea_policies.expansion import (
    EXTRA_ACTIONS_MENU_PLACEHOLDER,
    LlmExpansionPolicy,
    build_extra_actions_menu,
)
from agent.app.idea_policies.extra_actions.pack import ExtraActionPack
from agent.app.idea_policies.extra_actions.wikipedia import WikipediaSummaryAction


class _StubIO:
    telemetry = None
    connector_chroma = None

    def set_telemetry(self, telemetry):
        return None


def _graph_and_node():
    graph = IdeaDag(root_title="Research", root_details={"mandate": "Research"})
    return graph, graph.get_node(graph.root_id())


def _system_prompt(policy) -> str:
    graph, node = _graph_and_node()
    return policy._build_messages(graph, node)[0]["content"]


def _registry_with_extras() -> LeafActionRegistry:
    registry = LeafActionRegistry()
    registry.install_pack(ExtraActionPack())
    return registry


# --- the no-op guarantee ----------------------------------------------------------------------


def test_default_run_prompt_is_byte_identical_to_the_pre_menu_template():
    """The shipped template now hosts `{extra_actions_menu}`; with no extra action allowed the
    rendered prompt must equal what the template WITHOUT that placeholder renders — same bytes,
    not merely "no menu text"."""
    policy = LlmExpansionPolicy(io=MagicMock(), model_name="m", actions=_registry_with_extras())
    shipped_template = policy.settings["expansion_system_prompt"]
    assert EXTRA_ACTIONS_MENU_PLACEHOLDER in shipped_template, "the shipped template hosts the menu"

    # Reconstruct the literal pre-change template: the placeholder took over the blank line that
    # used to separate the VISIT DETAILS block from the output contract. Rebuilding it this way
    # (rather than substituting "" the way the code does) is what makes this an independent
    # check that the template edit itself was newline-neutral.
    pre_menu_template = shipped_template.replace(f"\n{EXTRA_ACTIONS_MENU_PLACEHOLDER}\n", "\n\n")
    before = LlmExpansionPolicy(
        io=MagicMock(), model_name="m", actions=_registry_with_extras(),
        settings={"expansion_system_prompt": pre_menu_template},
    )

    assert _system_prompt(policy) == _system_prompt(before)


def test_no_registry_and_default_registry_render_the_same_prompt():
    """A standalone policy (no registry injected) and an engine-style one with the default
    registry must agree — the menu never appears for built-ins."""
    without = LlmExpansionPolicy(io=MagicMock(), model_name="m")
    with_default = LlmExpansionPolicy(io=MagicMock(), model_name="m", actions=LeafActionRegistry())

    assert _system_prompt(without) == _system_prompt(with_default)
    assert "EXTRA ACTIONS" not in _system_prompt(without)
    assert EXTRA_ACTIONS_MENU_PLACEHOLDER not in _system_prompt(without)


def test_installed_but_not_allowed_actions_are_not_advertised():
    """Registering a pack must not leak it into the prompt: the menu is driven by
    `allowed_actions`, the same gate `_execute_action` enforces."""
    policy = LlmExpansionPolicy(
        io=MagicMock(), model_name="m", actions=_registry_with_extras(),
        settings={"allowed_actions": ["search", "visit", "think", "save"]},
    )

    prompt = _system_prompt(policy)

    assert "EXTRA ACTIONS" not in prompt
    assert "wikipedia_summary" not in prompt


def test_core_actions_are_never_described_twice():
    """Every `IdeaActionType` member already has a hand-written ACTIONS entry (or is deliberately
    undocumented, like plan_library_search); the menu must not restate any of them."""
    registry = _registry_with_extras()
    allowed = [member.value for member in IdeaActionType]

    assert registry.menu_lines(allowed) == []


# --- the menu itself --------------------------------------------------------------------------


def test_allowed_extra_action_is_described_with_args_and_placement():
    policy = LlmExpansionPolicy(
        io=MagicMock(), model_name="m", actions=_registry_with_extras(),
        settings={"allowed_actions": ["search", "visit", "think", "save", "wikipedia_summary"]},
    )

    prompt = _system_prompt(policy)

    assert "EXTRA ACTIONS" in prompt
    # Name, argument shape and the class's own one-line description all reach the model.
    assert "- wikipedia_summary: details={title, lang?}. Look up a Wikipedia article summary" in prompt
    # Only the allowed one — not the rest of the installed pack.
    assert "arxiv_search" not in prompt
    # It lands with the other action documentation, before the output contract, not after it.
    assert prompt.index("EXTRA ACTIONS") > prompt.index("ACTIONS:")
    assert prompt.index("EXTRA ACTIONS") < prompt.index("Output: JSON")
    # And it reconciles with the "every data-gathering task needs a visit" rule it would
    # otherwise contradict.
    assert "no separate visit node is needed for it" in prompt


def test_allowed_but_unregistered_name_is_not_advertised():
    """The menu never advertises a dead action: a name nobody registered a class for produces no
    line (it would fall back to THINK if the model picked it)."""
    policy = LlmExpansionPolicy(
        io=MagicMock(), model_name="m", actions=_registry_with_extras(),
        settings={"allowed_actions": ["search", "some_future_tool"]},
    )

    prompt = _system_prompt(policy)

    assert "EXTRA ACTIONS" not in prompt
    # The pre-existing bare-name sentence is unchanged — that behaviour is not what we fixed.
    assert "some_future_tool" in prompt


def test_template_without_the_placeholder_gets_the_menu_appended():
    """Alternate settings files and custom prompts do not host the placeholder; they must still
    learn about the extra action rather than silently losing it."""
    policy = LlmExpansionPolicy(
        io=MagicMock(), model_name="m", actions=_registry_with_extras(),
        settings={
            "expansion_system_prompt": "PLAIN TEMPLATE for {allowed_actions} / {max_children}",
            "allowed_actions": ["search", "wikipedia_summary"],
        },
    )

    prompt = _system_prompt(policy)

    assert prompt.startswith("PLAIN TEMPLATE for search, wikipedia_summary")
    assert "- wikipedia_summary: details={title, lang?}." in prompt


def test_menu_lines_follow_allowed_order_and_dedupe():
    registry = _registry_with_extras()

    lines = registry.menu_lines(["json_path", "wikipedia_summary", "json_path", "think"])

    assert [line.split(":")[0] for line in lines] == ["- json_path", "- wikipedia_summary"]


def test_empty_menu_renders_nothing():
    assert build_extra_actions_menu([]) == ""


# --- how an action describes itself -------------------------------------------------------------


def test_description_is_harvested_from_the_class_docstring():
    """One source of truth: the description comes from the docstring the action already has, so
    it cannot drift from the action's own documentation."""
    assert WikipediaSummaryAction.menu_description() == (
        "Look up a Wikipedia article summary by title."
    )


def test_explicit_description_overrides_the_docstring():
    class _Custom(LeafAction):
        """A docstring nobody wants in the prompt."""

        name = "custom_tool"
        description = "Do the one useful thing."
        args_hint = "details={thing}"

        async def execute(self, graph, node_id, io):  # pragma: no cover — never run here
            return {}

    assert _Custom.menu_line() == "- custom_tool: details={thing}. Do the one useful thing."


def test_action_without_args_hint_still_gets_a_line():
    class _Bare(LeafAction):
        """Just does a thing."""

        name = "bare_tool"

        async def execute(self, graph, node_id, io):  # pragma: no cover — never run here
            return {}

    assert _Bare.menu_line() == "- bare_tool: Just does a thing."


# --- engine integration -------------------------------------------------------------------------


def test_install_action_pack_reaches_the_prompt_and_the_dispatch_gate():
    """The failure mode `install_action_pack` exists to prevent: assigning a fresh list to
    `engine.settings["allowed_actions"]` updates the dispatch gate but leaves the expansion
    policy's own settings snapshot stale, so the model is never told the action exists."""
    engine = IdeaDagEngine(io=_StubIO(), settings={"allowed_actions": ["search", "visit", "think"]})

    installed = engine.install_action_pack(ExtraActionPack(settings=engine.settings))

    assert "wikipedia_summary" in installed
    assert "wikipedia_summary" in engine.settings["allowed_actions"]          # dispatch gate
    assert "wikipedia_summary" in engine.expansion.settings["allowed_actions"]  # prompt menu
    prompt = _system_prompt(engine.expansion)
    assert "- wikipedia_summary: details={title, lang?}." in prompt


def test_engine_default_run_advertises_no_extra_actions():
    engine = IdeaDagEngine(io=_StubIO())

    assert "EXTRA ACTIONS" not in _system_prompt(engine.expansion)


def test_install_action_pack_without_allow_registers_only():
    engine = IdeaDagEngine(io=_StubIO(), settings={"allowed_actions": ["search", "think"]})

    engine.install_action_pack(ExtraActionPack(settings=engine.settings), allow=False)

    assert engine.actions.has("wikipedia_summary")
    assert "wikipedia_summary" not in engine.settings["allowed_actions"]
    assert "EXTRA ACTIONS" not in _system_prompt(engine.expansion)


@pytest.mark.asyncio
async def test_menu_failure_never_sinks_an_expansion():
    """A registry that raises while describing itself must degrade to no menu, not to a failed
    expansion — the menu is an enhancement, not a dependency."""

    class _AngryRegistry:
        def menu_lines(self, allowed):
            raise RuntimeError("boom")

    policy = LlmExpansionPolicy(io=MagicMock(), model_name="m", actions=_AngryRegistry())

    prompt = _system_prompt(policy)

    assert "EXTRA ACTIONS" not in prompt
    assert EXTRA_ACTIONS_MENU_PLACEHOLDER not in prompt
