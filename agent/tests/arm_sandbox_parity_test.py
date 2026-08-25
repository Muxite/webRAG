"""All four arms must offer the same file surface, and none of them when there is no workdir.

Before this, only the native engine could manipulate a sandbox filesystem (``SandboxToolPack``),
and the codebench matrix drove only the compiled variant. A closed-environment task was therefore
runnable on one arm -- which cannot test "the DAG beats a linear agent here", because a measured
difference would be a difference in TOOLS, not in reasoning.

Two invariants:

* with a workdir, every arm advertises exactly :data:`PARITY_ACTIONS`;
* without one, no arm advertises any file action, and the web-research prompt is untouched.

The second matters as much as the first: every existing web benchmark result was produced by the
no-sandbox path, so a prompt change there would silently invalidate comparisons against them.

No network, no real filesystem.
"""
from __future__ import annotations

from agent.app.sandbox_tool_surface import PARITY_ACTIONS


class _IO:
    def __init__(self, sandbox=None):
        self.connector_sandbox = sandbox


def test_langgraph_gets_the_full_surface_with_a_workdir():
    from agent.app.langgraph_solver import _make_sandbox_tools

    tools = _make_sandbox_tools(_IO(sandbox=object()))
    assert {t.name for t in tools} == set(PARITY_ACTIONS)


def test_langgraph_gets_nothing_without_a_workdir():
    from agent.app.langgraph_solver import _make_sandbox_tools

    assert _make_sandbox_tools(_IO()) == []


def test_sequential_react_advertises_the_surface_with_a_workdir():
    from agent.app.testing.execution_sequential import _system_prompt

    prompt = _system_prompt(True)
    for name in PARITY_ACTIONS:
        assert name in prompt


def test_sequential_reacts_web_prompt_is_untouched_without_a_workdir():
    """Every existing web benchmark result came from this path; it must not drift."""
    from agent.app.testing.execution_sequential import _SYSTEM, _system_prompt

    assert _system_prompt(False) == _SYSTEM
    for name in PARITY_ACTIONS:
        assert name not in _system_prompt(False)


def test_the_native_pack_and_the_shared_surface_agree():
    from agent.app.idea_policies.extra_actions.sandbox_tools import SandboxToolPack

    assert {cls.name for cls in SandboxToolPack.ACTION_CLASSES} == set(PARITY_ACTIONS)


def test_no_arm_exposes_code_execution():
    """The capability boundary has to hold on every arm, not just the native one."""
    from agent.app.idea_policies.extra_actions.sandbox_tools import SandboxToolPack
    from agent.app.langgraph_solver import _make_sandbox_tools
    from agent.app.testing.execution_sequential import _system_prompt

    forbidden = {"run_python", "run_pytest", "search_web"}
    assert not forbidden & {cls.name for cls in SandboxToolPack.ACTION_CLASSES}
    assert not forbidden & {t.name for t in _make_sandbox_tools(_IO(sandbox=object()))}
    prompt = _system_prompt(True)
    for name in forbidden:
        assert name not in prompt


def test_the_sandbox_block_is_additive_to_the_web_surface():
    """A closed-environment run keeps search/visit; the file verbs are added, not swapped in."""
    from agent.app.testing.execution_sequential import _system_prompt

    prompt = _system_prompt(True)
    for name in ("search", "visit", "verify", "finish"):
        assert name in prompt
