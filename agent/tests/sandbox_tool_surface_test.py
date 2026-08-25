"""Every arm must get the SAME sandbox surface, or a closed-environment comparison is a confound.

The native engine could already manipulate a sandbox filesystem via ``SandboxToolPack``. The flat
arms had none, and the codebench matrix drives only the compiled variant -- so a closed-environment
task was measurable on exactly one arm, which cannot support any claim about the DAG versus a
linear agent.

Two properties are pinned here:

* **Parity.** The shared surface names exactly the verbs the native pack exposes. If the two lists
  drift, the arms are being compared on different capabilities -- the precise confound the shared
  surface exists to prevent.
* **The capability boundary.** ``run_python`` / ``run_pytest`` / ``search_web`` are reachable
  through the shared dispatcher and are deliberately NOT exposed. Giving three arms the same file
  surface is a much smaller decision than handing a web-research agent arbitrary code execution.

No network, no real filesystem: the connector is a stub.
"""
from __future__ import annotations

import asyncio

import pytest

from agent.app.sandbox_tool_surface import (
    NO_SANDBOX,
    PARITY_ACTIONS,
    format_sandbox_result,
    run_sandbox_action,
    sandbox_menu,
)


class _StubSandbox:
    """Records the call and returns a canned connector-shaped result."""

    def __init__(self, result=None):
        self.calls = []
        self._result = result if result is not None else {"ok": True, "action": "read_file",
                                                          "content": "hello"}

    async def read_file(self, path):
        self.calls.append(("read_file", path))
        return self._result

    async def write_file(self, path, content):
        self.calls.append(("write_file", path, content))
        return {"ok": True, "action": "write_file", "path": path, "bytes": len(content)}

    async def list_dir(self, path=None):
        self.calls.append(("list_dir", path))
        return {"ok": True, "action": "list_dir", "entries": ["a.txt", "b.txt"]}


def test_the_surface_matches_the_native_packs_actions():
    """Drift here means the arms are no longer comparable."""
    from agent.app.idea_policies.extra_actions.sandbox_tools import SandboxToolPack

    native = {cls.name for cls in SandboxToolPack.ACTION_CLASSES}
    assert set(PARITY_ACTIONS) == native


@pytest.mark.parametrize("forbidden", ["run_python", "run_pytest", "search_web", "patch_file"])
def test_code_execution_is_not_in_the_shared_surface(forbidden):
    """Reachable through the dispatcher, deliberately not offered here."""
    assert forbidden not in PARITY_ACTIONS


def test_the_menu_describes_every_action():
    menu = sandbox_menu()
    for name in PARITY_ACTIONS:
        assert name in menu


def test_the_menu_is_empty_when_nothing_is_advertised():
    assert sandbox_menu([]) == ""


def test_a_run_without_a_workdir_refuses_rather_than_raising():
    assert asyncio.run(run_sandbox_action(None, "read_file", {"path": "x"})) == NO_SANDBOX


def test_an_unknown_verb_is_reported_with_the_available_ones():
    obs = asyncio.run(run_sandbox_action(_StubSandbox(), "rm_rf", {"path": "/"}))
    assert "UNKNOWN FILE ACTION" in obs
    assert "read_file" in obs


def test_code_execution_is_refused_even_though_the_dispatcher_supports_it():
    """The boundary has to hold at the surface, not just by omission from the menu."""
    obs = asyncio.run(run_sandbox_action(_StubSandbox(), "run_python", {"code": "print(1)"}))
    assert "UNKNOWN FILE ACTION" in obs


def test_a_successful_read_returns_its_content():
    obs = asyncio.run(run_sandbox_action(_StubSandbox(), "read_file", {"path": "a.txt"}))
    assert "hello" in obs
    assert obs.startswith("READ_FILE OK")


def test_a_failed_action_is_reported_as_an_error_observation():
    sandbox = _StubSandbox({"ok": False, "action": "read_file", "error": "outside workdir"})
    obs = asyncio.run(run_sandbox_action(sandbox, "read_file", {"path": "../etc/passwd"}))
    assert "READ_FILE ERROR" in obs
    assert "outside workdir" in obs


def test_a_listing_is_rendered_line_by_line():
    obs = asyncio.run(run_sandbox_action(_StubSandbox(), "list_dir", {"path": "."}))
    assert "a.txt" in obs and "b.txt" in obs


def test_non_dict_args_are_tolerated():
    obs = asyncio.run(run_sandbox_action(_StubSandbox(), "list_dir", "not-a-dict"))
    assert "LIST_DIR OK" in obs


def test_a_raising_connector_becomes_an_observation():
    class _Boom:
        async def read_file(self, path):
            raise RuntimeError("disk on fire")

    obs = asyncio.run(run_sandbox_action(_Boom(), "read_file", {"path": "a"}))
    assert "READ_FILE ERROR" in obs and "disk on fire" in obs


def test_output_is_capped():
    sandbox = _StubSandbox({"ok": True, "action": "read_file", "content": "x" * 50_000})
    obs = asyncio.run(run_sandbox_action(sandbox, "read_file", {"path": "big"}))
    assert len(obs) < 10_000


def test_format_handles_a_malformed_result():
    assert "ERROR" in format_sandbox_result("read_file", None)


def test_format_reports_a_bare_success_without_payload():
    assert format_sandbox_result("write_file", {"ok": True, "action": "write_file"}) \
        == "WRITE_FILE OK"
