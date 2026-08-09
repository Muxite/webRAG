"""The shared sandbox-action dispatcher, and the two call sites that must not drift from it.

``sandbox_dispatch`` exists because the native engine's ``LeafAction`` pack and codebench's
``graph_compiled_code`` leaf loop used to translate ``{action, args}`` into a ``SandboxConnector``
call independently, and had already disagreed about which argument aliases they accept. So the
load-bearing tests here are the PARITY ones — run against both adapters, so a future one-sided
change fails immediately instead of being discovered live:

1. every path alias resolves the same way through both call sites;
2. the normalized result keeps every field each call site actually reads — an alias-only parity
   test would pass with ``entries`` or ``exit_code`` silently dropped, and still break
   ``_observation()`` / ``_from_sandbox()``;
3. malformed model-authored input (a lone surrogate, an embedded NUL) comes back as a failed
   result through both, rather than raising out of a leaf and taking the whole run with it;
4. sharing the DISPATCHER did not widen what the native engine can REACH — the pack still exposes
   eight actions and no code execution.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.app import sandbox_dispatch
from agent.app.connector_sandbox import SandboxConnector
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey
from agent.app.idea_policies.config import SandboxActionConfig
from agent.app.idea_policies.extra_actions import SandboxToolPack
from agent.app.sandbox_dispatch import ACTION_NAMES, dispatch_sandbox_action
from agent.app.testing import execution_compiled_code as ecc

#: A lone surrogate, constructed the way one actually reaches the engine: a model emits it,
#: ``json.loads`` accepts it happily, and every UTF-8 encoder downstream refuses it.
LONE_SURROGATE = json.loads('{"content": "\\ud800"}')["content"]

_NATIVE_CLASSES = {cls.name: cls for cls in SandboxToolPack.ACTION_CLASSES}


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxConnector:
    root = tmp_path / "work"
    root.mkdir()
    (root / "notes.txt").write_text("alpha\nbeta\nalpha again\n", encoding="utf-8")
    (root / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    return SandboxConnector(root, limits=SandboxActionConfig(workdir_root=str(root)))


def _fresh_sandbox(tmp_path: Path, name: str) -> SandboxConnector:
    """A second sandbox with the SAME starting fixture, so two adapters can be compared."""
    root = tmp_path / name
    root.mkdir()
    (root / "notes.txt").write_text("alpha\nbeta\nalpha again\n", encoding="utf-8")
    (root / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    return SandboxConnector(root, limits=SandboxActionConfig(workdir_root=str(root)))


# --- the two adapters, as one callable each -----------------------------------------------------
def _codebench(sandbox: SandboxConnector, action: str, args: dict) -> dict:
    """Through codebench's leaf-loop dispatcher (``graph_compiled_code``)."""
    return asyncio.run(ecc._dispatch_action(sandbox, action, dict(args)))


def _native(sandbox: SandboxConnector, action: str, args: dict) -> dict:
    """Through the native engine's ``LeafAction`` pack."""
    cls = _NATIVE_CLASSES[action]
    return asyncio.run(cls(settings={}).run(sandbox, dict(args)))


ADAPTERS = {"codebench": _codebench, "native": _native}


def _succeeded(result: dict) -> bool:
    """Both adapters' "it worked" flag: the native pack renames the connector's ``ok`` to
    ``success``; that rename is the ONLY difference the two shapes are allowed to have."""
    return bool(result.get("success", result.get("ok")))


def _both(action: str):
    """The adapter ids that can reach ``action`` (codebench advertises everything; the native
    pack advertises eight — see :mod:`agent.app.idea_policies.extra_actions.sandbox_tools`)."""
    return ["codebench"] + (["native"] if action in _NATIVE_CLASSES else [])


# --- 1. path-alias parity -----------------------------------------------------------------------
def test_the_path_alias_table_is_the_union_of_what_both_call_sites_used_to_accept():
    """Pinned as a LITERAL, not derived from the table under test.

    The parametrized parity test below iterates ``PATH_KEYS``, so shrinking the table would
    silently shrink the test rather than fail it. These two lists are what the native pack and
    codebench's loop each accepted before they were merged; dropping any of them takes a slot
    alias away from a live prompt, which costs a weak model a step it cannot see why it lost.
    """
    was_native = {"path", "file", "relpath", "filename", "dir", "directory"}
    was_codebench = {"path", "relpath", "file", "filename", "target", "file_path"}
    assert set(sandbox_dispatch.PATH_KEYS) == was_native | was_codebench
    assert sandbox_dispatch.PATH_KEYS[0] == "path", "the canonical name stays first"
    assert len(sandbox_dispatch.PATH_KEYS) == len(set(sandbox_dispatch.PATH_KEYS))


@pytest.mark.parametrize("adapter_id", ["codebench", "native"])
@pytest.mark.parametrize("alias", sandbox_dispatch.PATH_KEYS)
def test_both_call_sites_accept_every_path_alias(sandbox, adapter_id, alias):
    """The original drift: the native pack took ``dir``/``directory``, codebench took
    ``target``/``file_path``, and neither took the other's."""
    result = ADAPTERS[adapter_id](sandbox, "read_file", {alias: "notes.txt"})
    assert _succeeded(result), f"{adapter_id} does not accept the {alias!r} path alias"
    assert result["output"] == "alpha\nbeta\nalpha again\n"


@pytest.mark.parametrize("adapter_id", ["codebench", "native"])
def test_both_call_sites_report_a_missing_path_the_same_way(sandbox, adapter_id):
    result = ADAPTERS[adapter_id](sandbox, "read_file", {"nonsense_key": "notes.txt"})
    assert _succeeded(result) is False
    assert "missing 'path'" in result["error"]


@pytest.mark.parametrize("adapter_id", ["codebench", "native"])
def test_both_call_sites_refuse_the_same_escapes(sandbox, adapter_id, tmp_path):
    (tmp_path / "outside.txt").write_text("SECRET\n", encoding="utf-8")
    for escape in ("../outside.txt", "/etc/passwd", "sub/../../outside.txt"):
        result = ADAPTERS[adapter_id](sandbox, "read_file", {"path": escape})
        assert _succeeded(result) is False
        assert "escapes the workdir" in result["error"]
        assert "SECRET" not in str(result)


# --- 2. the normalized result keeps every field the call sites read -----------------------------
#: ``(action, args, the fields a call site actually reads off the result)``.
#: ``_observation()`` reads ``ok``/``output``/``error``; ``_run_code_leaf``'s step record reads
#: ``path``; the native pack's observations surface ``entries``/``stdout``/``exit_code``. A field
#: dropped from the normalized shape passes an alias-only parity test and still breaks the loop.
_SHAPE_CASES = [
    ("read_file", {"path": "notes.txt"}, {"path", "bytes", "output"}),
    ("write_file", {"path": "new.txt", "content": "hi\n"}, {"path", "bytes", "output"}),
    ("list_dir", {}, {"path", "entries", "output", "total_entries", "truncated"}),
    ("count_lines", {"path": "notes.txt"}, {"path", "stdout", "stderr", "exit_code", "output"}),
    ("count_lines", {"path": "notes.txt", "pattern": "alpha"},
     {"path", "stdout", "exit_code", "matched", "output"}),
    ("word_count", {"path": "notes.txt"}, {"path", "stdout", "exit_code", "output"}),
    ("head_file", {"path": "notes.txt", "lines": 1}, {"path", "stdout", "exit_code", "output"}),
    ("disk_usage", {}, {"path", "stdout", "exit_code", "output"}),
    ("find_files", {"name": "*.txt"}, {"stdout", "exit_code", "output"}),
    ("patch_file", {"path": "notes.txt", "old": "alpha", "new": "gamma"},
     {"path", "occurrences", "output"}),
    ("run_python", {"code": "print('hi')"}, {"stdout", "stderr", "exit_code", "output"}),
    ("run_pytest", {"path": "test_ok.py"}, {"path", "stdout", "exit_code", "output"}),
]


@pytest.mark.parametrize(
    "action, args, fields, adapter_id",
    [(action, args, fields, adapter_id)
     for action, args, fields in _SHAPE_CASES for adapter_id in _both(action)],
    ids=[f"{action}-{adapter_id}"
         for action, _, _ in _SHAPE_CASES for adapter_id in _both(action)],
)
def test_result_shape_keeps_the_fields_each_call_site_depends_on(sandbox, action, args, fields,
                                                                 adapter_id):
    result = ADAPTERS[adapter_id](sandbox, action, args)
    assert _succeeded(result) is True, result.get("error")
    assert fields <= set(result), f"{adapter_id}/{action} dropped {fields - set(result)}"
    assert result["action"] == action


@pytest.mark.parametrize("action, args, fields",
                         [case for case in _SHAPE_CASES if case[0] in _NATIVE_CLASSES],
                         ids=[case[0] for case in _SHAPE_CASES if case[0] in _NATIVE_CLASSES])
def test_both_call_sites_return_the_same_values_for_the_same_action(tmp_path, action, args, fields):
    """Not just the same KEYS: the same content, from two sandboxes with an identical fixture."""
    code = _codebench(_fresh_sandbox(tmp_path, "cb"), action, args)
    native = _native(_fresh_sandbox(tmp_path, "nat"), action, args)
    assert _succeeded(code) and _succeeded(native)
    for field in fields:
        assert code[field] == native[field], f"{action}.{field} differs between the call sites"


def test_search_web_result_shape(sandbox):
    sandbox.connector_search = type("S", (), {})()
    sandbox.connector_search.query_search = AsyncMock(return_value=[
        {"title": "Docs", "url": "https://example.test/x", "description": "how to"}])
    result = _codebench(sandbox, "search_web", {"q": "itertools"})
    assert result["ok"] is True
    assert {"results", "output"} <= set(result)
    assert "https://example.test/x" in result["output"]


def test_write_file_content_aliases_and_an_empty_write(sandbox):
    """``content: ""`` is a legitimate write (an empty ``__init__.py``); an ABSENT content is not."""
    assert _succeeded(_codebench(sandbox, "write_file", {"path": "a.py", "body": "x = 1\n"}))
    assert (sandbox.workdir / "a.py").read_text() == "x = 1\n"
    assert _succeeded(_codebench(sandbox, "write_file", {"path": "pkg/__init__.py", "content": ""}))
    assert (sandbox.workdir / "pkg" / "__init__.py").read_text() == ""
    missing = _codebench(sandbox, "write_file", {"path": "b.py"})
    assert _succeeded(missing) is False and "missing 'content'" in missing["error"]


# --- 3. malformed input is a result, never an exception -----------------------------------------
@pytest.mark.parametrize("adapter_id", ["codebench", "native"])
def test_unencodable_content_fails_cleanly_through_both_call_sites(sandbox, adapter_id):
    result = ADAPTERS[adapter_id](sandbox, "write_file", {"path": "bad.txt",
                                                         "content": LONE_SURROGATE})
    assert _succeeded(result) is False
    assert "cannot be encoded" in result["error"]


@pytest.mark.parametrize("adapter_id", ["codebench", "native"])
@pytest.mark.parametrize("bad, expected", [(LONE_SURROGATE, "cannot be encoded"),
                                           ("\x00", "null byte")])
def test_malformed_shell_pattern_fails_cleanly_through_both_call_sites(sandbox, adapter_id,
                                                                       bad, expected):
    """The ``_run()`` crash path, reachable from the NATIVE engine today: ``count_lines`` puts a
    model-authored pattern straight into argv, and ``create_subprocess_exec`` used to raise."""
    result = ADAPTERS[adapter_id](sandbox, "count_lines", {"path": "notes.txt",
                                                           "pattern": f"a{bad}"})
    assert _succeeded(result) is False
    assert "could not start" in result["error"] and expected in result["error"]


@pytest.mark.parametrize("bad, expected", [(LONE_SURROGATE, "cannot be encoded"),
                                           ("\x00", "null byte")])
def test_malformed_run_python_code_fails_cleanly(sandbox, bad, expected):
    result = _codebench(sandbox, "run_python", {"code": f"print('x'){bad}"})
    assert result["ok"] is False
    assert "could not start" in result["error"] and expected in result["error"]


class _StubIO:
    telemetry = None
    connector_chroma = None

    def __init__(self, sandbox=None):
        self.connector_sandbox = sandbox

    def set_telemetry(self, telemetry):
        return None


def test_malformed_content_does_not_sink_a_native_engine_run(sandbox):
    """The failure mode this fix is actually about: on the engine's NON-PARALLEL path an
    exception out of one leaf action ends the run, not just the leaf. It must arrive as an
    ordinary failed observation."""
    engine = IdeaDagEngine(io=_StubIO(sandbox),
                           settings={"allow_unscored_selection": True, "min_score_threshold": 0.0})
    engine.install_action_pack(SandboxToolPack(settings=engine.settings))
    graph = IdeaDag(root_title="root")
    node = graph.add_child(graph.root_id(), "leaf write_file",
                           details={DetailKey.ACTION.value: "write_file", "path": "bad.txt",
                                    "content": LONE_SURROGATE})

    result = asyncio.run(engine._execute_action(graph, graph.root_id(), node.node_id))

    assert result["action"] == "write_file"
    assert result["success"] is False and "cannot be encoded" in result["error"]


# --- 4. dispatchable is not the same as reachable -----------------------------------------------
def test_the_native_pack_exposes_eight_actions_and_no_code_execution():
    """Re-homing the dispatch logic must not widen the engine's reach by proximity.

    The pack has EIGHT actions. It has no ``patch_file`` — the native engine can only
    full-overwrite a file today, and adding one would be a capability addition, not a
    de-duplication — and none of the execution/web actions, which need their own security review.
    """
    names = SandboxToolPack().names()
    assert len(names) == 8 and len(set(names)) == 8
    assert set(names) == {"read_file", "write_file", "list_dir", "count_lines", "word_count",
                          "head_file", "disk_usage", "find_files"}
    for withheld in ("patch_file", "run_python", "run_pytest", "search_web"):
        assert withheld in ACTION_NAMES, f"{withheld} should still be dispatchable for codebench"
        assert withheld not in names, f"{withheld} must not be installable into the engine's pack"


def test_codebench_advertises_every_dispatchable_action():
    """The other half: codebench was missing the five inspection actions entirely, so an agent in
    the real Docker sandbox could not search or inspect files by pattern."""
    assert tuple(ecc._SANDBOX_ACTIONS) == ACTION_NAMES
    for action in ("count_lines", "word_count", "head_file", "disk_usage", "find_files"):
        assert action in ecc._SANDBOX_ACTIONS
        assert f"- {action}(" in ecc._CODE_LEAF_SYSTEM, "an action the prompt never names is unusable"
        assert action in ecc._CODE_LEAF_SYSTEM.rsplit("action", 1)[-1], "missing from the JSON enum"


def test_newly_exposed_inspection_actions_run_from_the_code_leaf_loop(tmp_path):
    """End to end through the leaf loop itself, not just the dispatcher."""
    root = tmp_path / "work"
    sb = SandboxConnector(root, limits=SandboxActionConfig(workdir_root=str(root)))
    asyncio.run(sb.write_file("notes.txt", "alpha\nbeta\n"))
    script = [
        {"action": "find_files", "args": {"name": "*.txt"}},
        {"action": "head_file", "args": {"path": "notes.txt", "lines": 1}},
        {"action": "count_lines", "args": {"path": "notes.txt", "pattern": "alpha"}},
        {"action": "word_count", "args": {"path": "notes.txt"}},
        {"action": "disk_usage", "args": {}},
        {"action": "finish", "args": {"summary": "looked around"}},
    ]
    io = _ScriptedIO(script)
    record = asyncio.run(ecc._run_code_leaf(sb, io, "inspect the tree", "", "m", 8, 2000, 512))
    assert record["finished"] is True
    assert [step["ok"] for step in record["steps"][:5]] == [True] * 5
    assert "notes.txt" in record["steps"][0]["observation"]
    assert "STDOUT:\nalpha\n" in record["steps"][1]["observation"], "head -n 1 of notes.txt"
    assert record["actions"] == 5, "the inspection actions must count as sandbox actions"


class _ScriptedIO:
    """Minimal AgentIO stand-in: replays a queued list of model decisions."""

    def __init__(self, script):
        self.script = list(script)

    def build_llm_payload(self, **kwargs):
        return {"messages": kwargs["messages"]}

    async def query_llm(self, payload, model_name=None, timeout_seconds=None):
        if not self.script:
            return json.dumps({"action": "finish", "args": {"summary": "done"}})
        item = self.script.pop(0)
        return item if isinstance(item, str) else json.dumps(item)


# --- dispatcher basics --------------------------------------------------------------------------
def test_unknown_action_names_the_caller_s_own_vocabulary(sandbox):
    """Each call site's hint must list what IT offers: ``finish`` is codebench's loop verb, not a
    sandbox action, so the shared default must not claim it."""
    default = asyncio.run(dispatch_sandbox_action(sandbox, "compile", {}))
    assert default["ok"] is False and default["action"] == "compile"
    assert "unknown action" in default["error"] and "finish" not in default["error"]

    from_codebench = _codebench(sandbox, "compile", {})
    assert "finish" in from_codebench["error"]
    assert "find_files" in from_codebench["error"], "the new actions belong in the recovery hint"


def test_dispatcher_survives_a_non_dict_args_object(sandbox):
    assert asyncio.run(dispatch_sandbox_action(sandbox, "list_dir", None))["ok"] is True
    assert asyncio.run(dispatch_sandbox_action(sandbox, "", {}))["action"] == "(none)"
    assert asyncio.run(dispatch_sandbox_action(sandbox, "  READ_FILE  ",
                                               {"path": "notes.txt"}))["ok"] is True
