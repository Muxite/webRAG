"""The workdir file + read-only shell leaf actions ported from ``badmodel-lab/localagent``.

Two things are being pinned here, and the second matters more than the first:

1. Each op does what its localagent counterpart did (list/read/write, wc/grep/du/find/head),
   delegating to ``SandboxConnector`` instead of re-implementing confinement.
2. This capability class stays CLOSED unless three independent things are opted into: the pack is
   installed in the registry, the names are in ``allowed_actions``, and the run's ``AgentIO``
   actually carries a sandbox. A plain web-research run satisfies none of them, so no mandate can
   talk the engine into touching a filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.app.connector_sandbox import READONLY_COMMANDS, SandboxConnector
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey, IdeaActionType
from agent.app.idea_policies.config import SandboxActionConfig
from agent.app.idea_policies.extra_actions import SandboxToolPack
from agent.app.idea_policies.extra_actions.pack import ExtraActionPack


class _StubIO:
    telemetry = None
    connector_chroma = None

    def __init__(self, sandbox=None):
        self.connector_sandbox = sandbox

    def set_telemetry(self, telemetry):
        return None


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    (root / "notes.txt").write_text("alpha\nbeta\nalpha again\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("SECRET\n", encoding="utf-8")
    return root


@pytest.fixture()
def sandbox(workdir: Path) -> SandboxConnector:
    return SandboxConnector(workdir, limits=SandboxActionConfig(workdir_root=str(workdir)))


def _engine(sandbox=None, *, allowed=None, install=True, allow=True) -> IdeaDagEngine:
    settings = {"allow_unscored_selection": True, "min_score_threshold": 0.0}
    if allowed is not None:
        settings["allowed_actions"] = allowed
    engine = IdeaDagEngine(io=_StubIO(sandbox), settings=settings)
    if install:
        engine.install_action_pack(SandboxToolPack(settings=engine.settings), allow=allow)
    return engine


async def _run(engine: IdeaDagEngine, graph: IdeaDag, action: str, details: dict) -> dict:
    node = graph.add_child(
        graph.root_id(), f"leaf {action}",
        details={DetailKey.ACTION.value: action, **details},
    )
    return await engine._execute_action(graph, graph.root_id(), node.node_id)


# --- each op dispatches and does its job ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action, details, expected",
    [
        ("count_lines", {"path": "notes.txt"}, "3"),
        ("count_lines", {"path": "notes.txt", "pattern": "alpha"}, "2"),
        ("word_count", {"path": "notes.txt"}, "4"),
        ("head_file", {"path": "notes.txt", "lines": 1}, "alpha"),
        ("find_files", {"name": "*.txt"}, "./notes.txt"),
        ("disk_usage", {}, "."),
    ],
)
async def test_readonly_shell_ops_dispatch(sandbox, action, details, expected):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, action, details)

    assert result["action"] == action, "the registry must dispatch by name, not fall back to THINK"
    assert result["success"] is True
    assert expected in result["stdout"]


@pytest.mark.asyncio
async def test_grep_with_no_match_is_a_result_not_a_failure(sandbox):
    """``grep`` exits 1 for "no matches" — a fact the model needs, not an error to retry."""
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, "count_lines", {"path": "notes.txt", "pattern": "zzz"})

    assert result["success"] is True
    assert result["matched"] is False
    assert result["stdout"].strip() == "0"


@pytest.mark.asyncio
async def test_file_ops_round_trip_through_one_shared_sandbox(sandbox, workdir):
    """A graph's leaves share ONE ``SandboxConnector`` (it lives on the run's AgentIO), so a file
    one leaf writes is there for the next — no per-leaf sandbox, no lost workdir."""
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    written = await _run(engine, graph, "write_file", {"path": "sub/out.md", "content": "hello"})
    read_back = await _run(engine, graph, "read_file", {"path": "sub/out.md"})
    listed = await _run(engine, graph, "list_dir", {})
    counted = await _run(engine, graph, "count_lines", {"path": "sub/out.md"})

    assert written["success"] and written["path"] == "sub/out.md"
    assert read_back["success"] and read_back["output"] == "hello"
    assert listed["success"] and listed["entries"] == ["notes.txt", "sub/"]
    assert counted["success"], "a shell op sees the same workdir the file op wrote into"
    assert (workdir / "sub" / "out.md").read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_missing_required_detail_is_an_observation_not_an_exception(sandbox):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    assert (await _run(engine, graph, "read_file", {}))["success"] is False
    assert (await _run(engine, graph, "write_file", {"path": "a.txt"}))["success"] is False
    assert (await _run(engine, graph, "find_files", {}))["success"] is False
    missing = await _run(engine, graph, "head_file", {"path": "nope.txt"})
    assert missing["success"] is False and "no such file" in missing["error"]


# --- confinement ----------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["read_file", "count_lines", "word_count", "head_file"])
@pytest.mark.parametrize("escape", ["../outside.txt", "/etc/passwd", "sub/../../outside.txt"])
async def test_read_paths_that_escape_the_workdir_are_refused(sandbox, action, escape):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, action, {"path": escape})

    assert result["success"] is False
    assert "escapes the workdir" in result["error"]
    assert "SECRET" not in str(result)


@pytest.mark.asyncio
async def test_write_outside_the_workdir_is_refused(sandbox, tmp_path):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")
    target = tmp_path / "pwned.txt"

    result = await _run(engine, graph, "write_file", {"path": str(target), "content": "x"})

    assert result["success"] is False
    assert not target.exists(), "an absolute path must not be written outside the workdir"


@pytest.mark.asyncio
async def test_symlink_out_of_the_workdir_is_refused(sandbox, workdir, tmp_path):
    """``confine()`` resolves symlinks, so a link planted INSIDE the workdir pointing out still
    refuses — the exfiltration path codebench's own audit closed."""
    (workdir / "link.txt").symlink_to(tmp_path / "outside.txt")
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    read_result = await _run(engine, graph, "read_file", {"path": "link.txt"})
    count_result = await _run(engine, graph, "count_lines", {"path": "link.txt"})

    assert read_result["success"] is False and "SECRET" not in str(read_result)
    assert count_result["success"] is False and "SECRET" not in str(count_result)


@pytest.mark.asyncio
async def test_list_and_disk_usage_cannot_escape(sandbox):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    assert (await _run(engine, graph, "list_dir", {"path": ".."}))["success"] is False
    assert (await _run(engine, graph, "disk_usage", {"path": "../.."}))["success"] is False


@pytest.mark.asyncio
async def test_find_glob_cannot_walk_out_of_the_workdir(sandbox):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, "find_files", {"name": "../outside.txt"})

    assert result["success"] is False
    assert "escapes the workdir" in result["error"]


@pytest.mark.asyncio
async def test_connector_refuses_a_non_allow_listed_command(sandbox):
    """The allow-list is the reason this surface is safe to expose: nothing that writes, deletes
    or reaches the network can be constructed, even by a caller inside this repo."""
    assert "rm" not in READONLY_COMMANDS

    result = await sandbox.run_readonly("evil", ["rm", "-rf", "."])

    assert result["ok"] is False
    assert "allow-listed" in result["error"]


@pytest.mark.asyncio
async def test_connector_refuses_argv_paths_that_escape(sandbox):
    assert (await sandbox.run_readonly("peek", ["wc", "-l", "../outside.txt"]))["ok"] is False
    assert (await sandbox.run_readonly("peek", ["wc", "-l", "/etc/passwd"]))["ok"] is False


# --- the capability stays closed unless opted into --------------------------------------------


def test_no_default_registry_or_pack_exposure():
    """Nothing installs these by default, and they are deliberately not part of the curated
    web-research ``ExtraActionPack`` — that pack is read-only public APIs, this one is a
    filesystem."""
    engine = IdeaDagEngine(io=_StubIO())

    for name in SandboxToolPack().names():
        assert not engine.actions.has(name), f"{name} must not be registered by default"
        assert name not in engine.settings.get("allowed_actions", [])
    assert not set(SandboxToolPack().names()) & set(ExtraActionPack().names())


@pytest.mark.asyncio
async def test_registered_but_not_allowed_falls_back_to_think(sandbox, workdir):
    """Installing the pack without allowing it (``allow=False``) leaves the dispatch gate shut."""
    engine = _engine(sandbox, allowed=["think", "search"], allow=False)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, "write_file", {"path": "nope.txt", "content": "x"})

    assert result["action"] == IdeaActionType.THINK.value
    assert not (workdir / "nope.txt").exists()


@pytest.mark.asyncio
async def test_never_registered_is_unreachable(sandbox, workdir):
    """Even if a name is somehow allowed, an uninstalled action cannot run."""
    engine = _engine(sandbox, allowed=["think", "write_file"], install=False)
    graph = IdeaDag(root_title="root")

    result = await _run(engine, graph, "write_file", {"path": "nope.txt", "content": "x"})

    assert result["action"] == IdeaActionType.THINK.value
    assert not (workdir / "nope.txt").exists()


@pytest.mark.asyncio
async def test_no_sandbox_on_the_run_means_no_filesystem_at_all(tmp_path):
    """The gate that survives every mistake above it: a run whose AgentIO carries no sandbox has
    no filesystem access, whatever the plan says."""
    engine = _engine(None)
    graph = IdeaDag(root_title="root")
    target = tmp_path / "pwned.txt"

    result = await _run(engine, graph, "write_file", {"path": str(target), "content": "x"})

    assert result["action"] == "write_file"
    assert result["success"] is False
    assert "no sandbox" in result["error"]
    assert not target.exists()


def test_web_research_agent_io_has_no_sandbox_by_default():
    from agent.app.agent_io import AgentIO

    io = AgentIO(connector_llm=None, connector_search=None, connector_http=None,
                 connector_chroma=None)

    assert io.connector_sandbox is None


def test_sandbox_actions_describe_themselves_for_the_prompt():
    """These only become selectable when the expansion menu explains them, so every one must
    carry a description and an args hint."""
    for cls in SandboxToolPack.ACTION_CLASSES:
        assert cls.menu_description(), f"{cls.__name__} has no description"
        assert cls.args_hint, f"{cls.__name__} has no args hint"
        assert cls.menu_line().startswith(f"- {cls.name}: details=")


@pytest.mark.asyncio
async def test_allowed_sandbox_actions_are_described_in_the_expansion_prompt(sandbox):
    engine = _engine(sandbox)
    graph = IdeaDag(root_title="root")

    prompt = engine.expansion._build_messages(graph, graph.get_node(graph.root_id()))[0]["content"]

    assert "- write_file: details={path, content}." in prompt
    assert "- find_files: details={name:" in prompt
