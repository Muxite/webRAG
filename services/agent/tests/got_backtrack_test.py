"""Tests for backtrack settings and threshold gating.

Tests `should_backtrack` and `find_backtrack_target` behavior in isolation
by stubbing the minimal `IdeaDag` surface they need (just `path_to_root`
and `get_node`/`root_id`). This avoids the bs4/chromadb import chain.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _install_stubs():
    """Stub the modules `got_operations.py` imports so it loads cleanly."""
    if "agent.app.idea_memory" in sys.modules:
        return

    pkg_agent = types.ModuleType("agent")
    pkg_app = types.ModuleType("agent.app")
    sys.modules["agent"] = pkg_agent
    sys.modules["agent.app"] = pkg_app

    # idea_memory: only MemoryManager symbol referenced; provide a stub class.
    idea_memory = types.ModuleType("agent.app.idea_memory")

    class _StubMemoryManager:
        async def retrieve_relevant_memories(self, **kw): return []
        async def write_memory(self, **kw): return True
    idea_memory.MemoryManager = _StubMemoryManager
    sys.modules["agent.app.idea_memory"] = idea_memory

    # idea_dag: minimal IdeaDag + IdeaNode so got_operations can type-hint.
    idea_dag = types.ModuleType("agent.app.idea_dag")

    @dataclass
    class _StubIdeaNode:
        node_id: str
        title: str = ""
        score: Optional[float] = None
        parent_id: Optional[str] = None
        details: Dict[str, Any] = field(default_factory=dict)

    class _StubIdeaDag: ...

    idea_dag.IdeaDag = _StubIdeaDag
    idea_dag.IdeaNode = _StubIdeaNode
    sys.modules["agent.app.idea_dag"] = idea_dag

    # idea_policies + base. Give the stub package a real __path__ so that
    # genuinely lightweight submodules (e.g. `config`, pure-stdlib dataclasses)
    # load from disk, while heavy ones stay stubbed via sys.modules below.
    pkg_policies = types.ModuleType("agent.app.idea_policies")
    pkg_policies.__path__ = [str(Path(__file__).resolve().parent.parent / "app" / "idea_policies")]
    sys.modules["agent.app.idea_policies"] = pkg_policies
    base_mod = types.ModuleType("agent.app.idea_policies.base")

    class _StubDetailKey:
        ACTION = type("E", (), {"value": "action"})
        JUSTIFICATION = type("E", (), {"value": "justification"})
        ACTION_RESULT = type("E", (), {"value": "action_result"})

    class _StubIdeaNodeStatus:
        class _S:
            def __init__(self, v): self.value = v
        DONE = _S("done")
        SKIPPED = _S("skipped")
        FAILED = _S("failed")
        ACTIVE = _S("active")
        PENDING = _S("pending")
        BLOCKED = _S("blocked")

    base_mod.DetailKey = _StubDetailKey
    base_mod.IdeaNodeStatus = _StubIdeaNodeStatus
    sys.modules["agent.app.idea_policies.base"] = base_mod

    # action_constants
    ac_mod = types.ModuleType("agent.app.idea_policies.action_constants")
    class _Extractor:
        @staticmethod
        def get_action(d): return (d or {}).get("action")
    ac_mod.NodeDetailsExtractor = _Extractor
    sys.modules["agent.app.idea_policies.action_constants"] = ac_mod

    # agent_io
    io_mod = types.ModuleType("agent.app.agent_io")
    class _StubAgentIO: ...
    io_mod.AgentIO = _StubAgentIO
    sys.modules["agent.app.agent_io"] = io_mod


_install_stubs()


def _load_got():
    here = Path(__file__).resolve().parent
    target = here.parent / "app" / "got_operations.py"
    spec = importlib.util.spec_from_file_location(
        "agent.app.got_operations",
        target,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent.app.got_operations"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_got_mod = _load_got()


# ---- minimal in-memory graph that satisfies what should_backtrack needs ----

from agent.app.idea_dag import IdeaNode as _IdeaNode  # noqa: E402  the stub


class _Graph:
    """Stub IdeaDag exposing only `get_node`, `path_to_root`, `root_id`."""

    def __init__(self, nodes: List[_IdeaNode], root_id: str):
        self._nodes = {n.node_id: n for n in nodes}
        self._root_id = root_id

    def get_node(self, nid: str): return self._nodes.get(nid)
    def root_id(self) -> str: return self._root_id

    def path_to_root(self, nid: str) -> List[_IdeaNode]:
        out = []
        cursor = self._nodes.get(nid)
        while cursor is not None and cursor.node_id != self._root_id:
            out.append(cursor)
            cursor = self._nodes.get(cursor.parent_id or "")
        if cursor is not None:
            out.append(cursor)
        return out


def _make_got(settings: Dict[str, Any]):
    return _got_mod.GoTOperations(settings=settings, io=None, memory_manager=None)


# ---- tests ----

def test_should_backtrack_disabled_returns_false():
    got = _make_got({"got_backtrack_enabled": False})
    graph = _Graph([_IdeaNode("r")], root_id="r")
    assert got.should_backtrack(graph, "r") is False


def test_should_backtrack_fires_at_threshold():
    # 3 consecutive low-score nodes, threshold 3, low_score 0.3 → fires.
    settings = {
        "got_backtrack_enabled": True,
        "got_backtrack_dead_end_threshold": 3,
        "got_backtrack_low_score_threshold": 0.3,
    }
    got = _make_got(settings)
    graph = _Graph(
        [
            _IdeaNode("r", score=None, parent_id=None),
            _IdeaNode("a", score=0.1, parent_id="r"),
            _IdeaNode("b", score=0.2, parent_id="a"),
            _IdeaNode("c", score=0.05, parent_id="b"),
        ],
        root_id="r",
    )
    assert got.should_backtrack(graph, "c") is True


def test_should_backtrack_respects_higher_threshold():
    # Threshold=7, only 3 consecutive low → does NOT fire.
    settings = {
        "got_backtrack_enabled": True,
        "got_backtrack_dead_end_threshold": 7,
        "got_backtrack_low_score_threshold": 0.3,
    }
    got = _make_got(settings)
    graph = _Graph(
        [
            _IdeaNode("r"),
            _IdeaNode("a", score=0.1, parent_id="r"),
            _IdeaNode("b", score=0.1, parent_id="a"),
            _IdeaNode("c", score=0.1, parent_id="b"),
        ],
        root_id="r",
    )
    assert got.should_backtrack(graph, "c") is False


def test_should_backtrack_respects_lower_score_threshold():
    # Score 0.25 NOT considered low when threshold is 0.2. → does NOT fire.
    settings = {
        "got_backtrack_enabled": True,
        "got_backtrack_dead_end_threshold": 3,
        "got_backtrack_low_score_threshold": 0.2,
    }
    got = _make_got(settings)
    graph = _Graph(
        [
            _IdeaNode("r"),
            _IdeaNode("a", score=0.25, parent_id="r"),
            _IdeaNode("b", score=0.25, parent_id="a"),
            _IdeaNode("c", score=0.25, parent_id="b"),
        ],
        root_id="r",
    )
    assert got.should_backtrack(graph, "c") is False


def test_find_backtrack_target_returns_parent_of_first_decent_ancestor():
    settings = {
        "got_backtrack_enabled": True,
        "got_backtrack_low_score_threshold": 0.3,
    }
    got = _make_got(settings)
    graph = _Graph(
        [
            _IdeaNode("r"),
            _IdeaNode("a", score=0.8, parent_id="r"),  # decent ancestor; we want its parent (root)
            _IdeaNode("b", score=0.1, parent_id="a"),
            _IdeaNode("c", score=0.1, parent_id="b"),
        ],
        root_id="r",
    )
    target = got.find_backtrack_target(graph, "c")
    assert target == "r"  # parent of "a" is the root


def test_find_backtrack_target_falls_back_to_root_when_no_decent_ancestor():
    settings = {
        "got_backtrack_enabled": True,
        "got_backtrack_low_score_threshold": 0.3,
    }
    got = _make_got(settings)
    graph = _Graph(
        [
            _IdeaNode("r"),
            _IdeaNode("a", score=0.1, parent_id="r"),
            _IdeaNode("b", score=0.1, parent_id="a"),
        ],
        root_id="r",
    )
    target = got.find_backtrack_target(graph, "b")
    assert target == "r"


def test_should_backtrack_increments_dead_end_count():
    settings = {
        "got_backtrack_enabled": True,
        "got_backtrack_dead_end_threshold": 2,
        "got_backtrack_low_score_threshold": 0.3,
    }
    got = _make_got(settings)
    graph = _Graph(
        [
            _IdeaNode("r"),
            _IdeaNode("a", score=0.1, parent_id="r"),
            _IdeaNode("b", score=0.1, parent_id="a"),
        ],
        root_id="r",
    )
    initial = got._dead_end_count
    assert got.should_backtrack(graph, "b") is True
    assert got._dead_end_count == initial + 1
