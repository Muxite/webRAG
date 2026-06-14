"""Unit tests for the pure-logic plugins in `extra_actions/`.

Loads each action module via importlib so we don't pull in
`agent.app.idea_policies.__init__` (which imports bs4 + chromadb).
The action class needs `LeafAction` from the same package, so we stub
a minimal `LeafAction` base for the duration of the test load.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


# ---- Test fixtures: stub the LeafAction base so action modules import cleanly.

def _install_stub_leaf_action() -> None:
    if "agent.app.idea_policies.actions" in sys.modules:
        return
    pkg_agent = types.ModuleType("agent")
    pkg_app = types.ModuleType("agent.app")
    pkg_policies = types.ModuleType("agent.app.idea_policies")
    actions_mod = types.ModuleType("agent.app.idea_policies.actions")

    class _StubLeafAction:
        def __init__(self, settings: Optional[Dict[str, Any]] = None) -> None:
            self.settings = dict(settings or {})

    actions_mod.LeafAction = _StubLeafAction
    sys.modules["agent"] = pkg_agent
    sys.modules["agent.app"] = pkg_app
    sys.modules["agent.app.idea_policies"] = pkg_policies
    sys.modules["agent.app.idea_policies.actions"] = actions_mod
    # base.py imports nothing problematic, but the per-action files import
    # via 'from agent.app.idea_policies.extra_actions.base import ...'
    # so we set up that package path too.
    pkg_extras = types.ModuleType("agent.app.idea_policies.extra_actions")
    pkg_extras.__path__ = [
        str(Path(__file__).resolve().parent.parent / "ideaengine" / "idea_policies" / "extra_actions")
    ]
    sys.modules["agent.app.idea_policies.extra_actions"] = pkg_extras


_install_stub_leaf_action()


def _load(name: str):
    here = Path(__file__).resolve().parent
    target = here.parent / "ideaengine" / "idea_policies" / "extra_actions" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"agent.app.idea_policies.extra_actions.{name}",
        target,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_load("base")
_regex = _load("regex_extract")
_jp = _load("json_path")
_unit = _load("unit_convert")
_dt = _load("datetime_now")


@dataclass
class _FakeNode:
    node_id: str = "n1"
    details: Dict[str, Any] = field(default_factory=dict)


class _FakeGraph:
    def __init__(self, node: _FakeNode) -> None:
        self._node = node

    def get_node(self, node_id: str) -> _FakeNode | None:
        return self._node if node_id == self._node.node_id else None


def _run(coro):
    return asyncio.run(coro)


# ----- regex_extract -----

def test_regex_extract_simple_matches():
    action = _regex.RegexExtractAction()
    node = _FakeNode(details={"pattern": r"\d+", "text": "I have 3 apples and 12 oranges"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["success"]
    assert res["matches"] == ["3", "12"]
    assert res["count"] == 2


def test_regex_extract_with_flags():
    action = _regex.RegexExtractAction()
    node = _FakeNode(details={"pattern": r"hello", "text": "Hello World", "flags": ["i"]})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["matches"] == ["Hello"]


def test_regex_extract_invalid_pattern():
    action = _regex.RegexExtractAction()
    node = _FakeNode(details={"pattern": "[unclosed", "text": "x"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["success"] is False
    assert res["error_type"] == "InvalidPattern"


def test_regex_extract_missing_inputs():
    action = _regex.RegexExtractAction()
    node = _FakeNode(details={"pattern": r"\d+"})  # text missing
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["success"] is False


# ----- json_path -----

def test_json_path_resolve_dotted():
    data = {"a": {"b": [{"c": 42}]}}
    found, value = _jp.resolve_json_path(data, "a.b[0].c")
    assert found is True and value == 42


def test_json_path_root():
    found, value = _jp.resolve_json_path({"x": 1}, "$")
    assert found is True and value == {"x": 1}


def test_json_path_missing_key():
    found, value = _jp.resolve_json_path({"a": 1}, "a.b.c")
    assert found is False and value is None


def test_json_path_negative_index():
    found, value = _jp.resolve_json_path({"x": [10, 20, 30]}, "x[-1]")
    assert found is True and value == 30


def test_json_path_action_single():
    action = _jp.JsonPathAction()
    node = _FakeNode(details={"json": {"a": {"b": 7}}, "path": "a.b"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["success"] and res["value"] == 7 and res["found"] is True


def test_json_path_action_many():
    action = _jp.JsonPathAction()
    node = _FakeNode(details={"json": {"a": 1, "b": 2}, "paths": ["a", "b", "c"]})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["values"] == {"a": 1, "b": 2}
    assert res["missing"] == ["c"]


def test_json_path_action_parses_json_string():
    action = _jp.JsonPathAction()
    node = _FakeNode(details={"json": '{"k": [1,2,3]}', "path": "k[2]"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["value"] == 3


# ----- unit_convert -----

def test_unit_convert_length():
    action = _unit.UnitConvertAction()
    node = _FakeNode(details={"value": 100, "from_unit": "ft", "to_unit": "m"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["success"]
    assert res["category"] == "length"
    assert abs(res["result"] - 30.48) < 1e-6


def test_unit_convert_mass():
    action = _unit.UnitConvertAction()
    node = _FakeNode(details={"value": 1, "from_unit": "kg", "to_unit": "lb"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert abs(res["result"] - 2.2046226218) < 1e-6


def test_unit_convert_temperature():
    action = _unit.UnitConvertAction()
    node = _FakeNode(details={"value": 100, "from_unit": "C", "to_unit": "F"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert abs(res["result"] - 212.0) < 1e-6
    assert res["category"] == "temperature"


def test_unit_convert_data():
    action = _unit.UnitConvertAction()
    node = _FakeNode(details={"value": 1, "from_unit": "GiB", "to_unit": "MiB"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert abs(res["result"] - 1024.0) < 1e-6


def test_unit_convert_category_mismatch():
    action = _unit.UnitConvertAction()
    node = _FakeNode(details={"value": 1, "from_unit": "kg", "to_unit": "m"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["success"] is False
    assert "category mismatch" in res["error"]


def test_unit_convert_unknown_unit():
    action = _unit.UnitConvertAction()
    node = _FakeNode(details={"value": 1, "from_unit": "smoot", "to_unit": "m"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["success"] is False


# ----- datetime_now -----

def test_datetime_now_default_utc():
    action = _dt.DatetimeNowAction()
    node = _FakeNode(details={})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["success"]
    assert res["tz_offset_hours"] == 0.0
    assert res["iso"].endswith("+00:00")
    assert "weekday" in res
    assert isinstance(res["unix_seconds"], float)


def test_datetime_now_with_offset_and_shift():
    action = _dt.DatetimeNowAction()
    node = _FakeNode(details={"tz_offset_hours": -5, "add_days": 1})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["success"]
    assert res["tz_offset_hours"] == -5
    # Just sanity-check the ISO carries an offset segment.
    assert "-05:00" in res["iso"]


def test_datetime_now_invalid_offset():
    action = _dt.DatetimeNowAction()
    node = _FakeNode(details={"tz_offset_hours": "not-a-number"})
    res = _run(action.execute(_FakeGraph(node), node.node_id, None))
    assert res["success"] is False
