"""Unit tests for the action IR: parse, route, fill/validate, repair. No LLM."""
from localagent.actions import ActionRegistry, ActionSpec, Slot
from localagent.catalog import build_default_registry
from localagent.ir import (INVALID_ENUM, MISSING_REQUIRED_SLOT, UNKNOWN_ACTION,
                      UNKNOWN_ENTITY, WRONG_ENTITY_KIND, fill_and_validate,
                      parse_block, repair_prompt, route)
from localagent.state import AgentState


def _enum_reg():
    return ActionRegistry().register(ActionSpec(
        "RUN_TEST", "run a test target", tool="shell",
        slots=[Slot("target", "enum", enum_values=["unit", "integration", "all"]),
               Slot("verbose", "bool", required=False, default=False)]))


def test_parse_line_form():
    a, kv, pos = parse_block("ACTION: READ_FILE\nfile: F1")
    assert a == "READ_FILE" and kv == {"file": "F1"} and pos == []


def test_parse_bare_and_fences():
    a, kv, pos = parse_block("```\nread file\npath: .\n```")
    assert a == "READ_FILE" and kv == {"path": "."}


def test_parse_pipe_form():
    a, kv, pos = parse_block("search_docs|redis streams ack")
    assert a == "SEARCH_DOCS" and pos == ["redis streams ack"]


def test_route_unknown_lists_allowed():
    reg = build_default_registry()
    spec, err = route("FLY", reg, AgentState())
    assert spec is None and err.code == UNKNOWN_ACTION
    assert "LIST_DIR" in err.allowed and "READ_FILE" not in err.allowed  # gated: no files yet


def test_route_gating_exposes_read_once_file_known():
    reg = build_default_registry()
    s = AgentState()
    s.add_entity("file", "a.txt", "/w/a.txt")
    spec, err = route("READ_FILE", reg, s)
    assert err is None and spec.name == "READ_FILE"


def test_enum_invalid():
    reg = _enum_reg()
    spec = reg.get("RUN_TEST")
    filled, errors = fill_and_validate(spec, {"target": "smoke"}, [], AgentState())
    assert errors and errors[0].code == INVALID_ENUM
    assert errors[0].allowed == ["unit", "integration", "all"]


def test_enum_valid_case_insensitive_and_default():
    reg = _enum_reg()
    spec = reg.get("RUN_TEST")
    filled, errors = fill_and_validate(spec, {"target": "ALL"}, [], AgentState())
    assert not errors and filled["target"] == "all" and filled["verbose"] is False


def test_unknown_entity_and_wrong_kind():
    reg = build_default_registry()
    s = AgentState()
    s.add_entity("dir", "sub", "/w/sub")           # a dir, not a file
    spec = reg.get("READ_FILE")
    _, e1 = fill_and_validate(spec, {"file": "F9"}, [], s)
    assert e1[0].code == UNKNOWN_ENTITY
    _, e2 = fill_and_validate(spec, {"file": "D1"}, [], s)   # D1 is a dir
    assert e2[0].code == WRONG_ENTITY_KIND


def test_missing_required():
    reg = build_default_registry()
    spec = reg.get("WRITE_FILE")
    _, errors = fill_and_validate(spec, {"path": "a.txt"}, [], AgentState())
    assert errors[0].code == MISSING_REQUIRED_SLOT and errors[0].slot == "content"


def test_positional_mapping():
    reg = build_default_registry()
    spec = reg.get("WRITE_FILE")
    filled, errors = fill_and_validate(spec, {}, ["a.txt", "hello"], AgentState())
    assert not errors and filled == {"path": "a.txt", "content": "hello"}


def test_repair_prompt_targets_one_slot():
    reg = build_default_registry()
    spec = reg.get("WRITE_FILE")
    _, errors = fill_and_validate(spec, {"path": "a.txt"}, [], AgentState())
    prompt, target = repair_prompt(spec, errors, {"path": "a.txt"})
    assert target == "content"
    assert "content" in prompt and "WRITE_FILE" in prompt
    assert "INVALID_ACTION" not in prompt           # positive phrasing, not an error dump
