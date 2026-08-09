"""Tests for the agent-debug command parser (interactive/controller.py)."""
from __future__ import annotations

import pytest

from agent.app.interactive.controller import Action, Cmd, Controller


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("s", Action.STEP),
        ("step", Action.STEP),
        ("n", Action.NEXT),
        ("next", Action.NEXT),
        ("i", Action.INFO),
        ("info", Action.INFO),
        ("p", Action.PRINT),
        ("print", Action.PRINT),
        ("l", Action.LIST),
        ("list", Action.LIST),
        ("g", Action.GRAPH),
        ("graph", Action.GRAPH),
        ("e", Action.EDIT),
        ("edit", Action.EDIT),
        ("f", Action.FEEDBACK),
        ("feedback", Action.FEEDBACK),
        ("h", Action.HELP),
        ("help", Action.HELP),
        ("?", Action.HELP),
        ("q", Action.QUIT),
        ("quit", Action.QUIT),
    ],
)
def test_all_aliases_parse(raw, expected):
    assert Controller._parse(raw).action is expected


def test_alias_is_case_insensitive():
    assert Controller._parse("EDIT").action is Action.EDIT
    assert Controller._parse("Feedback").action is Action.FEEDBACK


def test_empty_input_defaults_to_step():
    assert Controller._parse("").action is Action.STEP
    assert Controller._parse("   ").action is Action.STEP


def test_unknown_word_defaults_to_step():
    assert Controller._parse("frobnicate").action is Action.STEP


def test_argument_captured_after_command():
    cmd = Controller._parse("f steer toward primary sources")
    assert cmd.action is Action.FEEDBACK
    assert cmd.arg == "steer toward primary sources"


def test_info_argument_captured():
    cmd = Controller._parse("i nodes")
    assert cmd.action is Action.INFO
    assert cmd.arg == "nodes"


def test_ask_parses_prompted_line():
    ctrl = Controller(prompt_fn=lambda _p: "n", print_fn=lambda *_a, **_k: None)
    assert ctrl.ask(label="x").action is Action.NEXT


def test_ask_eof_returns_quit():
    def _raise(_p):
        raise EOFError

    ctrl = Controller(prompt_fn=_raise, print_fn=lambda *_a, **_k: None)
    assert ctrl.ask().action is Action.QUIT


def test_ask_keyboard_interrupt_returns_quit():
    def _raise(_p):
        raise KeyboardInterrupt

    ctrl = Controller(prompt_fn=_raise, print_fn=lambda *_a, **_k: None)
    assert ctrl.ask().action is Action.QUIT


def test_read_line_returns_text():
    ctrl = Controller(prompt_fn=lambda _p: "a note", print_fn=lambda *_a, **_k: None)
    assert ctrl.read_line() == "a note"


def test_read_line_eof_returns_empty():
    def _raise(_p):
        raise EOFError

    ctrl = Controller(prompt_fn=_raise, print_fn=lambda *_a, **_k: None)
    assert ctrl.read_line() == ""


def test_read_multiline_blank_ends_block():
    scripted = iter(["line one", "line two", ""])
    ctrl = Controller(prompt_fn=lambda _p: next(scripted), print_fn=lambda *_a, **_k: None)
    assert ctrl.read_multiline() == "line one\nline two"


def test_read_multiline_first_blank_is_empty():
    ctrl = Controller(prompt_fn=lambda _p: "", print_fn=lambda *_a, **_k: None)
    assert ctrl.read_multiline() == ""


def test_read_multiline_eof_returns_gathered():
    scripted = iter(["kept"])

    def _prompt(_p):
        try:
            return next(scripted)
        except StopIteration:
            raise EOFError

    ctrl = Controller(prompt_fn=_prompt, print_fn=lambda *_a, **_k: None)
    assert ctrl.read_multiline() == "kept"


def test_cmd_repr():
    assert repr(Cmd(Action.EDIT, "x")) == "Cmd(EDIT, 'x')"
