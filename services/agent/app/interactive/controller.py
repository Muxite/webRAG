"""Command parser for agent-debug."""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Optional


class Action(Enum):
    STEP = auto()
    NEXT = auto()
    INFO = auto()
    PRINT = auto()
    LIST = auto()
    GRAPH = auto()
    HELP = auto()
    QUIT = auto()


class Cmd:
    __slots__ = ("action", "arg")

    def __init__(self, action: Action, arg: str = ""):
        self.action = action
        self.arg = arg

    def __repr__(self) -> str:
        return f"Cmd({self.action.name}, {self.arg!r})"


_ALIASES = {
    "s": Action.STEP,
    "step": Action.STEP,
    "n": Action.NEXT,
    "next": Action.NEXT,
    "i": Action.INFO,
    "info": Action.INFO,
    "p": Action.PRINT,
    "print": Action.PRINT,
    "l": Action.LIST,
    "list": Action.LIST,
    "g": Action.GRAPH,
    "graph": Action.GRAPH,
    "h": Action.HELP,
    "help": Action.HELP,
    "?": Action.HELP,
    "q": Action.QUIT,
    "quit": Action.QUIT,
}


class Controller:

    def __init__(
        self,
        prompt_fn: Optional[Callable[[str], str]] = None,
        print_fn: Optional[Callable[..., None]] = None,
    ):
        self._prompt = prompt_fn or input
        self._print = print_fn or print

    def ask(self, label: str = "") -> Cmd:
        prompt_str = f"\n\033[1m(adb:{label})\033[0m \033[36m>\033[0m "
        try:
            raw = self._prompt(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            self._print("\nSession ended.")
            return Cmd(Action.QUIT)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> Cmd:
        if not raw:
            return Cmd(Action.STEP)
        parts = raw.split(None, 1)
        word = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        action = _ALIASES.get(word)
        if action:
            return Cmd(action, rest)
        return Cmd(Action.STEP)
