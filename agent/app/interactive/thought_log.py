"""Plain-text audit log of per-step thought cards for the interactive debugger.

Mirrors what the terminal renders (via ``Renderer.thought_card``/``decision_card``)
into an append-only text file, so a session's raw LLM interactions survive after
the terminal scrollback is gone or the process crashes mid-run.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from agent.app.interactive.renderer import Renderer

_log = logging.getLogger(__name__)

_ENV_VAR = "IDEA_THOUGHT_LOG"


class ThoughtLog:
    """Appends rendered (color-free) thought cards to a plain text file.

    Never raises: any IO failure is logged once (at warning level) and the log
    degrades to a silent no-op for the rest of the session, so a broken log path
    can never interrupt debugging.
    """

    def __init__(self, path: str, max_payload_chars: int = 2000, label: str = ""):
        """
        :param path: File path to append thought cards to.
        :param max_payload_chars: Passed through to Renderer.thought_card's clipping.
        :param label: Optional mandate/run label written into the header block.
        """
        self._path = path
        self._max_payload_chars = max_payload_chars
        self._label = label
        self._wrote_header = False
        self._disabled = False
        self._warned = False

    @classmethod
    def from_env(cls, label: str = "", max_payload_chars: int = 2000) -> Optional["ThoughtLog"]:
        """Classmethod alias for the module-level ``from_env()`` factory. See it for details."""
        return from_env(label=label, max_payload_chars=max_payload_chars)

    def write(self, thought: Any) -> None:
        """Render and append one Thought's card (and its decision trail) to the log.

        :param thought: Thought (or duck-typed equivalent) to render.
        :return: None
        """
        if self._disabled:
            return
        try:
            card = Renderer.thought_card(thought, self._max_payload_chars, color=False)
            decisions = getattr(thought, "decisions", None) or []
            decision_card = Renderer.decision_card(decisions, color=False)
            body = card + "\n" + decision_card + "\n"
            with open(self._path, "a", encoding="utf-8") as fh:
                if not self._wrote_header:
                    fh.write(self._header())
                    self._wrote_header = True
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
        except Exception as exc:  # noqa: BLE001. Log IO must never break the debugger
            if not self._warned:
                _log.warning(f"ThoughtLog: failed to write to {self._path}: {exc}")
                self._warned = True
            self._disabled = True

    def _header(self) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        label_line = f"mandate/run: {self._label}\n" if self._label else ""
        return (
            f"=== agent-debug thought log ===\n"
            f"started: {ts}\n"
            f"{label_line}"
            f"entries below are appended after each engine step completes\n"
            f"{'-' * 78}\n"
        )


def from_env(label: str = "", max_payload_chars: int = 2000) -> Optional[ThoughtLog]:
    """Build a ThoughtLog from the IDEA_THOUGHT_LOG env var, if set.

    :param label: Optional mandate/run label to stamp into the header.
    :param max_payload_chars: Passed through to ThoughtLog.
    :return: A ThoughtLog writing to that path, or None if the env var is unset/blank.
    """
    path = os.environ.get(_ENV_VAR)
    if not path or not path.strip():
        return None
    return ThoughtLog(path.strip(), max_payload_chars=max_payload_chars, label=label)
