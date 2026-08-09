"""TracedAgent(Agent) + SessionTraceRecorder(TraceRecorder) — wires full-transcript
logging and per-mandate attribution summaries into basic_cli.py's REPL loop without
touching agent.app.basic_cli or agent.app.agent.

`Agent` already accepts `tracer` and `idea_dag_settings` kwargs; basic_cli.py's own
`Agent(...)` call site (basic_cli.py:160) never passes either. chat_entrypoint.py
monkeypatches `basic_cli.Agent = TracedAgent` (a confirmed plain, reassignable module
global — basic_cli.py imports it as `from agent.app.agent import Agent` and calls it
only as a bare name inside main()), so every mandate constructed by the unmodified REPL
loop picks up tracing and settings for free via `kwargs.setdefault`.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from agent.app.agent import Agent
from agent.app.trace_recorder import TraceRecorder

from playground import session_log


class SessionTraceRecorder(TraceRecorder):
    """A TraceRecorder that stamps every event with session_id + the current
    mandate_id, so a multi-mandate session's transcript is still filterable per-mandate.
    """

    def __init__(self, path: Path, session_id: str) -> None:
        super().__init__(path)
        self.session_id = session_id
        self.mandate_id: Optional[str] = None

    def record(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        super().record(
            event,
            {**(payload or {}), "session_id": self.session_id, "mandate_id": self.mandate_id},
        )


class JsonlLogHandler(logging.Handler):
    """Mirrors every stdlib `logging` record into a SessionTraceRecorder.

    `Agent.tracer` (SessionTraceRecorder above) is only exercised via the explicit
    `self.tracer.record(...)` calls sprinkled through `Agent.run()`'s OLD manual
    tick-loop — but that whole loop is skipped whenever `use_idea_dag` is true (always,
    in this playground), replaced by a freshly-constructed `IdeaDagEngine(...)` that is
    never given a tracer at all (confirmed by reading agent.app.agent.Agent.run()). So
    without this handler, the transcript would contain exactly one "init" line
    regardless of how much the engine actually did — confirmed by the live smoke test.

    `IdeaDagEngine` and its policies (LlmExpansionPolicy, SearchLeafAction, etc.) already
    log every step via the stdlib `logging` module — the same lines a friend sees on
    their terminal. Attaching this handler to the root logger persists that exact detail
    to the transcript file too, with zero changes to agent/app/*.
    """

    def __init__(self, recorder: SessionTraceRecorder, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self.recorder = recorder

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.recorder.record(
                "log",
                {
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                },
            )
        except Exception:
            pass  # a logging failure must never break a friend's chat session


class TracedAgent(Agent):
    """Agent subclass that injects a shared, session-scoped tracer + resolved
    idea_dag_settings into every mandate, and appends a summary row after each run().

    The `shared_*` class attributes are set ONCE by chat_entrypoint.py before
    basic_cli.main() starts its REPL loop — that loop constructs a fresh `Agent(...)`
    per mandate but never passes `tracer=`/`idea_dag_settings=`, so `kwargs.setdefault`
    is what makes every mandate in the session inherit them without any change to
    basic_cli.py itself.
    """

    shared_tracer: Optional[SessionTraceRecorder] = None
    shared_settings: Optional[Dict[str, Any]] = None
    shared_tier: str = ""
    shared_profile: str = ""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("tracer", TracedAgent.shared_tracer)
        kwargs.setdefault("idea_dag_settings", TracedAgent.shared_settings)
        super().__init__(*args, **kwargs)
        self._mandate_id = uuid.uuid4().hex[:12]
        if isinstance(self.tracer, SessionTraceRecorder):
            self.tracer.mandate_id = self._mandate_id

    async def run(self) -> Dict[str, Any]:
        started_at = time.time()
        result = await super().run()
        try:
            transcript_file = str(self.tracer.path) if self.tracer else ""
            session_log.append_summary_row(
                session_id=getattr(self.tracer, "session_id", "unknown"),
                mandate_id=self._mandate_id,
                tier=TracedAgent.shared_tier,
                model=self.model_name,
                profile=TracedAgent.shared_profile,
                mandate_text=self.mandate,
                started_at=started_at,
                result=result,
                transcript_file=transcript_file,
            )
        except Exception:
            # A logging failure must never break a friend's chat session.
            self._logger.exception("Failed to write session summary row")
        return result
