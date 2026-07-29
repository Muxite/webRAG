"""Progress-narration layer — a human-facing "what am I doing" stream.

PURPOSE: make a watcher feel that steady, thoughtful progress is being made. It is a
COSMETIC UX channel, deliberately rough and sometimes embellished with filler
"thinking" lines. It is NOT the audit trail: the structured run trace (loop.RunResult)
and the container logs are the source of truth. Nothing downstream (scoring,
verification, safety) reads this stream — so a stray or vague line here is harmless.

Design: the loop calls semantic hooks (chose/running/observed/…); the narrator maps
each to a friendly first-person line drawn from a rotating phrase bank (deterministic,
no RNG, so tests are stable), optionally interleaving generic pondering filler. Lines
are pushed to pluggable sinks immediately (streaming) — stdout now; SSE/websocket/Redis
later behind the same Sink protocol.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol


@dataclass
class NarrationEvent:
    kind: str          # thinking | action | running | observed | repairing | finished
    text: str          # the friendly, watcher-facing line
    t: float


class Sink(Protocol):
    def __call__(self, ev: NarrationEvent) -> None: ...


class ListSink:
    """Captures events in memory (for tests / replay)."""
    def __init__(self) -> None:
        self.events: List[NarrationEvent] = []

    def __call__(self, ev: NarrationEvent) -> None:
        self.events.append(ev)

    def lines(self) -> List[str]:
        return [e.text for e in self.events]


class StdoutSink:
    def __init__(self, prefix: str = "· ") -> None:
        self.prefix = prefix

    def __call__(self, ev: NarrationEvent) -> None:
        print(f"{self.prefix}{ev.text}", flush=True)


# Rotating phrase banks. Keys are semantic stages; %s slots get a short subject.
_THINKING = [
    "Let me think about the best way to do this.",
    "Weighing a couple of options here.",
    "Getting my bearings on what's being asked.",
    "Cross-checking what I have so far.",
    "Planning the next move.",
]
_ACTION = {
    "file": ["Working with the files now.", "Let me sort out the files.", "Handling the file side of this."],
    "memory": ["Checking what I remember.", "Saving that so I don't forget.", "Consulting my notes."],
    "web": ["Looking that up.", "Reading up on this.", "Searching for the details."],
    "shell": ["Running a quick check.", "Inspecting the environment.", "Taking a look around."],
    "finish": ["Wrapping this up.", "Putting the answer together.", "Finalizing my response."],
    "_": ["On it.", "Taking the next step.", "Making progress."],
}
_OBSERVED = [
    "Got it — that's useful.",
    "Okay, that tells me something.",
    "Noted. Moving on.",
    "That worked; carrying on.",
]
_REPAIR = [
    "Hmm, let me fix that.",
    "Adjusting my approach.",
    "One correction and I'll continue.",
]
_FINISHED = [
    "All done.",
    "That should do it.",
    "Finished — here's the result.",
]


def _pick(bank: List[str], i: int) -> str:
    return bank[i % len(bank)] if bank else ""


@dataclass
class Narrator:
    sinks: List[Sink] = field(default_factory=list)
    filler_every: int = 0          # emit a generic "thinking" line every N steps (0 = off)
    _i: int = 0

    def add_sink(self, sink: Sink) -> "Narrator":
        self.sinks.append(sink)
        return self

    def _emit(self, kind: str, text: str) -> None:
        ev = NarrationEvent(kind=kind, text=text, t=time.time())
        for s in self.sinks:
            try:
                s(ev)
            except Exception:
                pass       # narration must never perturb the run

    # --- semantic hooks the loop calls -----------------------------------
    def thinking(self, hint: str = "") -> None:
        self._i += 1
        line = hint or _pick(_THINKING, self._i)
        self._emit("thinking", line)

    def chose(self, tool: str, label: str = "") -> None:
        self._i += 1
        base = _pick(_ACTION.get(tool, _ACTION["_"]), self._i)
        self._emit("action", f"{base} {label}".strip())

    def running(self, tool: str) -> None:
        if self.filler_every and self._i % self.filler_every == 0:
            self._emit("thinking", _pick(_THINKING, self._i + 1))

    def observed(self, _summary: str = "") -> None:
        self._i += 1
        self._emit("observed", _pick(_OBSERVED, self._i))

    def repairing(self) -> None:
        self._i += 1
        self._emit("repairing", _pick(_REPAIR, self._i))

    def finished(self) -> None:
        self._emit("finished", _pick(_FINISHED, self._i))
