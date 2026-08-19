"""Thought buffer for agent-debug: surfaces raw telemetry per engine step.

Bridges the gap between ``TelemetrySession`` (which accumulates decisions/
events for the whole run) and the stepping debugger (which wants "what did
the agent think during step N"). Watermark the telemetry lists before a step
runs, then diff after it finishes to get the slice that belongs to that step.

Everything here is defensive: telemetry may be disabled (``enabled=False``),
records may be summarized (``connector_io`` payloads become ``{chars: N}``
shapes when full-capture is off), and callers may hand in malformed or
partial records. None of that should ever raise inside a debugger session.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class Thought:
    """Everything telemetry recorded during one engine step."""

    step_index: int
    node_id: Optional[str]
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    # True when at least one connector_io payload in this step was summarized
    # ({chars: N} / {count: N} shapes) rather than raw text, i.e. full-capture
    # was not in effect (or the connector never records raw text at all).
    truncated: bool = False


def _looks_summarized(value: Any) -> bool:
    """Detect ConnectorBase._summarize_payload's output shapes.

    :param value: A single payload field value.
    :return: True if the value looks like a summarized string/list/dict.
    """
    if not isinstance(value, dict):
        return False
    keys = set(value.keys())
    return keys in ({"chars"}, {"count"}, {"keys", "count"})


def _safe_list(obj: Any, attr: str) -> List[Any]:
    """Best-effort read of a list attribute off a (possibly fake/None) telemetry object."""
    try:
        val = getattr(obj, attr, None)
    except Exception:
        return []
    if isinstance(val, list):
        return val
    return []


class ThoughtBuffer:
    """Ring buffer of per-step ``Thought`` snapshots pulled from a TelemetrySession.

    Raw prompts/completions can be large, so only the last ``max_steps``
    thoughts are kept in memory.
    """

    def __init__(self, telemetry: Optional[Any], max_steps: int = 20) -> None:
        """
        :param telemetry: TelemetrySession (or telemetry-like duck type) to read from.
            May be None if telemetry is disabled.
        :param max_steps: Ring buffer size (number of Thought objects retained).
        """
        self._telemetry = telemetry
        self._max_steps = max(1, int(max_steps)) if max_steps else 1
        self._watermarks: Dict[int, Dict[str, Any]] = {}
        self._buffer: Deque[Thought] = deque(maxlen=self._max_steps)

    def capture(self, step_index: int, node_id: Optional[str] = None) -> None:
        """Record the current telemetry list lengths as a watermark before a step runs.

        :param step_index: Engine step counter for the step about to run.
        :param node_id: Node id the step is about to act on, if known.
        :return: None
        """
        try:
            self._watermarks[int(step_index)] = {
                "node_id": node_id,
                "decisions_len": len(_safe_list(self._telemetry, "decisions")),
                "events_len": len(_safe_list(self._telemetry, "events")),
            }
        except Exception:
            # Never let a bookkeeping failure interrupt the debugger.
            self._watermarks[step_index] = {"node_id": node_id, "decisions_len": 0, "events_len": 0}

    def collect(self, step_index: int, node_id: Optional[str] = None) -> Thought:
        """Diff telemetry against the step's watermark and build a Thought.

        Appends the result to the ring buffer (evicting the oldest entry once
        ``max_steps`` is exceeded) and returns it.

        :param step_index: Engine step counter that just finished.
        :param node_id: Node id the step acted on, if known (falls back to the
            node_id passed to ``capture`` for this step).
        :return: The collected Thought (always returned, never None).
        """
        wm = self._watermarks.pop(step_index, None) if isinstance(self._watermarks, dict) else None
        decisions_start = wm.get("decisions_len", 0) if wm else 0
        events_start = wm.get("events_len", 0) if wm else 0
        resolved_node_id = node_id if node_id is not None else (wm.get("node_id") if wm else None)

        decisions = self._slice_since(_safe_list(self._telemetry, "decisions"), decisions_start)
        events = self._slice_since(_safe_list(self._telemetry, "events"), events_start)

        llm_calls, truncated = self._extract_llm_calls(events)

        thought = Thought(
            step_index=step_index,
            node_id=resolved_node_id,
            decisions=decisions,
            events=events,
            llm_calls=llm_calls,
            truncated=truncated,
        )
        self._buffer.append(thought)
        return thought

    @staticmethod
    def _slice_since(items: List[Any], start: int) -> List[Dict[str, Any]]:
        """Return the tail of ``items`` from ``start`` on, tolerating bad indices/entries."""
        try:
            tail = items[start:] if 0 <= start <= len(items) else list(items)
        except Exception:
            tail = []
        out: List[Dict[str, Any]] = []
        for entry in tail:
            if isinstance(entry, dict):
                out.append(entry)
        return out

    @staticmethod
    def _extract_llm_calls(events: List[Dict[str, Any]]) -> "tuple[List[Dict[str, Any]], bool]":
        """Pair up connector_io llm_query in/out events into best-effort llm_calls.

        connector_llm.query_llm currently only records char/word counts (never
        the raw prompt/completion text) regardless of full-capture, so
        prompt_text/completion_text will usually be None. If a connector ever
        starts including raw text fields (e.g. "prompt_text"/"completion_text"/
        "prompt"/"completion"), they are picked up here.

        :param events: Event records since the step's watermark.
        :return: (llm_calls, truncated) tuple.
        """
        calls: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []
        truncated = False

        for entry in events:
            try:
                if not isinstance(entry, dict) or entry.get("event") != "connector_io":
                    continue
                inner = entry.get("payload")
                if not isinstance(inner, dict):
                    continue
                if inner.get("operation") != "llm_query":
                    continue
                raw_payload = inner.get("payload")
                if not isinstance(raw_payload, dict):
                    raw_payload = {}
                if any(_looks_summarized(v) for v in raw_payload.values()):
                    truncated = True

                direction = inner.get("direction")
                if direction == "in":
                    call_in = {
                        "model": raw_payload.get("model"),
                        "prompt_text": raw_payload.get("prompt_text") or raw_payload.get("prompt"),
                        "completion_text": None,
                        "usage": None,
                    }
                    # Opportunistic: carry role-labelled messages when the connector
                    # recorded them (full capture only). Never raise on odd shapes.
                    try:
                        messages = raw_payload.get("messages")
                        if isinstance(messages, list):
                            call_in["messages"] = messages
                    except Exception:
                        pass
                    pending.append(call_in)
                elif direction == "out":
                    call = pending.pop(0) if pending else {
                        "model": raw_payload.get("model"),
                        "prompt_text": None,
                        "completion_text": None,
                        "usage": None,
                    }
                    call["completion_text"] = raw_payload.get("completion_text") or raw_payload.get("completion")
                    if not call.get("model"):
                        call["model"] = raw_payload.get("model")
                    if inner.get("error"):
                        call["error"] = inner.get("error")
                    calls.append(call)
            except Exception:
                # Malformed record: skip it, keep going.
                continue

        # Any "in" events left without a matching "out" (call still in flight,
        # or malformed pairing) are surfaced too, best-effort.
        calls.extend(pending)
        return calls, truncated

    def latest(self) -> Optional[Thought]:
        """
        :return: The most recently collected Thought, or None if the buffer is empty.
        """
        if not self._buffer:
            return None
        return self._buffer[-1]

    def get(self, step_index: int) -> Optional[Thought]:
        """Look up a Thought by step index within the ring buffer.

        :param step_index: Engine step counter.
        :return: The matching Thought, or None if it was never collected or was
            already evicted.
        """
        for thought in self._buffer:
            if thought.step_index == step_index:
                return thought
        return None
