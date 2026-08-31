#!/usr/bin/env python3
"""
trace_read.py: read a ``TraceRecorder`` JSONL trace back into a per-call table.

Nothing in this repo read a trace back until this script -- ``agent/app/trace_recorder.py``
only ever WRITES (``TraceRecorder.record``), and the result-JSON summarizer
(``agent/app/testing/utils.py``'s ``slim_telemetry_raw``) deliberately strips raw prompt/
completion text on the way into the per-cell result JSON. When full LLM-I/O capture is on
(``IDEA_TEST_CAPTURE_LLM_IO=1``) and traces are retained (``IDEA_TEST_KEEP_TRACES=1``), the
trace JSONL is the only durable place the real prompt/completion text lives -- this is the
reader that makes that usable (e.g. for a replay tool that reconstructs an LLM payload from a
call's ``in`` event).

Each line of a trace is one ``TraceRecorder.record()`` entry: ``{"ts", "event", "payload"}``.
An LLM call shows up as a PAIR of ``event == "connector_io"`` lines -- one ``direction: "in"``
(the request) and one ``direction: "out"`` (the response or error) -- correlated by
``payload["call_id"]`` (a process-wide monotonic id allocated once per call by
``ConnectorBase._next_call_id``, shared by both events; see ``connector_base.py``). Pairing by
``call_id`` -- never by file order -- matters because concurrent calls (the in-flight LLM
semaphore alone allows up to 32) can interleave their events.

CLI usage:  python scripts/trace_read.py <trace.jsonl> [--json]
    Prints a one-line-per-call summary table; ``--json`` dumps the full parsed calls instead.

Programmatic usage:
    from trace_read import load_trace, calls_from_trace
    events = load_trace("run_152_qwen2.5:7b_graph.jsonl")   # -> List[Event], file order
    calls = calls_from_trace("run_152_qwen2.5:7b_graph.jsonl")  # -> List[Call], call_id-paired
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class Event:
    """One raw line of a ``TraceRecorder`` JSONL trace, in file order."""

    ts: float
    event: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Call:
    """One LLM (or other connector) call, reconstructed by pairing its ``in``/``out`` events.

    ``call_id`` is the trace's own id when present (int, from ``ConnectorBase._next_call_id``);
    for a legacy trace written before call ids existed, it falls back to a synthetic per-file
    string key so pairing degrades gracefully instead of crashing -- callers should treat a
    string ``call_id`` as "best effort, no concurrency guarantee" (there is nothing else in an
    old trace to correlate on but file order, which is exactly the thing this reader exists to
    stop relying on).
    """

    call_id: Union[int, str, None]
    connector: Optional[str] = None
    operation: Optional[str] = None
    stage: Optional[str] = None
    node_id: Optional[str] = None
    t_start: Optional[float] = None
    t_end: Optional[float] = None
    model: Optional[str] = None
    prompt_text: Optional[str] = None
    completion_text: Optional[str] = None
    prompt_chars: Optional[int] = None
    completion_chars: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    num_ctx: Optional[int] = None
    response_format: Any = None
    attempts: Optional[int] = None
    error: Optional[str] = None
    in_payload: Dict[str, Any] = field(default_factory=dict)
    out_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Wall-clock seconds between the ``in`` and ``out`` events, or ``None`` if unpaired."""
        if self.t_start is None or self.t_end is None:
            return None
        return max(0.0, self.t_end - self.t_start)

    @property
    def paired(self) -> bool:
        """True when both the request and the response/error side of this call were found."""
        return self.t_start is not None and self.t_end is not None


def load_trace(path: Union[str, Path]) -> List[Event]:
    """
    Parse a ``TraceRecorder`` JSONL trace into its raw events, in file order.

    :param path: Path to the ``.jsonl`` trace file.
    :returns: One :class:`Event` per non-blank line. A line that fails to parse as JSON is
        skipped rather than raising, so one corrupt/truncated line (e.g. a crash mid-write)
        does not lose every event before it.
    """
    events: List[Event] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            events.append(
                Event(
                    ts=float(raw.get("ts") or 0.0),
                    event=str(raw.get("event") or ""),
                    payload=raw.get("payload") if isinstance(raw.get("payload"), dict) else {},
                )
            )
    return events


def calls_from_trace(path: Union[str, Path]) -> List[Call]:
    """
    Build the per-call table from a trace, pairing ``in``/``out`` ``connector_io`` events.

    Only ``event == "connector_io"`` lines participate (the events ``ConnectorBase._record_io``
    writes for every LLM/search/http/chroma call); other event kinds (``decision``, ``timing``,
    ``document_seen``, ``summary``, ...) are ignored here.

    :param path: Path to the ``.jsonl`` trace file.
    :returns: One :class:`Call` per distinct ``call_id`` seen, in FIRST-SEEN order (not
        necessarily file order once concurrent calls interleave -- callers that need wall-clock
        ordering should sort on ``t_start``).
    """
    calls: "Dict[Union[int, str], Call]" = {}
    order: List[Union[int, str]] = []
    fallback_seq = 0

    for ev in load_trace(path):
        if ev.event != "connector_io":
            continue
        outer = ev.payload
        call_id = outer.get("call_id")
        if call_id is None:
            # Legacy trace (no call_id): synthesize a private key per event so a single
            # unmatched event still surfaces as its own row instead of silently merging with
            # an unrelated call. This has no concurrency guarantee -- see the Call docstring.
            fallback_seq += 1
            call_id = f"_no_call_id_{fallback_seq}"

        call = calls.get(call_id)
        if call is None:
            call = Call(call_id=call_id)
            calls[call_id] = call
            order.append(call_id)

        call.connector = call.connector or outer.get("connector")
        call.operation = call.operation or outer.get("operation")
        call.stage = call.stage or outer.get("stage")
        call.node_id = call.node_id or outer.get("node_id")

        inner = outer.get("payload") if isinstance(outer.get("payload"), dict) else {}
        direction = outer.get("direction")
        if direction == "in":
            call.t_start = ev.ts
            call.in_payload = inner
            call.model = call.model or inner.get("model")
            call.prompt_text = inner.get("prompt_text")
            call.prompt_chars = inner.get("prompt_chars")
            call.temperature = inner.get("temperature")
            call.top_p = inner.get("top_p")
            call.seed = inner.get("seed")
            call.num_ctx = inner.get("num_ctx")
            call.response_format = inner.get("response_format")
        elif direction == "out":
            call.t_end = ev.ts
            call.out_payload = inner
            call.model = call.model or inner.get("model")
            call.completion_text = inner.get("completion_text")
            call.completion_chars = inner.get("completion_chars")
            call.attempts = inner.get("attempts")
            if outer.get("error"):
                call.error = outer.get("error")

    return [calls[cid] for cid in order]


def _format_row(call: Call) -> str:
    duration = f"{call.duration:.3f}s" if call.duration is not None else "?"
    prompt_len = len(call.prompt_text) if isinstance(call.prompt_text, str) else call.prompt_chars
    completion_len = (
        len(call.completion_text) if isinstance(call.completion_text, str) else call.completion_chars
    )
    status = "ERROR" if call.error else ("ok" if call.paired else "unpaired")
    return (
        f"call_id={call.call_id!r:<8} stage={call.stage or '-':<12} node={call.node_id or '-':<8} "
        f"model={call.model or '-':<20} attempts={call.attempts if call.attempts is not None else '-'} "
        f"seed={call.seed if call.seed is not None else '-':<6} dur={duration:<9} "
        f"prompt_chars={prompt_len if prompt_len is not None else '-'} "
        f"completion_chars={completion_len if completion_len is not None else '-'} status={status}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Read a TraceRecorder JSONL trace into paired calls.")
    parser.add_argument("trace_path", help="Path to the .jsonl trace file")
    parser.add_argument("--json", action="store_true", help="Dump the full parsed calls as JSON")
    args = parser.parse_args(argv)

    calls = calls_from_trace(args.trace_path)
    if args.json:
        json.dump([asdict(c) for c in calls], sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    if not calls:
        print(f"No connector_io calls found in {args.trace_path}")
        return 0
    for call in calls:
        print(_format_row(call))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
