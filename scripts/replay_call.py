#!/usr/bin/env python3
"""
replay_call.py: reconstruct one LLM call from a trace and re-issue it through
``ConnectorLLM.query_llm``, reporting the original completion next to the replayed one(s).

Companion to ``scripts/trace_read.py`` (Lane D), which this script reuses for pairing a trace's
``in``/``out`` ``connector_io`` events into a :class:`~trace_read.Call` by ``call_id``.
``ConnectorLLM.query_llm`` takes a plain dict payload, so it is the right seam for a replay: no
engine machinery has to be reconstructed, just the wire request.

This ONLY works on a trace captured with ``IDEA_TEST_CAPTURE_LLM_IO=1`` (and
``IDEA_TEST_KEEP_TRACES=1`` to retain the file) -- that is the only capture mode that writes the
per-message ``messages`` list (and the completion text) into the trace. Without it, a trace has
only prompt/completion CHAR COUNTS, which is not enough to reconstruct a payload; this script
refuses rather than guessing.

THE HONEST LIMIT (this is the point of the tool, not a footnote):
  * A seed + temperature > 0 makes a re-run reproducible; a single replay is meaningful there.
  * With NO seed on the original call, a single re-run proves nothing -- ordinary sampling noise
    looks identical to a real behavior change. ``--repeat N`` with N > 1 gives an indicative
    variance estimate instead, and the report says so rather than presenting a one-shot diff as
    evidence.
  * Anthropic's Messages API has no seed parameter at all, so a SEEDED replay is impossible there
    by construction. The report detects the provider and says this instead of silently sending a
    seed that will just be dropped.
  * Lane D found a real cold-start caveat: the FIRST Ollama call after a model loads can return a
    different completion than every WARM call afterward at the same seed. ``--warmup`` issues one
    throwaway call first to mitigate it; the report surfaces the caveat either way when replaying
    against a local Ollama backend with a seed.

CLI usage:
    python scripts/replay_call.py <trace.jsonl> <call_id> \\
        [--replace OLD NEW ...] [--temperature T] [--top-p P] [--seed N] [--max-tokens N] \\
        [--repeat N] [--warmup] [--model NAME] [--json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_read import Call, calls_from_trace  # noqa: E402


class ReplayNotCapturedError(RuntimeError):
    """The trace's ``in`` event has no per-message text to replay from."""


def find_call(trace_path: Union[str, Path], call_id: Union[str, int]) -> Call:
    """Look up one call in a trace by ``call_id`` (matched as both int and str).

    :param trace_path: Path to the ``.jsonl`` trace file.
    :param call_id: The call id to find -- as printed by ``trace_read.py``.
    :returns: The matching :class:`~trace_read.Call`.
    :raises ValueError: if no call in the trace has that id.
    """
    calls = calls_from_trace(trace_path)
    target = str(call_id)
    for call in calls:
        if str(call.call_id) == target:
            return call
    raise ValueError(f"No call with call_id={call_id!r} found in {trace_path}")


def reconstruct_messages(call: Call) -> List[Dict[str, Any]]:
    """Rebuild the exact messages list sent on the original call.

    Only present when the trace was written with ``IDEA_TEST_CAPTURE_LLM_IO=1`` -- that flag is
    the only thing that puts ``in_payload["messages"]`` (role/content pairs) into the trace.
    Without it there is nothing to reconstruct a payload from but char/word counts.

    :param call: The paired :class:`~trace_read.Call` to replay.
    :returns: A fresh list of ``{"role", "content"}`` dicts (safe to mutate).
    :raises ReplayNotCapturedError: if the call's ``in`` event has no ``messages``.
    """
    messages = call.in_payload.get("messages") if isinstance(call.in_payload, dict) else None
    if isinstance(messages, list) and messages:
        return [dict(m) for m in messages]
    raise ReplayNotCapturedError(
        "This trace's call has no captured message text -- only IDEA_TEST_CAPTURE_LLM_IO=1 "
        "(with IDEA_TEST_KEEP_TRACES=1 to retain the file) writes it. Re-run the reference call "
        "with full capture on, then replay from that trace."
    )


def apply_replacements(
    messages: List[Dict[str, Any]], replacements: List[Tuple[str, str]]
) -> List[Dict[str, Any]]:
    """Substring-replace within every string message ``content``, leaving other fields alone.

    This is the "changed prompt" / "changed evidence" mutation lever: a caller can perturb any
    piece of the reconstructed prompt (a quoted page snippet, an instruction) without having to
    retype the whole message list.

    :param messages: Messages as returned by :func:`reconstruct_messages`.
    :param replacements: ``(old, new)`` pairs, applied in order to every message.
    :returns: A NEW list; ``messages`` is not mutated.
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            for old, new in replacements:
                content = content.replace(old, new)
        out.append({**m, "content": content})
    return out


def build_replay_payload(
    call: Call,
    messages: List[Dict[str, Any]],
    *,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the payload dict to hand to ``ConnectorLLM.query_llm``.

    Every sampling parameter falls back to the ORIGINAL call's value from the trace when the
    caller does not explicitly override it, so an unmutated replay reproduces the original call
    as closely as the trace lets it.

    :param call: The original call, for its recorded sampling parameters.
    :param messages: The (possibly mutated) message list to send.
    :param temperature: Override, or ``None`` to reuse the original.
    :param top_p: Override, or ``None`` to reuse the original.
    :param seed: Override, or ``None`` to reuse the original.
    :param max_tokens: Override, or ``None`` to reuse the original.
    :param model: Override, or ``None`` to reuse the original.
    :returns: A plain payload dict.
    """
    payload: Dict[str, Any] = {"messages": messages}
    resolved_model = model if model is not None else call.model
    if resolved_model:
        payload["model"] = resolved_model
    t = temperature if temperature is not None else call.temperature
    if t is not None:
        payload["temperature"] = t
    tp = top_p if top_p is not None else call.top_p
    if tp is not None:
        payload["top_p"] = tp
    s = seed if seed is not None else call.seed
    if s is not None:
        payload["seed"] = s
    mt = max_tokens if max_tokens is not None else (call.in_payload or {}).get("max_tokens")
    if mt is not None:
        payload["max_tokens"] = mt
    rf_type = (call.in_payload or {}).get("response_format_type")
    if rf_type:
        payload["response_format"] = {"type": rf_type}
    return payload


# Backend "kind" strings this module classifies against -- see classify_regime.
KIND_ANTHROPIC = "anthropic"
KIND_OLLAMA_NATIVE = "ollama_native"
KIND_OPENAI_COMPATIBLE = "openai_compatible"

REGIME_SEEDED_DETERMINISTIC = "seeded_deterministic"
REGIME_SEED_UNSUPPORTED_PROVIDER = "seed_unsupported_provider"
REGIME_UNSEEDED_SINGLE_RUN = "unseeded_single_run"
REGIME_UNSEEDED_MULTI_RUN = "unseeded_multi_run"


def classify_regime(backend_kind: str, seed: Optional[int], repeat: int) -> Dict[str, Any]:
    """Say, in plain terms, what standing this particular replay has as evidence.

    This is the honest-limit half of the tool: a diff between an original and a replayed
    completion means something different in each of these regimes, and a reader should not have
    to read the source to tell which one they are looking at.

    :param backend_kind: One of :data:`KIND_ANTHROPIC`, :data:`KIND_OLLAMA_NATIVE`,
        :data:`KIND_OPENAI_COMPATIBLE`.
    :param seed: The seed that will actually be sent (after any ``--seed`` override), or
        ``None``.
    :param repeat: Number of replay calls that will be issued.
    :returns: ``{"regime", "reproducible", "warnings": [...]}``. ``reproducible`` is ``True``,
        ``False``, or ``"indicative"`` (repeats taken, but not proof).
    """
    if backend_kind == KIND_ANTHROPIC:
        return {
            "regime": REGIME_SEED_UNSUPPORTED_PROVIDER,
            "reproducible": False,
            "warnings": [
                "Anthropic's Messages API has no seed parameter -- a seeded replay is "
                "impossible on this provider by construction. Any completion diff below is "
                "ordinary sampling variance, not evidence of a real behavior change."
            ],
        }
    if seed is None:
        if repeat <= 1:
            return {
                "regime": REGIME_UNSEEDED_SINGLE_RUN,
                "reproducible": False,
                "warnings": [
                    "The original call had no seed. A single re-run proves nothing here -- "
                    "pass --repeat N with N well above 1 to see the natural variance before "
                    "treating any diff as meaningful."
                ],
            }
        return {
            "regime": REGIME_UNSEEDED_MULTI_RUN,
            "reproducible": "indicative",
            "warnings": [
                f"The original call had no seed; {repeat} repeats give an INDICATIVE variance "
                "estimate, not proof of reproducibility."
            ],
        }
    warnings: List[str] = []
    if backend_kind == KIND_OLLAMA_NATIVE:
        warnings.append(
            "Cold-start caveat (found live): the FIRST Ollama call after a model load can "
            "return a different completion than every WARM call afterward at the same seed. "
            "Pass --warmup to issue a throwaway call first, or treat a lone mismatch on an "
            "otherwise-warm model as suspect rather than conclusive."
        )
    elif backend_kind == KIND_OPENAI_COMPATIBLE:
        warnings.append(
            "This provider's seed support is documented as best-effort, not guaranteed -- a "
            "provider/model update can still change output at the same seed."
        )
    return {"regime": REGIME_SEEDED_DETERMINISTIC, "reproducible": True, "warnings": warnings}


QueryFn = Callable[[Dict[str, Any]], Awaitable[Optional[str]]]


async def run_replay(
    query_fn: QueryFn,
    payload: Dict[str, Any],
    *,
    repeat: int = 1,
    warmup: bool = False,
) -> Dict[str, Any]:
    """Issue the replay call(s) through an injected async query function.

    :param query_fn: An async callable ``payload -> completion text`` -- in production this is
        a bound ``ConnectorLLM.query_llm``; tests inject a fake.
    :param payload: The payload built by :func:`build_replay_payload`.
    :param repeat: How many timed replay calls to issue and report.
    :param warmup: If True, issue one untimed/unreported call first (mitigates the Ollama
        cold-start caveat instead of just warning about it).
    :returns: ``{"payload": payload, "completions": [str_or_None, ...]}`` -- one entry per
        repeat, in order, excluding the warmup call.
    """
    if warmup:
        await query_fn(dict(payload))
    completions: List[Optional[str]] = []
    for _ in range(max(1, repeat)):
        completions.append(await query_fn(dict(payload)))
    return {"payload": payload, "completions": completions}


def _backend_kind(connector: Any) -> str:
    """Classify a live ``ConnectorLLM``'s backend into one of this module's KIND_* strings."""
    backend_cls_name = type(getattr(connector, "_backend", None)).__name__
    if backend_cls_name == "AnthropicMessagesBackend":
        return KIND_ANTHROPIC
    if backend_cls_name == "OllamaNativeBackend":
        return KIND_OLLAMA_NATIVE
    return KIND_OPENAI_COMPATIBLE


def _build_connector() -> Any:
    """Construct a live ``ConnectorLLM`` from the process environment.

    Kept as its own function (rather than inlined in ``main``) so tests can monkeypatch it and
    inject a fake connector without touching real network/env state.
    """
    from shared.connector_config import ConnectorConfig
    from agent.app.connector_llm import ConnectorLLM

    return ConnectorLLM(ConnectorConfig())


def format_report(
    call: Call, replay: Dict[str, Any], regime: Dict[str, Any], *, stage: str = "replay"
) -> str:
    """Render the original completion next to the replayed one(s), with the regime up front.

    :param call: The original call being replayed.
    :param replay: Result of :func:`run_replay`.
    :param regime: Result of :func:`classify_regime`.
    :param stage: Label the replay was tagged with, for the header line.
    :returns: A human-readable report string.
    """
    lines: List[str] = []
    lines.append("=== replay_call report ===")
    lines.append(
        f"call_id={call.call_id!r} original_stage={call.stage or '-'} model={call.model or '-'}"
    )
    lines.append(
        f"regime={regime['regime']} reproducible={regime['reproducible']!r} "
        f"repeats={len(replay['completions'])}"
    )
    for w in regime["warnings"]:
        lines.append(f"WARNING: {w}")
    lines.append("")
    lines.append("--- ORIGINAL COMPLETION ---")
    lines.append(call.completion_text if call.completion_text is not None else "(not captured)")
    matches = 0
    for i, completion in enumerate(replay["completions"], 1):
        lines.append("")
        lines.append(f"--- REPLAY #{i} ({stage}) ---")
        lines.append(completion if completion is not None else "(no completion)")
        is_match = completion == call.completion_text
        matches += int(is_match)
        lines.append(f"MATCH original: {is_match}")
    total = len(replay["completions"])
    if total:
        lines.append("")
        lines.append(f"SUMMARY: {matches}/{total} replay(s) matched the original exactly.")
    return "\n".join(lines)


async def _amain(args: argparse.Namespace) -> Dict[str, Any]:
    call = find_call(args.trace_path, args.call_id)
    messages = reconstruct_messages(call)
    if args.replace:
        messages = apply_replacements(messages, [tuple(pair) for pair in args.replace])
    payload = build_replay_payload(
        call,
        messages,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_tokens=args.max_tokens,
        model=args.model,
    )
    connector = _build_connector()
    backend_kind = _backend_kind(connector)
    regime = classify_regime(backend_kind, payload.get("seed"), args.repeat)
    try:
        async with connector:
            replay = await run_replay(
                lambda p: connector.query_llm(p, stage="replay"),
                payload,
                repeat=args.repeat,
                warmup=args.warmup,
            )
    finally:
        pass
    return {"call": call, "replay": replay, "regime": regime}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay one LLM call from a trace, with a mutation, through ConnectorLLM."
    )
    parser.add_argument("trace_path", help="Path to the .jsonl trace file")
    parser.add_argument("call_id", help="call_id to replay, as printed by trace_read.py")
    parser.add_argument(
        "--replace", nargs=2, action="append", metavar=("OLD", "NEW"), default=[],
        help="Substring-replace OLD with NEW in every message's content; repeatable.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--repeat", type=int, default=1, help="Number of replay calls to issue.")
    parser.add_argument(
        "--warmup", action="store_true",
        help="Issue one untimed throwaway call before the reported replay(s) (mitigates the "
        "Ollama cold-start caveat).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of the text report.")
    args = parser.parse_args(argv)

    try:
        result = asyncio.run(_amain(args))
    except (ReplayNotCapturedError, ValueError) as e:
        print(f"replay_call: {e}", file=sys.stderr)
        return 1

    if args.json:
        json.dump(
            {
                "call_id": result["call"].call_id,
                "regime": result["regime"],
                "payload": result["replay"]["payload"],
                "original_completion": result["call"].completion_text,
                "replay_completions": result["replay"]["completions"],
            },
            sys.stdout,
            indent=2,
            default=str,
        )
        sys.stdout.write("\n")
    else:
        print(format_report(result["call"], result["replay"], result["regime"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
