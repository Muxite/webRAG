"""Provider-neutral prompted tool-calling shim.

A model with no HTTP ``tools`` parameter (Ollama's ``400 ... does not support tools`` on
tinyllama/phi3:mini/gemma2:2b, OpenRouter's ``404 no endpoints found that support tool use`` on
llama-3.2-1b) can still act like a tool-calling agent if it is simply told, in plain text, what
tools exist and asked to reply with ``{"thought", "action", "args"}`` JSON each turn. That is the
whole mechanism here: a protocol prompt, a tolerant JSON extractor with a deterministic pre-nudge
repair step, and a small step loop that dispatches the parsed action.

This module is deliberately framework-free — no ``langgraph``/``langchain`` import anywhere in
it — so any caller that already speaks plain text/JSON can use it, not only the
``langgraph_react`` comparison arm. It knows nothing about message objects; a caller wires the
loop to its own transcript representation via two callables (``call_model``, ``dispatch_tool``)
and a per-step observer (``on_step``). :mod:`agent.app.langgraph_solver`'s
``_EmulatedToolCallTransport`` is the first consumer: it renders/parses LangChain messages around
this loop but never passes one across the boundary.

Extracted 2026-09-01 from ``langgraph_solver.py``'s ``_EmulatedToolCallTransport`` (which had run,
unmeasured, as the default path for every non-tool-calling model since its introduction — see
``docs/LEDGER_PLAN_2026-09-01.md`` section 5). The extraction changes no behavior: the ~30
existing emulation tests in ``agent/tests/langgraph_toolcall_emulation_test.py`` pass unmodified
against the post-extraction ``langgraph_solver.py``, which is the proof.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple,
)

_logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- data types


@dataclass(frozen=True)
class ToolSpec:
    """One tool as the protocol prompt and loop need to know it — no LangChain ``Tool`` object,
    just its name, a one-line description, and its argument slot names."""

    name: str
    description: str = ""
    arg_names: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCall:
    """A parsed action: which tool, with what arguments, and the thought that led to it."""

    name: str
    args: Dict[str, Any]
    thought: str = ""


@dataclass(frozen=True)
class JsonExtraction:
    """The result of trying to recover a ``{"action": ...}`` object from a model completion.

    :ivar value: The decoded object, or None when nothing in the completion parses as one.
    :ivar source: How it was recovered — ``"direct"`` (the completion itself, no fence, no
        nested span), ``"fenced"`` (inside a ```json fence), ``"span"`` (a balanced ``{...}``
        found inside a larger completion), or ``"failed"`` (nothing parsed).
    :ivar repaired: Whether a deterministic zero-cost repair (see :func:`repair_json_text`) had
        to run before the candidate parsed.
    :ivar repair_attempts: How many repair fixers actually changed the candidate text, whether or
        not the repair ultimately produced something that parsed.
    """

    value: Optional[Dict[str, Any]]
    source: str
    repaired: bool = False
    repair_attempts: int = 0


class ToolLoopExhausted(Exception):
    """The loop ran out of turns without a ``finish`` action or a malformed-output give-up.

    Framework-free by design — the caller translates this into whatever step-budget exception
    its own runtime expects (``langgraph_solver`` re-raises ``GraphRecursionError``, the same
    exception its native transport raises for the same condition).
    """


@dataclass
class ToolLoopStep:
    """One step's outcome, message-free — the caller turns this into its own transcript shape.

    :ivar kind: One of ``"tool_call"``, ``"finish"``, ``"invalid_action"``, ``"malformed_nudge"``
        (unparseable output, budget remains — a nudge should be appended), or
        ``"malformed_give_up"`` (unparseable output, budget exhausted — the loop is returning
        the model's last prose as the transcript's tail).
    :ivar raw_text: The model's raw completion this step.
    :ivar usage: Whatever token-usage object ``call_model`` returned alongside the text (passed
        through untouched; this module never inspects its shape).
    :ivar thought: The parsed ``thought`` field, truncated to the loop's ``thought_chars`` limit.
    :ivar call: The parsed action, when ``kind`` is ``"tool_call"``, ``"finish"``, or
        ``"invalid_action"``.
    :ivar observation: The tool's result text, when ``kind == "tool_call"``.
    :ivar extraction: The :class:`JsonExtraction` this step's decision came from.
    :ivar call_id: A step-scoped id the caller can use to correlate a call with its observation.
    :ivar malformed_count: Consecutive unparseable turns so far, including this one (only
        meaningful when ``kind`` starts with ``"malformed"``).
    """

    kind: str
    raw_text: str
    usage: Any
    thought: str = ""
    call: Optional[ToolCall] = None
    observation: Optional[str] = None
    extraction: Optional[JsonExtraction] = None
    call_id: str = ""
    malformed_count: int = 0


CallModel = Callable[[str], Awaitable[Tuple[str, Any]]]
DispatchTool = Callable[[str, Dict[str, Any]], Awaitable[str]]
RenderView = Callable[[], str]
OnStep = Callable[[ToolLoopStep], None]


# --------------------------------------------------------------------------- protocol prompt

#: How many characters of a step's ``thought`` are kept by default. Mirrors
#: ``execution_sequential._run_react``'s own 300-char thought cap.
THOUGHT_CHARS_DEFAULT = 300
#: Consecutive unparseable model turns tolerated before the loop stops nudging and accepts the
#: model's prose as final. A model that cannot emit JSON at all would otherwise burn its whole
#: step budget on nudges and deliver nothing.
MAX_MALFORMED_TURNS_DEFAULT = 3
#: How many ``{"action": ...}`` spans are attempted when the whole completion is not itself JSON.
#: Bounded so a long prose completion full of braces cannot make extraction quadratic.
MAX_JSON_CANDIDATES = 5

PROTOCOL_HEADER = (
    "You do NOT have a function-calling API. Work ONE step at a time: think, then call exactly "
    "one tool by returning JSON.\nTools:\n"
)
PROTOCOL_FOOTER = (
    "- finish(answer): output the FINAL answer and end the task. Cite the source URLs you used.\n"
    "Each step, return ONLY JSON and nothing else: {\"thought\": \"...\", \"action\": "
    "\"<tool name>\", \"args\": {\"<slot>\": \"...\"}}."
)
STEP_SUFFIX = "Return the next step as JSON."
NUDGE = (
    "Your last message was not valid JSON, so NO tool ran and you made no progress. Reply with "
    "ONLY a JSON object of the form {\"thought\": \"...\", \"action\": \"<tool name>\", "
    "\"args\": {...}} and nothing else — no prose, no markdown fence."
)


def one_line(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def build_protocol(tools: Sequence[ToolSpec]) -> str:
    """The tool menu and action format the loop prompts with.

    A ``finish`` spec (if present) is skipped — the protocol always documents ``finish`` itself
    in :data:`PROTOCOL_FOOTER`, since without it there is no way for the model to stop.

    :param tools: The run's tool specs.
    :returns: The protocol block to append to the caller's own system prompt.
    """
    lines = []
    for t in tools:
        if t.name == "finish":
            continue
        slots = ", ".join(t.arg_names)
        lines.append(f"- {t.name}({slots}): {one_line(t.description)}")
    return PROTOCOL_HEADER + "\n".join(lines) + "\n" + PROTOCOL_FOOTER


def invalid_action_message(names: Sequence[str]) -> str:
    return f"INVALID ACTION — no such tool. Use one of: {'/'.join(list(names) + ['finish'])}."


# --------------------------------------------------------------------------- JSON extraction


def _balanced_span(text: str, start: int) -> Optional[str]:
    """The ``{...}``/``[...]`` span opening at ``start``, or None when it never closes.

    String-aware (a brace inside a JSON string does not change nesting depth), so a page
    quotation embedded in an argument cannot truncate the extraction.
    """
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _json_candidates(text: str) -> List[str]:
    """``text`` itself, then each balanced JSON span inside it (most likely first)."""
    out = [text]
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        span = _balanced_span(text, i)
        if span and span not in out:
            out.append(span)
        if len(out) > MAX_JSON_CANDIDATES:
            break
    return out


def _parse_dict(candidate: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, dict)), None)
    return value if isinstance(value, dict) else None


def _fix_trailing_comma(text: str) -> Optional[str]:
    """``{"a": 1,}`` / ``[1, 2,]`` -> drop the comma before the closer."""
    fixed = re.sub(r",(\s*[}\]])", r"\1", text)
    return fixed if fixed != text else None


def _fix_single_quotes(text: str) -> Optional[str]:
    """A completion that used ``'`` throughout instead of ``"`` (no double quote anywhere, so
    this can't corrupt a real string literal that legitimately contains an apostrophe next to
    real double quotes)."""
    if '"' in text or "'" not in text:
        return None
    return text.replace("'", '"')


def _fix_unterminated_string(text: str) -> Optional[str]:
    """A string literal opened but never closed before the text ends — close it.

    A trailing run of structural characters (``}``, ``]``, whitespace) is treated as intended
    STRUCTURE, not string content: a model that drops one closing quote almost always meant the
    following ``}``/``]`` to close the object, not to be swallowed into the string. The quote is
    inserted before that trailing run rather than appended at the absolute end, so
    ``'"query": "q}'`` repairs to ``'"query": "q"}'`` (closing brace preserved as structure), not
    to ``'"query": "q}"'`` (closing brace absorbed into the string value).
    """
    in_string = False
    escaped = False
    open_at: Optional[int] = None
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
            open_at = i
    if not in_string:
        return None
    trail_start = len(text)
    while trail_start > 0 and text[trail_start - 1] in "}] \t\r\n":
        trail_start -= 1
    if open_at is not None and trail_start <= open_at:
        trail_start = len(text)
    return text[:trail_start] + '"' + text[trail_start:]


def _fix_unbalanced_brackets(text: str) -> Optional[str]:
    """A completion truncated mid-object — append whatever closers are still owed, in the
    right order (string-aware, so a stray brace inside a string is never counted)."""
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    if in_string or not stack:
        return None
    return text + "".join(reversed(stack))


def _fix_bare_action_line(text: str) -> Optional[str]:
    """``action: search`` (or ``action:"search"``) with no braces at all -> wrap it as a decision
    object naming that action, so a model that dropped the JSON envelope entirely still gets
    through."""
    stripped = text.strip()
    match = re.match(r'^action\s*:\s*"?([A-Za-z_][\w\-]*)"?\s*$', stripped, re.IGNORECASE)
    if not match:
        return None
    return json.dumps({"action": match.group(1).lower()})


#: Applied in order; each fixer sees the previous fixer's output, so composite damage (e.g. a
#: trailing comma AND a truncated close) can be repaired in one pass. A fixer returns None when
#: it finds nothing to change, never when it "fixes" nothing.
_REPAIR_FIXERS: Tuple[Callable[[str], Optional[str]], ...] = (
    _fix_bare_action_line,
    _fix_trailing_comma,
    _fix_single_quotes,
    _fix_unterminated_string,
    _fix_unbalanced_brackets,
)


def repair_json_text(text: str) -> Optional[str]:
    """Deterministic, zero-cost repair of a near-miss JSON candidate.

    Tried BEFORE spending a whole extra model turn on a nudge — a malformed reply otherwise
    costs a full round trip for what is very often one dropped character. Idempotent on
    well-formed input: every fixer only fires on a textual defect it detects (an actual trailing
    comma, an actual unterminated string, ...), so a candidate that already parses cleanly never
    reaches this function from :func:`extract_decision` in the first place, and calling it
    directly on valid JSON returns None (nothing to change) rather than silently rewriting it —
    see ``prompted_tools_test.py``'s idempotency tests.

    :param text: A JSON-ish candidate that failed ``json.loads``.
    :returns: The repaired text, or None if no fixer found anything to change.
    """
    if not (text or "").strip():
        return None
    candidate = text
    changed = False
    for fixer in _REPAIR_FIXERS:
        fixed = fixer(candidate)
        if fixed is not None and fixed != candidate:
            candidate = fixed
            changed = True
    return candidate if changed else None


def extract_decision(raw: Optional[str]) -> JsonExtraction:
    """Recover a ``{"action": ...}`` object from a model completion, repairing near-misses.

    Deliberately tolerant: this exists for models with the least schema discipline, so a fenced
    block, leading prose, a single-element list wrapper, or a small deterministic defect (a
    trailing comma, single quotes, a truncated close, an unterminated string) all still parse.

    :param raw: A model completion, possibly wrapped in prose or a ```json fence.
    :returns: A :class:`JsonExtraction`. ``.value`` is None only when nothing — not even a
        repaired candidate — parses as a JSON object.
    :raises: Never.
    """
    text = (raw or "").strip()
    if not text:
        return JsonExtraction(None, "failed")

    fenced_match = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    was_fenced = fenced_match is not None
    if fenced_match:
        text = fenced_match.group(1).strip()
    whole_source = "fenced" if was_fenced else "direct"

    candidates = _json_candidates(text)
    whole = candidates[0]

    # The whole completion (or its fenced content) IS the intended decision, so it gets first
    # priority both to parse and to repair — before an accidentally-valid inner span (e.g. a
    # complete `args` sub-object left over after the outer object's closing punctuation was
    # damaged) can be mistaken for the decision itself.
    value = _parse_dict(whole)
    if value is not None:
        return JsonExtraction(value, whole_source)

    repaired_whole = repair_json_text(whole)
    if repaired_whole is not None:
        value = _parse_dict(repaired_whole)
        if value is not None:
            return JsonExtraction(value, whole_source, repaired=True, repair_attempts=1)

    for candidate in candidates[1:]:
        value = _parse_dict(candidate)
        if value is not None:
            return JsonExtraction(value, "span")

    attempts = 1 if repaired_whole is not None else 0
    for candidate in candidates[1:]:
        repaired_text = repair_json_text(candidate)
        if repaired_text is None:
            continue
        attempts += 1
        value = _parse_dict(repaired_text)
        if value is not None:
            return JsonExtraction(value, "span", repaired=True, repair_attempts=attempts)

    return JsonExtraction(None, "failed", repair_attempts=attempts)


# --------------------------------------------------------------------------- the loop


def _record_tool_turn_timing(
    telemetry: Any, started_at: float, *, action: str, extraction: JsonExtraction,
    invalid: bool, success: bool,
) -> None:
    """One ``telemetry.record_timing(...)`` per tool turn, following this repo's existing
    convention (see e.g. ``agent_io.py``'s ``llm_call`` timings) — a new payload shape, not a
    parallel telemetry mechanism. ``telemetry`` is duck-typed (only ``.record_timing`` is used)
    so this module never needs to import :mod:`agent.app.telemetry`.

    :raises: Never — telemetry must not fail the run it is observing.
    """
    if telemetry is None:
        return
    try:
        telemetry.record_timing(
            name="tool_call_emulation",
            started_at=started_at,
            success=success,
            payload={
                "transport": "prompted",
                "action": action,
                "json_source": extraction.source,
                "repaired": extraction.repaired,
                "repair_attempts": extraction.repair_attempts,
                "invalid_action": invalid,
            },
        )
    except Exception as exc:  # noqa: BLE001 — telemetry must never fail a run
        _logger.warning("[TOOL-EMULATION] telemetry.record_timing failed: %s", exc)


async def run_tool_loop(
    *,
    tools: Sequence[ToolSpec],
    render_view: RenderView,
    call_model: CallModel,
    dispatch_tool: DispatchTool,
    on_step: OnStep,
    turns: int,
    max_malformed_turns: int = MAX_MALFORMED_TURNS_DEFAULT,
    thought_chars: int = THOUGHT_CHARS_DEFAULT,
    telemetry: Any = None,
) -> None:
    """Run the think -> act -> observe loop until ``finish``, a malformed give-up, or the turn
    budget runs out.

    Message-free: the caller supplies ``render_view`` (what to show the model this turn, as
    plain text — the caller owns and renders its own transcript), ``call_model`` (send that text
    plus this loop's fixed system/protocol prompt, get back ``(raw_text, usage)``),
    ``dispatch_tool`` (run one named tool, get back its observation text), and ``on_step`` (told
    about every step's outcome so it can append to its own transcript and, e.g., log). This loop
    holds no transcript of its own — ``render_view`` reads the caller's, which ``on_step`` grows.

    :param tools: The run's tool specs — used only to validate an action name; ``"finish"`` is
        checked first and unconditionally recognized whether or not it appears here.
    :param render_view: Returns the current transcript rendered as the text the model should see
        next (before this loop's own :data:`STEP_SUFFIX` is appended).
    :param call_model: ``(user_prompt) -> (raw_completion_text, usage)``. The system/protocol
        prompt is the caller's concern (fixed for the whole loop), not this function's.
    :param dispatch_tool: ``(action_name, args) -> observation_text``. Called only for a
        recognized non-``finish`` action; never raises to this loop's caller (a tool failure is
        the dispatcher's own concern to turn into an observation string).
    :param on_step: Called once per step with a :class:`ToolLoopStep`; must apply the step to the
        caller's transcript (this loop calls ``render_view`` again next turn expecting that).
    :param turns: Maximum number of model turns.
    :param max_malformed_turns: Consecutive unparseable turns tolerated before giving up.
    :param thought_chars: How much of a parsed ``thought`` field survives into the step.
    :param telemetry: Optional duck-typed object exposing ``record_timing`` (see
        :func:`_record_tool_turn_timing`); None records nothing.
    :raises ToolLoopExhausted: When ``turns`` is spent without a ``finish`` action or a
        malformed-output give-up. The caller translates this into whatever step-budget exception
        its own runtime expects.
    """
    tool_names = {t.name for t in tools}
    malformed = 0

    for step in range(turns):
        started_at = time.perf_counter()
        prompt_text = f"{render_view()}\n\n{STEP_SUFFIX}"
        raw_text, usage = await call_model(prompt_text)
        extraction = extract_decision(raw_text)
        call_id = f"emu_{step}"

        if extraction.value is None:
            malformed += 1
            give_up = malformed >= max_malformed_turns
            on_step(ToolLoopStep(
                kind="malformed_give_up" if give_up else "malformed_nudge",
                raw_text=raw_text, usage=usage, extraction=extraction, call_id=call_id,
                malformed_count=malformed,
            ))
            _record_tool_turn_timing(
                telemetry, started_at, action="", extraction=extraction,
                invalid=False, success=False,
            )
            if give_up:
                _logger.warning(
                    "[TOOL-EMULATION] %d consecutive unparseable turns; accepting the model's "
                    "prose instead of burning the remaining budget on nudges.", malformed,
                )
                return
            continue

        malformed = 0
        decision = extraction.value
        action = str(decision.get("action", "")).strip().lower()
        args = decision.get("args")
        if not isinstance(args, dict):
            args = {}
        thought = str(decision.get("thought", ""))[:thought_chars]

        if action == "finish":
            answer = str(args.get("answer", "") or "")
            call = ToolCall(name="finish", args={"answer": answer}, thought=thought)
            on_step(ToolLoopStep(
                kind="finish", raw_text=raw_text, usage=usage, thought=thought,
                call=call, extraction=extraction, call_id=call_id,
            ))
            _record_tool_turn_timing(
                telemetry, started_at, action="finish", extraction=extraction,
                invalid=False, success=True,
            )
            return

        if action not in tool_names:
            call = ToolCall(name=action, args=args, thought=thought)
            on_step(ToolLoopStep(
                kind="invalid_action", raw_text=raw_text, usage=usage, thought=thought,
                call=call, extraction=extraction, call_id=call_id,
            ))
            _record_tool_turn_timing(
                telemetry, started_at, action=action, extraction=extraction,
                invalid=True, success=False,
            )
            continue

        observation = await dispatch_tool(action, args)
        call = ToolCall(name=action, args=args, thought=thought)
        on_step(ToolLoopStep(
            kind="tool_call", raw_text=raw_text, usage=usage, thought=thought,
            call=call, observation=observation, extraction=extraction, call_id=call_id,
        ))
        _record_tool_turn_timing(
            telemetry, started_at, action=action, extraction=extraction,
            invalid=False, success=True,
        )

    raise ToolLoopExhausted(f"tool loop exhausted after {turns} turn(s) without a finish action")
