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

    :ivar kind: One of ``"tool_call"``, ``"finish"``, ``"invalid_action"`` (unrecognized tool
        name, budget remains), ``"invalid_action_give_up"`` (unrecognized tool name, budget
        exhausted — mirrors ``"malformed_give_up"`` for the invalid-action counter),
        ``"malformed_nudge"`` (unparseable output, budget remains — a nudge should be appended),
        or ``"malformed_give_up"`` (unparseable output, budget exhausted — the loop is returning
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
    :ivar invalid_count: Consecutive unrecognized-action turns so far, including this one (only
        meaningful when ``kind`` starts with ``"invalid_action"``).
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
    invalid_count: int = 0


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


_WRAPPER_TAG_RE = re.compile(r"</?tool_call>", re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^<>|]*\|>")


def _strip_wrapper_tags(text: str) -> str:
    """``<tool_call>``/``</tool_call>`` and chat-template special tokens (``<|start_header_id|>``
    and friends) leaking into a completion — stripped, but ONLY outside a quoted JSON string.

    String-aware for the same reason as :func:`_fix_trailing_comma`: a legitimate argument value
    can genuinely contain the substring ``<tool_call>`` as content (e.g. a query ABOUT chat
    templates), and stripping it there would silently corrupt an otherwise well-formed decision
    instead of merely failing to help one. Every real occurrence in the fault corpus sits outside
    any quoted string (the tags wrap or precede the JSON, they never appear inside a value), so
    this loses no recovery.
    """
    out: List[str] = []
    i = 0
    n = len(text)
    in_string = False
    escaped = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        match = _WRAPPER_TAG_RE.match(text, i) or _SPECIAL_TOKEN_RE.match(text, i)
        if match:
            i = match.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_fenced_block(text: str) -> Tuple[str, bool]:
    """The content of a ```/```json fence, tolerating a fence that is never closed (the model ran
    out of tokens mid-block). Returns ``(content_or_original_text, was_fenced)``.

    A closed fence behaves exactly as the previous regex-based extraction did (first ``` to the
    next ```, optional ``json`` language tag skipped). An unterminated fence — opened but never
    closed — takes everything after the opening marker instead of falling through to the
    generic balanced-span search, which would have found the JSON but mislabeled the source
    ``"span"`` instead of ``"fenced"``.
    """
    open_idx = text.find("```")
    if open_idx == -1:
        return text, False
    after_open = open_idx + 3
    lang_match = re.match(r"[A-Za-z]*\s*", text[after_open:])
    content_start = after_open + lang_match.end()
    close_idx = text.find("```", content_start)
    if close_idx != -1:
        return text[content_start:close_idx].strip(), True
    # Unterminated fence: only trust it as THE decision when nothing but whitespace precedes the
    # opening marker. Discarding a prefix that has real content in it (prose, an earlier `k={}`
    # span, ...) could throw away a better candidate; a stray ``` deep in a long completion with
    # no closer is more likely noise than a truncated real answer.
    if text[:open_idx].strip():
        return text, False
    return text[content_start:].strip(), True


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
    """``{"a": 1,}`` / ``[1, 2,]`` -> drop the comma before the closer.

    String-aware: a comma only counts as "trailing" when it sits outside any quoted string, so a
    literal ``", }"`` inside an argument value (e.g. a quoted query ending in a comma) is never
    mistaken for a dropped-comma defect and rewritten.
    """
    out: List[str] = []
    changed = False
    in_string = False
    escaped = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                changed = True
                i += 1
                continue  # drop the comma; whitespace + closer fall through untouched
        out.append(ch)
        i += 1
    fixed = "".join(out)
    return fixed if changed else None


_CURLY_QUOTE_TRANSLATION = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _fix_curly_quotes(text: str) -> Optional[str]:
    """A completion that used curly/smart quotes throughout instead of straight ones — same
    guard as :func:`_fix_single_quotes`: only fires when there is no ASCII ``"`` anywhere in the
    text, so a completion that is properly ASCII-quoted but merely CONTAINS a curly quote as
    ordinary punctuation inside a string value (e.g. a quoted sentence using “smart quotes”) is
    left untouched — translating those unconditionally would splice a legitimate string open,
    turning a recoverable unterminated-string defect into unrecoverable garbage.
    """
    if '"' in text or not any(c in text for c in "“”‘’"):
        return None
    return text.translate(_CURLY_QUOTE_TRANSLATION)


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


_PYTHON_LITERALS = {"True": "true", "False": "false", "None": "null"}


def _fix_python_literals(text: str) -> Optional[str]:
    """Bare Python ``True``/``False``/``None`` tokens (a model that thinks in Python syntax
    instead of JSON) -> their JSON equivalents ``true``/``false``/``null``.

    String- and word-boundary-aware: a token is only replaced outside a quoted string and only
    when it is not part of a longer identifier, so a query VALUE like ``"True Grit movie"`` is
    left untouched (there is no bare ``True`` token there — it's string content, not JSON
    syntax), matching the same discipline as :func:`_fix_trailing_comma`.
    """
    out: List[str] = []
    changed = False
    in_string = False
    escaped = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        matched = None
        for word in ("True", "False", "None"):
            if text.startswith(word, i):
                before_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
                after = i + len(word)
                after_ok = after >= n or not (text[after].isalnum() or text[after] == "_")
                if before_ok and after_ok:
                    matched = word
                    break
        if matched:
            out.append(_PYTHON_LITERALS[matched])
            changed = True
            i += len(matched)
            continue
        out.append(ch)
        i += 1
    fixed = "".join(out)
    return fixed if changed else None


_KV_LINE_HEAD_RE = re.compile(
    r'^(?:ACTION|action|ACT|act|TOOL|tool)\s*[:=]\s*"?([A-Za-z_][\w\-]*)"?', re.IGNORECASE,
)
_KV_LINE_ARGS_RE = re.compile(r"^args\s*[:=]\s*(\{.*)", re.IGNORECASE)
_KV_LINE_SLOT_RE = re.compile(r"^([A-Za-z_][\w\-]*)\s*[:=]\s*(.+)$")


def _fix_kv_line_form(text: str) -> Optional[str]:
    """The kv-line action form ported from ``badmodel-lab/localagent/ir.py::parse_block``'s
    tolerant line-based fallback grammar: the first meaningful line names the action (a bare
    identifier, optionally prefixed with ``action:``/``action=``/``ACTION:``/``tool=``/...), and
    every following ``key: value`` or ``key=value`` line fills a slot — either ``args={...}`` (a
    JSON object taken verbatim) or a bare ``key=value`` pair collected into ``args``.

    Only fires when the FIRST line actually names an action this way; a text that starts with
    ``{`` (already JSON, however broken) never matches, so this cannot compete with or corrupt
    the brace-based repair path for near-miss JSON.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    # A model's malformed reply often carries a preamble before the action line ("STEP 9:",
    # restated context, ...) — scan for the FIRST line that names an action, rather than
    # requiring it to be lines[0], so that preamble doesn't sink the whole repair.
    head_idx = None
    match = None
    for idx, line in enumerate(lines):
        candidate_match = _KV_LINE_HEAD_RE.match(line)
        if candidate_match:
            head_idx = idx
            match = candidate_match
            break
    if match is None:
        return None
    head = lines[head_idx]
    action = match.group(1).lower()
    rest_of_head = head[match.end():].strip()
    # Stop at the next action-naming line (a later step in the same completion) so its slots
    # don't bleed into THIS action's args.
    end_idx = len(lines)
    for idx in range(head_idx + 1, len(lines)):
        if _KV_LINE_HEAD_RE.match(lines[idx]):
            end_idx = idx
            break
    remaining = ([rest_of_head] if rest_of_head else []) + lines[head_idx + 1:end_idx]

    args_obj: Dict[str, Any] = {}
    found_anything = False
    for line in remaining:
        args_match = _KV_LINE_ARGS_RE.match(line)
        if args_match:
            span = _balanced_span(args_match.group(1), 0)
            candidate = span if span is not None else args_match.group(1)
            parsed = _parse_dict(candidate)
            if parsed is not None:
                args_obj.update(parsed)
                found_anything = True
            continue
        slot_match = _KV_LINE_SLOT_RE.match(line)
        if not slot_match:
            continue
        key = slot_match.group(1).strip().lower()
        if key in ("action", "thought"):
            continue
        value = slot_match.group(2).strip()
        if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        args_obj[key] = value
        found_anything = True

    if not found_anything and len(remaining) > 0:
        # every remaining line failed to parse as a slot — too little signal to trust this shape
        return None

    decision: Dict[str, Any] = {"action": action}
    if args_obj:
        decision["args"] = args_obj
    return json.dumps(decision)


#: Applied in order; each fixer sees the previous fixer's output, so composite damage (e.g. a
#: trailing comma AND a truncated close) can be repaired in one pass. A fixer returns None when
#: it finds nothing to change, never when it "fixes" nothing.
_REPAIR_FIXERS: Tuple[Callable[[str], Optional[str]], ...] = (
    _fix_bare_action_line,
    _fix_kv_line_form,
    _fix_python_literals,
    _fix_trailing_comma,
    _fix_curly_quotes,
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

    text = _strip_wrapper_tags(text).strip()

    text, was_fenced = _extract_fenced_block(text)
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



#: Slot names a model plausibly puts its final answer in, beyond the documented ``answer``.
#: Measured on phi3:mini, which emits 98% valid JSON and finishes with ``{"answer1": ...,
#: "answer2": ...}`` on a two-part question -- the loop read only ``answer`` and silently dropped
#: the submission, so a model that had done the work scored zero for naming a slot.
_ANSWER_KEY_RE = re.compile(r"^(?:final_?)?(?:answer|response|result|output|text)_?\d*$", re.I)


def _finish_answer(args: Dict[str, Any]) -> str:
    """The answer a ``finish`` call is submitting, however the model named its slots.

    An explicit ``answer`` always wins outright and is never diluted by joining it with other
    keys -- that is the model's own choice of slot. Only when there is no ``answer`` at all are
    answer-shaped slots gathered, in SORTED key order so a two-part answer (``answer1``,
    ``answer2``) reassembles in the order the model numbered it rather than in dict order.

    Nothing is invented: a ``finish`` carrying no answer-shaped slot submits an empty answer
    rather than scraping an unrelated field (``confidence``, ``sources``) into one. Being lenient
    about a slot NAME is not the same as guessing at content.

    :param args: the parsed ``args`` object of a finish decision.
    :returns: the answer text, possibly empty.
    """
    direct = args.get("answer")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts = [str(args[key]) for key in sorted(args)
             if _ANSWER_KEY_RE.match(str(key)) and str(args[key] or "").strip()]
    return "\n".join(parts) if parts else str(direct or "")


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


def _record_json_telemetry(hook: Any, raw_text: str, parsed_ok: bool) -> None:
    """Best-effort call into a caller-supplied json-telemetry hook — see ``json_telemetry_hook``
    on :func:`run_tool_loop`. Never raises to the loop it observes.
    """
    if hook is None:
        return
    try:
        hook(raw_text, parsed_ok)
    except Exception as exc:  # noqa: BLE001 — telemetry must never fail a run
        _logger.warning("[TOOL-EMULATION] json_telemetry_hook failed: %s", exc)


def _split_inline_action_argument(
    action: str, args: Dict[str, Any], tools_by_name: Dict[str, "ToolSpec"],
) -> Tuple[str, Dict[str, Any]]:
    """``{"action": "visit https://en.wikipedia.org/wiki/X"}`` -> ``action="visit"``,
    ``args={"url": "https://en.wikipedia.org/wiki/X"}``.

    Conservative by construction: only fires when (1) the action string splits into a known tool
    name plus a non-empty remainder, (2) that tool has EXACTLY one arg slot (so there is no
    ambiguity about which slot the remainder belongs to), and (3) that slot isn't already
    explicitly filled by ``args`` (an explicit value always wins over a guessed one).

    ``action`` should be the RAW (not lower-cased) action string — the remainder is a URL or
    query and must keep its original case; only the leading tool-name token is matched
    case-insensitively.
    """
    parts = action.split(None, 1)
    if len(parts) != 2:
        return action.lower(), args
    name, rest = parts[0].lower(), parts[1].strip()
    if not rest:
        return action.lower(), args
    spec = tools_by_name.get(name)
    if spec is None or len(spec.arg_names) != 1:
        return action.lower(), args
    slot = spec.arg_names[0]
    new_args = dict(args)
    if not args.get(slot):
        new_args[slot] = rest  # an explicit value already present always wins over a guess
    return name, new_args


def _fuzzy_match_action(action: str, tool_names: Sequence[str]) -> Optional[str]:
    """A conservative fuzzy match of an unrecognized action name against the bound tool names —
    tried before counting a turn as a genuine invalid action.

    Handles: a trailing plural ``s`` (``searches`` -> ``search``), and an underscore-joined alias
    whose first or last token names a real tool (``search_web`` / ``web_search`` -> ``search``).
    Returns None — never a guess — when more than one tool name would match equally well, or when
    nothing matches at all.
    """
    if not action:
        return None
    names = set(tool_names)
    if action in names:
        return action
    candidates = set()
    if action.endswith("es") and action[:-2] in names:
        candidates.add(action[:-2])  # "searches" -> "search"
    if action.endswith("s") and action[:-1] in names:
        candidates.add(action[:-1])  # "visits" -> "visit"
    if (action + "s") in names:
        candidates.add(action + "s")
    if "_" in action:
        tokens = action.split("_")
        if tokens[0] in names:
            candidates.add(tokens[0])
        if tokens[-1] in names:
            candidates.add(tokens[-1])
    if len(candidates) == 1:
        return next(iter(candidates))
    return None


#: Consecutive turns naming an unrecognized action (after the inline-argument split and fuzzy
#: match both fail) tolerated before the loop gives up — same shape as
#: :data:`MAX_MALFORMED_TURNS_DEFAULT`, so a model stuck repeating a wrong tool name cannot burn
#: the entire step budget either.
MAX_INVALID_ACTIONS_DEFAULT = 3


async def run_tool_loop(
    *,
    tools: Sequence[ToolSpec],
    render_view: RenderView,
    call_model: CallModel,
    dispatch_tool: DispatchTool,
    on_step: OnStep,
    turns: int,
    max_malformed_turns: int = MAX_MALFORMED_TURNS_DEFAULT,
    max_invalid_actions: int = MAX_INVALID_ACTIONS_DEFAULT,
    thought_chars: int = THOUGHT_CHARS_DEFAULT,
    telemetry: Any = None,
    json_telemetry_hook: Optional[Callable[[str, bool], None]] = None,
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
    :param max_invalid_actions: Consecutive unrecognized-tool-name turns tolerated (after the
        inline-argument split and fuzzy match both fail to resolve the name) before giving up —
        the same shape as ``max_malformed_turns``, so a model stuck repeating a wrong tool name
        cannot burn the entire step budget either.
    :param thought_chars: How much of a parsed ``thought`` field survives into the step.
    :param telemetry: Optional duck-typed object exposing ``record_timing`` (see
        :func:`_record_tool_turn_timing`); None records nothing.
    :param json_telemetry_hook: Optional ``(raw_text, parsed_ok) -> None`` called once per turn.
        Deliberately duck-typed rather than a hard import of
        :mod:`agent.app.testing.json_telemetry` — this module stays framework/testing-infra-free;
        a caller that wants the fault corpus populated binds the real ``record`` function's other
        arguments (model name, ``phase="prompted_tool_loop"``) itself and passes the two-argument
        closure here. None records nothing (matches ``telemetry``'s own default).
    :raises ToolLoopExhausted: When ``turns`` is spent without a ``finish`` action or a
        malformed-output give-up. The caller translates this into whatever step-budget exception
        its own runtime expects.
    """
    tool_names = {t.name for t in tools}
    tools_by_name = {t.name: t for t in tools}
    malformed = 0
    invalid_streak = 0

    for step in range(turns):
        started_at = time.perf_counter()
        prompt_text = f"{render_view()}\n\n{STEP_SUFFIX}"
        raw_text, usage = await call_model(prompt_text)
        extraction = extract_decision(raw_text)
        call_id = f"emu_{step}"
        _record_json_telemetry(json_telemetry_hook, raw_text, extraction.value is not None)

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
        raw_action = str(decision.get("action", "")).strip()
        action = raw_action.lower()
        args = decision.get("args")
        if not isinstance(args, dict):
            args = {}
        thought = str(decision.get("thought", ""))[:thought_chars]

        if action == "finish":
            answer = _finish_answer(args)
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
            action, args = _split_inline_action_argument(raw_action, args, tools_by_name)

        if action not in tool_names:
            match = _fuzzy_match_action(action, tool_names)
            if match is not None:
                action = match

        if action not in tool_names:
            invalid_streak += 1
            give_up = invalid_streak >= max_invalid_actions
            call = ToolCall(name=action, args=args, thought=thought)
            on_step(ToolLoopStep(
                kind="invalid_action_give_up" if give_up else "invalid_action",
                raw_text=raw_text, usage=usage, thought=thought,
                call=call, extraction=extraction, call_id=call_id,
                invalid_count=invalid_streak,
            ))
            _record_tool_turn_timing(
                telemetry, started_at, action=action, extraction=extraction,
                invalid=True, success=False,
            )
            if give_up:
                _logger.warning(
                    "[TOOL-EMULATION] %d consecutive unrecognized-action turns; giving up "
                    "instead of burning the remaining budget on a wrong tool name.",
                    invalid_streak,
                )
                return
            continue

        invalid_streak = 0
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
