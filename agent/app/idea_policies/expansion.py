from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode

from agent.app.agent_io import AgentIO
from agent.app.idea_policies.base import ExpansionPolicy, DetailKey, IdeaActionType
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.shape_classifier import classify_shape
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.llm_backends import json_instruction_from_response_format
from agent.app.testing import json_telemetry as _json_telemetry


def _safe_serialize_details(details: Dict[str, Any]) -> str:
    try:
        return json.dumps(details, ensure_ascii=True, default=str)
    except Exception as e:
        return json.dumps({"error": f"Serialization failed: {str(e)}"}, ensure_ascii=True)


# --- malformed-expansion-JSON repair ---------------------------------------------------------
# A cheap executor model that emits prose-wrapped, fence-wrapped or TRUNCATED JSON used to fall
# through a single non-nesting regex (``\{[^{}]*"candidates"[^{}]*\}``) that cannot match once the
# candidates array contains objects — i.e. always. The plan silently became EMPTY (no children ->
# a tool-free run). These helpers salvage the object instead: brace-balanced extraction first,
# then a bounded "close the open brackets at the last complete element" repair for truncation.
# Every path fails safe (``None`` -> the caller returns an empty plan), never raises.
_JSON_REPAIR_MAX_STARTS = 32     # candidate '{' offsets tried for a complete object
_JSON_REPAIR_MAX_CUTS = 64       # truncation cut points tried, longest first
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _loads_lenient(text: str) -> Optional[Any]:
    """``json.loads`` with the two forgiving retries an LLM payload usually needs.

    ``strict=False`` tolerates raw control characters inside strings (a model pasting a page
    excerpt with literal newlines); the trailing-comma strip covers the other common slip.
    """
    for attempt in (text, _TRAILING_COMMA_RE.sub(r"\1", text)):
        try:
            return json.loads(attempt, strict=False)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None


def _scan_json(text: str, start: int) -> tuple[Optional[int], List[str], List[int]]:
    """String-aware bracket scan of the JSON value beginning at ``text[start]``.

    :returns: ``(end, stack, cuts)`` where ``end`` is the index AFTER the matching closing
        bracket (``None`` if the value never closes, i.e. truncated), ``stack`` is the still-open
        bracket list at the end of the text, and ``cuts`` are offsets just past a COMPLETE nested
        value — the only places a truncated fragment may safely be cut before auto-closing.
    """
    stack: List[str] = []
    cuts: List[int] = []
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
                if stack:
                    cuts.append(i + 1)
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None, stack, cuts  # unbalanced -> not a value we can trust
            stack.pop()
            if not stack:
                return i + 1, stack, cuts
            cuts.append(i + 1)
        elif ch in "0123456789truefalsn." and stack:
            # End of a bare literal (number / true / false / null) is a safe cut too.
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt in ("", ",", " ", "\t", "\r", "\n", "}", "]"):
                cuts.append(i + 1)
    return None, stack, cuts


def _autoclose(fragment: str) -> Optional[str]:
    """Close the brackets left open by a truncated ``fragment`` (``None`` if it can't be closed)."""
    end, stack, _ = _scan_json(fragment, 0)
    if end is not None or not stack:
        return None
    return fragment + "".join(reversed(stack))


def _repair_json_object(content: str, required_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Best-effort recovery of a JSON object from a malformed LLM response.

    Order: (1) every ``{`` offset is tried as the start of a brace-balanced object — this is what
    handles fences/prose around an otherwise VALID object with nested structures; (2) if nothing
    complete parses the payload is treated as TRUNCATED and repaired by cutting back to the last
    complete nested value and closing the open brackets; (3) a bare top-level array is accepted as
    the ``required_key`` list (a frequent cheap-model shape slip). Returns ``None`` when the text is
    genuinely unrepairable — the caller degrades to an empty plan rather than raising.
    """
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):  # drop a markdown fence so a bare array still starts at offset 0
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    starts = [i for i, ch in enumerate(text) if ch == "{"][:_JSON_REPAIR_MAX_STARTS]

    # An object that parses but lacks ``required_key`` is only a fallback, and only from the
    # OUTERMOST offset — a later ``{`` is an inner element (e.g. one candidate), never the plan.
    best: Optional[Dict[str, Any]] = None
    for start in starts:
        end, _stack, _cuts = _scan_json(text, start)
        if end is None:
            continue
        data = _loads_lenient(text[start:end])
        if not isinstance(data, dict):
            continue
        if required_key is None or required_key in data:
            return data
        if best is None and start == starts[0]:
            best = data

    for start in starts:
        end, _stack, cuts = _scan_json(text, start)
        if end is not None:
            continue  # already tried as a complete object above
        for cut in reversed(cuts[-_JSON_REPAIR_MAX_CUTS:]):
            closed = _autoclose(text[start:cut])
            if closed is None:
                continue
            data = _loads_lenient(closed)
            if isinstance(data, dict) and (required_key is None or required_key in data):
                return data

    # A bare top-level array of candidates ("[{...}, {...}]" with no wrapper object). Only when the
    # payload actually STARTS with the array — an inner array under some other key is that key's
    # data, not the plan, and re-labelling it would invent candidates.
    bracket = text.find("[")
    if required_key and bracket == 0:
        end, _stack, cuts = _scan_json(text, bracket)
        slices = [text[bracket:end]] if end is not None else [
            _autoclose(text[bracket:cut]) for cut in reversed(cuts[-_JSON_REPAIR_MAX_CUTS:])
        ]
        for candidate in slices:
            data = _loads_lenient(candidate) if candidate else None
            if isinstance(data, list) and data:
                return {required_key: data}
    return best


# Opt-in expansion addendum (``expansion_expect_contract_enabled``): borrows the compiled
# path's leaf discipline — one atomic fact per leaf, read off an authoritative page (never
# guessed from memory), reported as an EXACT value alongside its source URL. Injected into
# the expansion system prompt only when the flag is on (default path is byte-identical).
_EXPECT_CONTRACT_ADDENDUM = (
    "MEASURABLE OUTPUT CONTRACT: for every LEAF candidate (a concrete search/visit/think "
    "sub-task that resolves ONE fact), add an \"expect\" field at the candidate's top level: "
    "a single line naming the EXACT value to report AND requiring its source URL alongside "
    "it. Keep each leaf to ONE atomic fact read off an authoritative page — never guessed "
    "from memory. Example: \"expect\": \"the exact founding year (e.g. 1861) AND the source "
    "URL it was read from\". Omit \"expect\" for non-leaf or aggregation candidates."
)


# --- opt-in input/output framing (``expansion_input_output_framing_enabled``) -------------------
# The shipped expansion USER prompt opens with an OUTPUT instruction immediately followed by an
# INPUT blob:
#     Return your response as valid JSON. {"path": [...], "parent_id": ..., "event_log": [...]}
# Read literally, that sentence says "return this object" — and the expected output shape
# ({candidates: [...]}) is stated exactly once, far away, on the LAST line of a long system prompt.
#
# Live telemetry (2026-08-06, qwen2.5:0.5b, native `graph` variant, phase=native_expansion) shows a
# weak model doing what the text says: of 8 syntactically-valid completions, 5 echoed the "path"
# context straight back, 1 echoed the schema hint's own {"name": "expansion_result", "schema": ...}
# envelope complete with placeholder values, and only 2 produced a real {"candidates": [...]} plan.
# This is NOT a malformed-JSON problem (every one parsed) — it is the WRONG SHAPE, which nothing
# downstream can rescue: no ``candidates`` key means ``_create_fallback_candidate`` fires, and for
# the ROOT node (whose title IS the mandate) it emits a search whose query is the mandate's first
# 100 characters — an instruction preamble, not an entity — so the run makes zero page visits.
#
# The fix is pure prompt hygiene, in the spirit of the leaf-extraction source-ask removal: drop the
# misleading lead sentence, label the blob as read-only INPUT, and restate the OUTPUT shape
# immediately AFTER it — the recency position the model was already copying from.
_INPUT_FRAMING_HEADER = (
    "INPUT CONTEXT (read-only). The JSON below is this run's own history and state. It is DATA "
    "FOR YOU TO READ — not a template to fill in, not an example of your answer, and never "
    "something to copy back."
)
_INPUT_FRAMING_FOOTER = (
    "END OF INPUT CONTEXT.\n"
    "\n"
    "NOW WRITE A NEW JSON OBJECT OF YOUR OWN: the plan. It has a DIFFERENT shape from everything "
    "above, with exactly one required top-level key, \"candidates\":\n"
    "{\"candidates\": [{\"title\": \"<subproblem>\", \"action\": \"<one allowed action>\", "
    "\"details\": {<arguments for that action>}}], \"meta\": {\"execute_all_children\": false}}\n"
    "- Your reply must begin with {\"candidates\": and have no other top-level key except the "
    "optional \"meta\". A reply whose top-level key is \"path\", \"node_id\", \"name\" or "
    "\"schema\" is the WRONG object and will be discarded.\n"
    "- Every title, action and details value must be text YOU write for this plan, never copied "
    "from the input context above.\n"
    "- Output that one JSON object only: no markdown fences, no commentary."
)
#: The shipped user template's opening OUTPUT instruction. Removed (exact prefix match, no regex)
#: when the framing is on, so the framed message carries no leftover "return this" cue in front of
#: the input blob. A custom template that does not start with it is wrapped unchanged.
_USER_PROMPT_OUTPUT_LEAD = "Return your response as valid JSON. "


def frame_expansion_user_prompt(user: str) -> str:
    """Wrap a rendered expansion user message in explicit INPUT/OUTPUT framing.

    Wraps rather than replaces so a custom ``expansion_user_prompt`` (alternate settings files,
    sequential mode) keeps supplying the context payload and still gains the framing.
    """
    body = (user or "").lstrip()
    if body.startswith(_USER_PROMPT_OUTPUT_LEAD):
        body = body[len(_USER_PROMPT_OUTPUT_LEAD):]
    return f"{_INPUT_FRAMING_HEADER}\n{body}\n\n{_INPUT_FRAMING_FOOTER}"


# --- opt-in input-echo retry (``expansion_echo_retry_enabled``) ----------------------------------
# Safety net for whatever fraction of the failure above the prompt fix alone does not resolve, and
# deliberately narrow: it fires ONLY on the echo shape (parsed object, no usable ``candidates``,
# but one of the input/schema keys present), never on malformed JSON — ``_repair_json_object``
# already owns that class. Hard-bounded at ONE extra call per expansion, mirroring the compiled
# path's leaf-extract retry lever.
#: Keys that only ever appear in what the model was SHOWN — the user message's context blob
#: ("path"/"parent_id"/"parent_title"/"blocked_sites"/"errors"/"memories"/"event_log", plus the
#: per-entry "node_id"), or the schema hint's envelope ("name"/"schema").
_ECHO_INPUT_KEYS = (
    "path", "parent_id", "parent_title", "blocked_sites", "errors", "memories", "event_log",
    "node_id", "schema", "name",
)
_ECHO_RETRY_PROMPT = (
    "That reply was the WRONG OBJECT: its top-level key was \"{key}\", which came from the "
    "read-only context you were shown. That context is input; it is never the answer.\n"
    "Reply again with ONLY a new object of your own:\n"
    "{{\"candidates\": [{{\"title\": \"<subproblem>\", \"action\": \"<one allowed action>\", "
    "\"details\": {{<arguments for that action>}}}}]}}\n"
    "It must start with {{\"candidates\": and contain no \"path\", \"node_id\", \"name\" or "
    "\"schema\" key, no copied context, and no commentary."
)


def detect_input_echo(content: Optional[str]) -> Optional[str]:
    """Name the echoed key when a reply is syntactically valid JSON of the WRONG shape.

    Returns the offending top-level key for a parsed object that carries no usable ``candidates``
    but does carry one of the input/schema keys; ``None`` for anything else (unparseable, a real
    plan, or an empty/odd object that is some other failure). The key reported is the reply's OWN
    first offending key (JSON preserves document order), because the corrective prompt quotes it
    back at the model. Never raises.
    """
    if not content:
        return None
    try:
        data = json.loads(content)
    except Exception:  # noqa: BLE001 — malformed JSON is a different failure class
        return None
    if not isinstance(data, dict) or data.get("candidates"):
        return None
    for key in data:
        if key in _ECHO_INPUT_KEYS:
            return key
    return None


# --- extra-actions menu ------------------------------------------------------------------------
# The system prompt hand-writes ACTIONS entries for the core actions only, and its SEARCH + VISIT
# PIPELINE section actively steers every web task toward search/visit. So an action added through
# ``LeafActionRegistry.register()``/``install_pack()`` used to reach the model as a bare NAME in the
# ``{allowed_actions}`` sentence and nothing else: dispatchable (since the enum-coercion fix) but
# not practically selectable, because the prompt never said what it does, what it takes, or when it
# beats search+visit. This block closes that gap — it lists exactly the allowed NON-CORE actions,
# each with its own one-line description + argument shape, and reconciles them with the "every
# data-gathering task must include at least one visit" rule that would otherwise veto them.
#
# It is EMPTY whenever no non-core action is allowed (the default on every benchmark task and
# fixture), and an empty menu substitutes/appends nothing, so the prompt stays byte-identical.
_EXTRA_ACTIONS_MENU_HEADER = (
    "EXTRA ACTIONS (registered for this run, in addition to the four above):"
)
_EXTRA_ACTIONS_MENU_FOOTER = (
    "- Use one of these ONLY when it directly answers the subproblem; it is then cheaper and "
    "more exact than search+visit, and it counts as the data-gathering step for that child (no "
    "separate visit node is needed for it). For anything it does not cover, use search+visit "
    "exactly as described above."
)
#: Placeholder a prompt template may host so the menu lands in a chosen spot. A template that
#: does not host it gets the menu appended instead, so alternate/custom templates work too.
EXTRA_ACTIONS_MENU_PLACEHOLDER = "{extra_actions_menu}"


def build_extra_actions_menu(lines: List[str]) -> str:
    """Render the prompt block for ``lines`` (already-formatted per-action menu lines).

    Returns ``""`` for an empty list — the caller relies on that to keep the default prompt
    byte-identical.
    """
    if not lines:
        return ""
    return "\n".join([_EXTRA_ACTIONS_MENU_HEADER, *lines, _EXTRA_ACTIONS_MENU_FOOTER])


# --- core-actions menu (shared with non-GoT loops) ---------------------------------------------
# The CORE actions' model-facing prose is hand-written in exactly one place: the shipped
# ``expansion_system_prompt``'s ``ACTIONS:`` block. A loop that is not the Generate operation (the
# flat ``naive_discretion`` variant) still has to describe the same tools to the model, and a
# hand-copied second version of that prose would drift — so it reads the block back out of the
# settings instead. Holding the TOOL DOCUMENTATION fixed across arms is the point: an arm
# comparison is then attributable to structure, not to one arm's tools being better explained.
_CORE_ACTIONS_BLOCK_HEADER = "ACTIONS:"


def core_actions_menu_lines(
    settings: Optional[Dict[str, Any]] = None, allowed: Optional[List[Any]] = None,
) -> List[str]:
    """The ``ACTIONS:`` entries of the expansion system prompt, filtered to ``allowed``.

    One string per bullet, with its indented continuation lines (``* optional_url: ...``) kept
    attached, in the prompt's own order. The template's doubled braces (``details={{query}}``)
    are unescaped, since a caller outside ``str.format`` should show the model real braces.

    An allowed action the block does not document produces nothing — ``merge`` has no
    hand-written entry because it is engine-driven rather than model-selected — mirroring
    ``LeafActionRegistry.menu_lines``' rule that the menu never advertises what it cannot
    describe. ``allowed=None`` returns every documented entry.

    :param settings: Settings carrying ``expansion_system_prompt`` (falls back to the shipped JSON).
    :param allowed: This run's ``allowed_actions``; ``None`` means "no filtering".
    :returns: Prompt-ready menu lines (empty when the block is missing).
    """
    prompt = str((settings or {}).get("expansion_system_prompt") or "").strip()
    if not prompt:
        prompt = str(load_idea_dag_settings().get("expansion_system_prompt") or "")
    lines = prompt.splitlines()
    try:
        start = lines.index(_CORE_ACTIONS_BLOCK_HEADER)
    except ValueError:
        return []
    entries: List[List[str]] = []
    names: List[str] = []
    for raw in lines[start + 1:]:
        if not raw.strip():
            break
        if raw.startswith("- "):
            names.append(raw[2:].split(":", 1)[0].strip())
            entries.append([raw])
        elif entries:
            entries[-1].append(raw)
        else:
            break
    permitted = None if allowed is None else {str(item) for item in allowed}
    return [
        "\n".join(block).replace("{{", "{").replace("}}", "}")
        for name, block in zip(names, entries)
        if permitted is None or name in permitted
    ]


# IDEA_TEST_REASONING_EXEMPLAR: per-run toggle that injects a general reasoning
# demonstration (a task-shape few-shot) into the expansion system prompt, so a
# cheap executor model learns HOW to reason through a chain/mixed/parallel task
# without leaking any answer. Follows the IDEA_TEST_GOT_REEXPAND convention: unset
# or "none" means byte-identical prompt behavior. Exemplars are read from disk once
# and cached per name for the process lifetime.
#
# DISPROVEN, kept only for reference — do not reach for this as a default. Live R=3
# validation (ADAPTIVE_DISTILLATION_HANDOFF.md Phase 1) found narrative exemplars
# unreliable on weak models: they backfired twice on the branch_eliminate/mixed
# shape (the model copied the exemplar's surface structure, not its intent, and
# under-decomposed the task further) and a seeming win on the parallel shape
# dissolved to noise at R=3. Prefer `IDEA_TEST_REASONING_RULES` (a flat imperative
# checklist) instead, which this project's research found narrative few-shot
# reasoning demonstrations are unusually sensitive to being copied for shape, not
# intent (see RESEARCH_NOTES.md, "Why narrative few-shot exemplars backfired").
_EXEMPLAR_NAMES = ("chain", "mixed", "parallel")
_EXEMPLAR_DIR = Path(__file__).resolve().parent.parent / "reasoning_exemplars"
_EXEMPLAR_CACHE: Dict[str, str] = {}
_EXEMPLAR_WARNED: set = set()


def _load_reasoning_exemplar() -> str:
    """Return the labeled few-shot exemplar block for IDEA_TEST_REASONING_EXEMPLAR,
    or "" when unset/none/invalid. Reads and caches the .md content once per name."""
    name = os.environ.get("IDEA_TEST_REASONING_EXEMPLAR", "").strip().lower()
    if not name or name == "none":
        return ""
    if name not in _EXEMPLAR_NAMES:
        if name not in _EXEMPLAR_WARNED:
            _EXEMPLAR_WARNED.add(name)
            logging.getLogger("LlmExpansionPolicy").warning(
                "[EXPANSION] Ignoring invalid IDEA_TEST_REASONING_EXEMPLAR=%r "
                "(expected one of %s or 'none')",
                name,
                ", ".join(_EXEMPLAR_NAMES),
            )
        return ""
    if name not in _EXEMPLAR_WARNED:
        _EXEMPLAR_WARNED.add(name)
        logging.getLogger("LlmExpansionPolicy").warning(
            "[EXPANSION] IDEA_TEST_REASONING_EXEMPLAR=%r is a DISPROVEN mechanism "
            "(see ADAPTIVE_DISTILLATION_HANDOFF.md Phase 1) — it backfired on the "
            "hardest task shape in live R=3 validation. Kept for reference only; "
            "prefer IDEA_TEST_REASONING_RULES instead.",
            name,
        )
    if name in _EXEMPLAR_CACHE:
        return _EXEMPLAR_CACHE[name]
    try:
        text = (_EXEMPLAR_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    except OSError as exc:
        if name not in _EXEMPLAR_WARNED:
            _EXEMPLAR_WARNED.add(name)
            logging.getLogger("LlmExpansionPolicy").warning(
                "[EXPANSION] Could not read reasoning exemplar %r: %s", name, exc
            )
        _EXEMPLAR_CACHE[name] = ""
        return ""
    block = (
        "## Reference reasoning pattern (a general demonstration, not this task's answer)\n\n"
        f"{text}\n\n"
        "## Now apply that reasoning pattern to the current task below.\n"
    )
    _EXEMPLAR_CACHE[name] = block
    return block


# IDEA_TEST_REASONING_RULES: per-run toggle that injects a FLAT IMPERATIVE checklist
# (not a narrative) into the expansion system prompt, to enforce a discipline (e.g.
# "check ALL candidates before electing a survivor") that a weak executor tends to
# skip. Fully independent of IDEA_TEST_REASONING_EXEMPLAR; either can be set alone.
# Follows the same convention: unset/none/invalid means byte-identical prompt behavior.
# Rule files are read from disk once and cached per name for the process lifetime.
_RULES_NAMES = ("branch_eliminate",)
_RULES_DIR = Path(__file__).resolve().parent.parent / "reasoning_rules"
_RULES_CACHE: Dict[str, str] = {}
_RULES_WARNED: set = set()


def _read_rules_block(name: str) -> str:
    """Read, cache and wrap the rule-checklist .md for ``name`` (already validated as a
    member of ``_RULES_NAMES``). Returns "" and warns once if the file is unreadable."""
    if name in _RULES_CACHE:
        return _RULES_CACHE[name]
    try:
        text = (_RULES_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    except OSError as exc:
        if name not in _RULES_WARNED:
            _RULES_WARNED.add(name)
            logging.getLogger("LlmExpansionPolicy").warning(
                "[EXPANSION] Could not read reasoning rules %r: %s", name, exc
            )
        _RULES_CACHE[name] = ""
        return ""
    block = (
        "## Mandatory reasoning rules (follow every rule literally)\n\n"
        f"{text}\n"
    )
    _RULES_CACHE[name] = block
    return block


def _load_reasoning_rules() -> str:
    """Return the imperative rule-checklist block for IDEA_TEST_REASONING_RULES,
    or "" when unset/none/invalid. Reads and caches the .md content once per name."""
    name = os.environ.get("IDEA_TEST_REASONING_RULES", "").strip().lower()
    if not name or name == "none":
        return ""
    if name not in _RULES_NAMES:
        if name not in _RULES_WARNED:
            _RULES_WARNED.add(name)
            logging.getLogger("LlmExpansionPolicy").warning(
                "[EXPANSION] Ignoring invalid IDEA_TEST_REASONING_RULES=%r "
                "(expected one of %s or 'none')",
                name,
                ", ".join(_RULES_NAMES),
            )
        return ""
    return _read_rules_block(name)


def _auto_reasoning_rules(mandate: str) -> str:
    """Auto-select a rule-checklist block from the mandate's classified shape.

    Only invoked when IDEA_TEST_REASONING_RULES is UNSET (the manual override always
    wins). Uses the deterministic ``classify_shape``. Today only ``branch_eliminate``
    has a matching rule file, so a correctly-classified ``chain``/``parallel_merge``
    mandate intentionally yields NO block (documented gap — no placeholder files are
    fabricated). Fails open to "" for unclassified mandates."""
    shape = classify_shape(mandate or "")
    if not shape:
        return ""
    if shape not in _RULES_NAMES or not (_RULES_DIR / f"{shape}.md").exists():
        logging.getLogger("LlmExpansionPolicy").info(
            "[EXPANSION] Auto-classified mandate shape=%s but no reasoning rule file "
            "exists yet; skipping auto-injection.",
            shape,
        )
        return ""
    logging.getLogger("LlmExpansionPolicy").info(
        "[EXPANSION] Auto-selected reasoning rules for classified shape=%s", shape
    )
    return _read_rules_block(shape)


class LlmExpansionPolicy(ExpansionPolicy):
    """Generate operation: ask the model for this node's children.

    :param actions: The engine's live ``LeafActionRegistry``. Optional and read-only — it is
        used solely to DESCRIBE allowed non-core actions in the prompt (see
        ``build_extra_actions_menu``). Passing the engine's own instance (rather than a copy)
        means an action registered after construction still gets described. ``None`` (a
        standalone policy, e.g. in a test) simply renders no menu — and since only a NON-CORE
        action can ever produce a menu line, that is the same prompt a default registry gives.
    """

    def __init__(self, io: AgentIO, settings: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None,
                 actions: Optional[Any] = None):
        default_settings = load_idea_dag_settings()
        merged_settings = {**default_settings, **(settings or {})}
        super().__init__(settings=merged_settings)
        self._cfg = IdeaConfig.from_settings(merged_settings)
        self.io = io
        self.model_name = model_name
        self.actions = actions
        self._logger = logging.getLogger(self.__class__.__name__)

    def _extra_actions_menu(self, allowed: List[Any]) -> str:
        """The extra-actions prompt block for this run's ``allowed`` list (``""`` when none).

        Never raises: a registry that cannot describe itself must not sink an expansion.
        """
        registry = self.actions
        if registry is None:
            return ""
        try:
            return build_extra_actions_menu(registry.menu_lines(allowed))
        except Exception as exc:  # noqa: BLE001 — a menu is an enhancement, never a failure mode
            self._logger.warning(f"[EXPANSION] Could not build the extra-actions menu: {exc}")
            return ""

    async def expand(self, graph: IdeaDag, node_id: str, memories: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        node = graph.get_node(node_id)
        if not node:
            return []
        messages = self._build_messages(graph, node, memories=memories)

        # The expansion schema's candidate ``details`` is a free-form, per-action
        # object. Strict structured output (OpenAI/Azure) rejects any object lacking
        # ``additionalProperties: false``, which we cannot add without forbidding the
        # action keys (query/url/mandate/...). So convey the candidate shape as a
        # text instruction and drop to ``json_object`` mode (provider-agnostic — no
        # model-name special-casing).
        json_schema = self.settings.get("expansion_json_schema")
        # Opt-in: when the measurable-output contract is enabled, use the schema variant
        # that advertises the optional per-candidate ``expect`` field, so the text schema
        # hint tells the model it may declare a leaf's measurable target. Default path is
        # byte-identical (the plain schema from settings).
        if self._cfg.expansion.expect_contract_enabled:
            from agent.app.idea_dag_schemas import EXPANSION_JSON_SCHEMA_WITH_EXPECT
            json_schema = EXPANSION_JSON_SCHEMA_WITH_EXPECT
        # Opt-in, same lever as the user-prompt framing: ``json_instruction_from_response_format``
        # dumps the whole {"name": "expansion_result", "schema": {...}} envelope into the system
        # prompt, and a weak model copies that ENVELOPE instead of producing an instance of it
        # (one of the 8 recorded qwen2.5:0.5b completions did exactly that, placeholders and all).
        # Passing the inner schema keeps every constraint and removes the copyable wrapper.
        if self._cfg.expansion.input_output_framing_enabled and isinstance(json_schema, dict):
            inner_schema = json_schema.get("schema")
            if isinstance(inner_schema, dict):
                json_schema = inner_schema
        schema_hint = (
            json_instruction_from_response_format({"type": "json_schema", "json_schema": json_schema})
            if json_schema
            else None
        )
        if schema_hint:
            messages = self._inject_schema_hint(messages, schema_hint)

        total_prompt_size = sum(len(msg.get("content", "")) for msg in messages)
        if total_prompt_size > 50000:
            self._logger.warning(f"[EXPANSION] Large prompt detected ({total_prompt_size} chars) for node {node_id} - may cause slow expansion")

        model_name = self.model_name or self._cfg.expansion.model
        reasoning_effort = self._cfg.generation.reasoning_effort
        text_verbosity = self._cfg.generation.text_verbosity
        max_tokens = self._cfg.expansion.max_tokens

        payload = self.io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=model_name,
            temperature=self._cfg.expansion.temperature,
            max_tokens=max_tokens,
            json_schema=None,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )
        try:
            estimated_tokens = (total_prompt_size // 4) + (max_tokens or 4096)
            self._logger.debug(f"[EXPANSION] Calling LLM for node {node_id} with model={model_name}, prompt={total_prompt_size} chars, max_tokens={max_tokens}, estimated ~{estimated_tokens} total tokens")
            
            default_timeout = self._cfg.timeouts.llm or 120
            expansion_timeout = self._cfg.timeouts.expansion or default_timeout
            if total_prompt_size > 50000 or estimated_tokens > 10000:
                expansion_timeout = max(expansion_timeout, 180)
            else:
                expansion_timeout = max(expansion_timeout, 120)
            
            try:
                preview = json.dumps(messages, indent=2, ensure_ascii=True)
            except Exception:
                preview = str(messages)
            if len(preview) > 2000:
                preview = preview[:2000] + "... [truncated]"
            self._logger.debug(f"[EXPANSION] LLM Input preview: {preview}")
            content = await self.io.query_llm_with_fallback(
                payload,
                model_name=model_name,
                fallback_model=self._cfg.generation.fallback_model,
                timeout_seconds=expansion_timeout,
            )
            output_preview = content[:2000] + "... [truncated]" if isinstance(content, str) and len(content) > 2000 else content
            self._logger.debug(f"[EXPANSION] LLM Output preview: {output_preview}")
            # Bad-model lab (env-gated, no-op by default): classify WHY a weak model's plan
            # JSON failed. ``parsed_ok`` is the raw completion's JSON-syntax validity BEFORE
            # ``_parse_candidates``' repair fallback — same measure as the react leaf call
            # sites — so the whole block (including the throwaway parse) is skipped when the
            # flag is off.
            if _json_telemetry.enabled():
                try:
                    json.loads(content or "")
                    _parsed_ok = True
                except Exception:
                    _parsed_ok = False
                _json_telemetry.record(model_name, content, True, _parsed_ok, phase="native_expansion")
            candidates, meta = self._parse_candidates(content, graph=graph, parent_node_id=node_id)
            self._logger.info(f"[EXPANSION] Parsed {len(candidates)} candidates from LLM response, meta={meta}")
            # Opt-in safety net (``expansion_echo_retry_enabled``): the reply parsed but was the
            # INPUT echoed back, not a plan. ONE corrective retry that names the offending key,
            # then the existing fallback path takes over exactly as before. Self-contained failure
            # handling so a failed retry degrades to that fallback, never to an empty plan.
            if not candidates and self._cfg.expansion.echo_retry_enabled:
                echoed_key = detect_input_echo(content)
                if echoed_key:
                    self._logger.warning(
                        f"[EXPANSION] Reply echoed the input context (top-level key "
                        f"'{echoed_key}') instead of a plan - one corrective retry"
                    )
                    try:
                        retry_messages = self._append_user_directive(
                            messages, _ECHO_RETRY_PROMPT.format(key=echoed_key)
                        )
                        retry_payload = self.io.build_llm_payload(
                            messages=retry_messages,
                            json_mode=True,
                            model_name=model_name,
                            temperature=self._cfg.expansion.temperature,
                            max_tokens=max_tokens,
                            json_schema=None,
                            reasoning_effort=reasoning_effort,
                            text_verbosity=text_verbosity,
                        )
                        retry_content = await self.io.query_llm_with_fallback(
                            retry_payload,
                            model_name=model_name,
                            fallback_model=self._cfg.generation.fallback_model,
                            timeout_seconds=expansion_timeout,
                        )
                        if _json_telemetry.enabled():
                            try:
                                json.loads(retry_content or "")
                                _retry_parsed_ok = True
                            except Exception:
                                _retry_parsed_ok = False
                            _json_telemetry.record(
                                model_name, retry_content, True, _retry_parsed_ok,
                                phase="native_expansion_echo_retry",
                            )
                        candidates, meta = self._parse_candidates(
                            retry_content, graph=graph, parent_node_id=node_id
                        )
                        self._logger.info(
                            f"[EXPANSION] Echo retry parsed {len(candidates)} candidates"
                        )
                    except Exception as retry_err:  # noqa: BLE001 — a retry never sinks a run
                        self._logger.warning(f"[EXPANSION] Echo retry failed: {retry_err}")
            if not candidates:
                self._logger.error(f"[EXPANSION] CRITICAL: No candidates parsed from LLM response!")
                self._logger.error(f"[EXPANSION] LLM response length: {len(content) if content else 0} chars")
                self._logger.error(f"[EXPANSION] LLM response preview: {content[:500] if content else 'None'}")
                fallback_candidate = self._create_fallback_candidate(node, graph)
                if fallback_candidate:
                    self._logger.warning(f"[EXPANSION] Created fallback candidate: {fallback_candidate.get('title', 'Unknown')[:60]}...")
                    candidates = [fallback_candidate]
            if meta:
                node.details[DetailKey.EXPANSION_META.value] = meta
            return candidates
        except asyncio.TimeoutError as e:
            self._logger.error(f"[EXPANSION] Timeout during expansion for node {node_id}: {e}")
            self._logger.warning(f"[EXPANSION] Expansion timeout - returning empty candidates. Consider increasing expansion_timeout_seconds.")
            return []
        except Exception as e:
            self._logger.error(f"[EXPANSION] Exception during expansion: {e}", exc_info=True)
            return []

    def _append_user_directive(self, messages: List[Dict[str, str]], directive: str) -> List[Dict[str, str]]:
        """Append ``directive`` to the LAST user message (a copy; message count unchanged).

        The corrective retry rides on the existing user turn rather than adding a second one:
        the count stays system + user (as ``_inject_schema_hint`` also guarantees), which keeps
        every backend happy about role alternation, and the directive lands in the last position
        the model reads.
        """
        out = [dict(msg) for msg in messages]
        for msg in reversed(out):
            if msg.get("role") == "user":
                existing = msg.get("content") or ""
                msg["content"] = f"{existing}\n\n{directive}" if existing else directive
                return out
        out.append({"role": "user", "content": directive})
        return out

    def _inject_schema_hint(self, messages: List[Dict[str, str]], hint: str) -> List[Dict[str, str]]:
        """Append the candidate-shape instruction to the system message.

        The expansion schema cannot ride on the wire as strict structured output
        (free-form ``details``), so its shape is folded into the prompt instead.
        Operates on a copy and keeps the message count stable (system + user);
        if there is no system message, one is prepended.
        """
        out = [dict(msg) for msg in messages]
        for msg in out:
            if msg.get("role") == "system":
                existing = msg.get("content") or ""
                msg["content"] = f"{existing}\n\n{hint}" if existing else hint
                return out
        out.insert(0, {"role": "system", "content": hint})
        return out

    def _enhance_details_with_inline_links(self, details: Dict[str, Any]) -> Dict[str, Any]:
        from agent.app.idea_policies.action_constants import ActionResultKey
        enhanced = dict(details)
        
        action_result = details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(action_result, dict):
            return enhanced
        
        action = action_result.get(ActionResultKey.ACTION.value)
        if action != IdeaActionType.VISIT.value:
            return enhanced
        
        success = action_result.get(ActionResultKey.SUCCESS.value, False)
        if not success:
            return enhanced
        
        links = action_result.get("links") or action_result.get("links_full") or []
        if not isinstance(links, list) or len(links) == 0:
            return enhanced
        
        link_contexts = action_result.get(ActionResultKey.LINK_CONTEXTS.value) or {}
        max_links_to_show = self._cfg.action.max_links_per_visit
        links_to_show = links[:max_links_to_show]
        
        inline_links_section = []
        for link_url in links_to_show:
            if not isinstance(link_url, str) or not link_url.startswith(("http://", "https://")):
                continue
            
            context_text = ""
            if isinstance(link_contexts, dict) and link_url in link_contexts:
                context = link_contexts[link_url]
                if isinstance(context, str) and context.strip():
                    context_text = context.strip()[:150]
            
            if context_text:
                inline_links_section.append(f"{context_text} [link: {link_url}]")
            else:
                inline_links_section.append(f"[link: {link_url}]")
        
        if inline_links_section:
            enhanced_action_result = dict(action_result)
            enhanced_action_result["_links_inline"] = "\n".join(inline_links_section)
            if len(links) > max_links_to_show:
                enhanced_action_result["_links_inline"] += f"\n... and {len(links) - max_links_to_show} more links (see 'links' field for full list)"
            enhanced[DetailKey.ACTION_RESULT.value] = enhanced_action_result
        
        return enhanced
    
    def _compact_details_for_expansion(self, details: Dict[str, Any]) -> Dict[str, Any]:
        from agent.app.idea_policies.action_constants import ActionResultKey
        compact = dict(details)
        
        action_result = details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(action_result, dict):
            return compact
        
        compact_result = dict(action_result)
        
        large_fields_to_remove = [
            ActionResultKey.CONTENT_FULL.value,
            ActionResultKey.CONTENT_WITH_LINKS.value,
            "content_full",
            "content_with_links",
        ]
        
        for field in large_fields_to_remove:
            if field in compact_result:
                del compact_result[field]
        
        if ActionResultKey.CONTENT.value in compact_result:
            content = compact_result[ActionResultKey.CONTENT.value]
            if isinstance(content, str) and len(content) > 1000:
                compact_result[ActionResultKey.CONTENT.value] = content[:1000] + "... [truncated]"
        
        if "links_full" in compact_result:
            links_full = compact_result.get("links_full", [])
            if isinstance(links_full, list) and len(links_full) > 20:
                compact_result["links_full"] = links_full[:20]
                compact_result["_links_full_truncated"] = f"... and {len(links_full) - 20} more links"
        
        compact[DetailKey.ACTION_RESULT.value] = compact_result
        return compact
    
    def _extract_key_outcome(self, node: IdeaNode) -> Optional[str]:
        from agent.app.idea_policies.action_constants import ActionResultKey
        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            return None
        action = node.details.get(DetailKey.ACTION.value, "")
        if not result.get(ActionResultKey.SUCCESS.value, False):
            error = result.get(ActionResultKey.ERROR.value, "unknown error")
            return f"FAILED: {str(error)[:80]}"
        if action == IdeaActionType.SEARCH.value:
            results = result.get(ActionResultKey.RESULTS.value, [])
            count = len(results) if isinstance(results, list) else 0
            top_urls = []
            if isinstance(results, list):
                for r in results[:3]:
                    if isinstance(r, dict) and r.get("url"):
                        top_urls.append(str(r["url"])[:80])
            if top_urls:
                return f"Found {count} results. Top URLs: {', '.join(top_urls)}"
            return f"Found {count} results"
        if action == IdeaActionType.VISIT.value:
            url = result.get(ActionResultKey.URL.value, "")
            page_title = result.get("page_title", "")
            content_chars = result.get("content_total_chars", 0)
            links_count = result.get("links_count", 0)
            parts = []
            if url:
                parts.append(f"Visited {str(url)[:80]}")
            if page_title:
                parts.append(f"page='{str(page_title)[:50]}'")
            parts.append(f"{content_chars} chars, {links_count} links")
            return ". ".join(parts)
        if action == IdeaActionType.THINK.value:
            return "Internal reasoning completed"
        return None

    def _build_messages(self, graph: IdeaDag, node: IdeaNode, memories: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, str]]:
        max_nodes = self._cfg.expansion.max_context_nodes
        max_detail_chars = self._cfg.expansion.max_detail_chars
        max_children = self._cfg.engine.max_branching
        if max_children <= 1:
            max_children = 1
        path = graph.path_to_root(node.node_id)
        path = path[:max_nodes]
        
        serialized = []
        for entry in path:
            enhanced_details = self._enhance_details_with_inline_links(entry.details)
            compact_details = self._compact_details_for_expansion(enhanced_details)
            details_text = _safe_serialize_details(compact_details)
            if len(details_text) > max_detail_chars:
                details_text = details_text[:max_detail_chars]
            serialized.append(
                {
                    "node_id": entry.node_id,
                    "title": entry.title,
                    "status": entry.status.value,
                    "score": entry.score,
                    "action": entry.details.get(DetailKey.ACTION.value, "expansion"),
                    "goal": entry.details.get(DetailKey.GOAL.value, ""),
                    "justification": (
                        entry.details.get(DetailKey.JUSTIFICATION.value)
                        or entry.details.get(DetailKey.WHY_THIS_NODE.value)
                        or ""
                    ),
                    "key_outcome": self._extract_key_outcome(entry),
                    "details": details_text,
                }
            )
        allowed = self.settings.get("allowed_actions") or [a.value for a in IdeaActionType]
        allowed_actions = ", ".join(
            str(item) for item in allowed
            if str(item) != IdeaActionType.MERGE.value
        )
        path_json = json.dumps(serialized, ensure_ascii=True)
        
        blocked_sites = graph._blocked_sites if hasattr(graph, "_blocked_sites") else {}
        blocked_sites_list = [f"{domain}: {reason}" for domain, reason in blocked_sites.items()]
        blocked_sites_text = "\n".join(blocked_sites_list) if blocked_sites_list else "None"
        
        errors = []
        for entry in path:
            error = entry.details.get(DetailKey.ACTION_ERROR.value)
            if error:
                errors.append(f"{entry.title}: {error}")
        errors_text = "\n".join(errors) if errors else "None"
        
        memories_text = "None"
        if memories:
            from agent.app.idea_memory import MemoryManager
            temp_mm = MemoryManager(connector_chroma=None, namespace="temp")
            memories_text = temp_mm.format_memories_for_llm(memories, max_chars=4000)
        
        event_log = graph.build_event_log_table(node.node_id, max_events=15)
        event_log_json = json.dumps(event_log) if event_log else json.dumps("No events")
        
        system_template = self.settings.get("expansion_system_prompt")
        user_template = self.settings.get("expansion_user_prompt")
        effective_range = f"exactly {max_children}" if max_children <= 1 else f"2-{max_children}"
        # Non-core actions this run allows, described for the model. Empty (and therefore a
        # no-op on the prompt bytes) unless a caller registered AND allowed an extra action.
        extra_menu = self._extra_actions_menu(allowed)
        hosts_placeholder = EXTRA_ACTIONS_MENU_PLACEHOLDER in (system_template or "")
        # A hosted placeholder occupies the blank line where the block belongs, so the empty
        # menu collapses back to that blank line and the filled one keeps its blank lines.
        menu_substitution = f"\n{extra_menu}\n" if extra_menu else ""
        try:
            system = system_template.format(
                allowed_actions=allowed_actions,
                max_children=effective_range,
                extra_actions_menu=menu_substitution,
            ) if system_template else ""
        except KeyError as fmt_err:
            self._logger.error(f"[EXPANSION] System prompt format error (missing key: {fmt_err}) - using raw template")
            system = (
                (system_template or "")
                .replace("{allowed_actions}", str(allowed_actions))
                .replace("{max_children}", str(effective_range))
                .replace(EXTRA_ACTIONS_MENU_PLACEHOLDER, menu_substitution)
            )
        # Templates that do not host the placeholder (alternate settings files, custom prompts)
        # still get the menu — appended, in the same way the planning addendum below is.
        if extra_menu and not hosts_placeholder:
            system = f"{system}\n\n{extra_menu}" if system else extra_menu
        planning_addendum = str(
            self.settings.get(
                "expansion_planning_addendum",
                "Before producing candidates, build an internal plan with target facts, source strategy, and verification steps.",
            )
        ).strip()
        if planning_addendum:
            system = f"{system}\n\n{planning_addendum}" if system else planning_addendum
        # Opt-in measurable-output contract: append the leaf ``expect`` discipline only when
        # ``expansion_expect_contract_enabled`` is set. Default path is byte-identical.
        if self._cfg.expansion.expect_contract_enabled:
            system = f"{system}\n\n{_EXPECT_CONTRACT_ADDENDUM}" if system else _EXPECT_CONTRACT_ADDENDUM
        # Optional prompt prefixes, ordered top-to-bottom: reasoning exemplar (a
        # narrative demonstration) then the imperative rule checklist, then the existing
        # system template. The two env vars are fully independent — either may be set alone.
        prefix_blocks = []
        exemplar_block = _load_reasoning_exemplar()
        if exemplar_block:
            prefix_blocks.append(exemplar_block)
        rules_block = _load_reasoning_rules()
        if not rules_block and not os.environ.get("IDEA_TEST_REASONING_RULES", "").strip():
            # Env var UNSET: fall back to deterministic auto-classification of the root
            # mandate. Manual IDEA_TEST_REASONING_RULES (set) always takes priority.
            rules_block = _auto_reasoning_rules(self._root_mandate(graph))
        if rules_block:
            prefix_blocks.append(rules_block)
        # Single-use human steer injected via the interactive debugger (agent-debug
        # `f`/`feedback`). Surface it once, clearly labeled as a human steer, then
        # consume-and-clear it so it never appears on a later expansion. No-op /
        # byte-identical prompt when no feedback was ever injected.
        if DetailKey.HUMAN_FEEDBACK.value in node.details:
            # Consume-and-clear: remove the key from the live graph node so it never
            # surfaces on a later expansion (of this or any other node). A blank
            # value is cleared without surfacing, so it can't pollute later prompts.
            feedback = node.details.pop(DetailKey.HUMAN_FEEDBACK.value, None)
            if isinstance(feedback, str) and feedback.strip():
                prefix_blocks.append(
                    "HUMAN STEER (operator guidance for THIS expansion only; not part of "
                    f"the task itself):\n{feedback.strip()}"
                )
        # Corrective re-expansion context (opt-in via `got_reexpand_corrective_context_enabled`;
        # engine only writes this detail when the flag is on, so the default path never sees
        # it). Surfaces the triggering confidence-judge/follow-up-detector reason once, then
        # consume-and-clear so it never leaks into a later, unrelated expansion of this or any
        # other node — mirroring the HUMAN_FEEDBACK single-use pattern above.
        if DetailKey.REEXPAND_REASON.value in node.details:
            reexpand_reason = node.details.pop(DetailKey.REEXPAND_REASON.value, None)
            if isinstance(reexpand_reason, str) and reexpand_reason.strip():
                prefix_blocks.append(
                    "CORRECTIVE CONTEXT (the previous step at this node was inadequate for "
                    f"this reason):\n{reexpand_reason.strip()}\n"
                    "Target a source that actually provides the required attribute; do not "
                    "re-read the same insufficient page."
                )
        if prefix_blocks:
            prefix = "\n".join(prefix_blocks)
            system = f"{prefix}\n{system}" if system else prefix
        format_kwargs = dict(
            path_json=path_json,
            parent_id=node.node_id,
            parent_title=node.title,
            blocked_sites=blocked_sites_text,
            errors=errors_text,
            memories=memories_text,
            event_log=event_log_json,
        )
        try:
            user = user_template.format(**format_kwargs) if user_template else json.dumps(
                {
                    "path": serialized,
                    "parent_id": node.node_id,
                    "parent_title": node.title,
                    "blocked_sites": blocked_sites,
                    "errors": errors,
                    "memories": memories_text,
                    "event_log": event_log,
                },
                ensure_ascii=True,
            )
        except KeyError as fmt_err:
            self._logger.error(f"[EXPANSION] User prompt format error (missing key: {fmt_err}) - using manual substitution")
            user = user_template or ""
            for k, v in format_kwargs.items():
                user = user.replace("{" + k + "}", str(v))
        # Opt-in: label the context blob as read-only input and restate the output shape right
        # after it (see ``frame_expansion_user_prompt``). Default path is byte-identical.
        if self._cfg.expansion.input_output_framing_enabled:
            user = frame_expansion_user_prompt(user)
        from agent.app.idea_policies.action_constants import PromptBuilder
        messages = PromptBuilder.build_messages(system_content=system, user_content=user)
        
        total_prompt_size = sum(len(msg.get("content", "")) for msg in messages)
        self._logger.debug(f"[EXPANSION] Prompt size: system={len(system)} chars, user={len(user)} chars, total={total_prompt_size} chars")
        if total_prompt_size > 50000:
            self._logger.warning(f"[EXPANSION] Large prompt detected ({total_prompt_size} chars) - may cause slow expansion. Consider reducing expansion_max_context_nodes or expansion_max_detail_chars")
        
        return messages

    def _extract_url_from_text(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None
        
        import re
        link_pattern = r'\[link:\s*(https?://[^\]]+)\]'
        match = re.search(link_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        url_pattern = r'https?://[^\s\)\]\>\"\']+'
        match = re.search(url_pattern, text)
        if match:
            url = match.group(0).rstrip('.,;:!?)')
            if url.startswith(("http://", "https://")):
                return url
        
        return None
    
    def _root_mandate(self, graph: Optional["IdeaDag"]) -> str:
        """The root node's mandate text (the task statement), or "" if unavailable."""
        if graph is None:
            return ""
        try:
            root = graph.get_node(graph.root_id())
        except Exception:
            return ""
        if root and isinstance(root.details, dict):
            return str(root.details.get("mandate") or "")
        return ""

    def _mandate_urls(self, graph: Optional["IdeaDag"]) -> List[str]:
        """URLs named in the root mandate (the task statement), in order.

        Used to recover a visit candidate's URL when the LLM emitted a visit node
        without one — explicit-URL mandates otherwise fail because the planner names
        the page in the title but drops the URL from details.
        """
        if graph is None:
            return []
        import re
        try:
            root = graph.get_node(graph.root_id())
        except Exception:
            return []
        mandate = ""
        if root and isinstance(root.details, dict):
            mandate = str(root.details.get("mandate") or "")
        if not mandate:
            return []
        urls: List[str] = []
        for u in re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', mandate):
            cleaned = u.rstrip('.,;:!?)]')
            if cleaned not in urls:
                urls.append(cleaned)
        return urls

    def _match_mandate_url(self, title: str, urls: List[str]) -> Optional[str]:
        """Best mandate URL for a URL-less visit candidate, matched by title overlap.

        Scores each URL by how many slug tokens of its last path segment (e.g.
        ``Eiffel_Tower`` -> {eiffel, tower}) appear in the candidate title. With a
        single mandate URL and no signal, returns it (the only sensible target).
        """
        if not urls:
            return None
        if len(urls) == 1:
            return urls[0]
        import re
        title_l = (title or "").lower()
        best_url, best_score = None, 0
        for u in urls:
            slug = u.rstrip('/').rsplit('/', 1)[-1]
            tokens = [t for t in re.split(r'[_\-%]+', slug.lower()) if len(t) > 2]
            score = sum(1 for t in tokens if t in title_l)
            if score > best_score:
                best_url, best_score = u, score
        return best_url if best_score > 0 else None

    def _is_url_from_visit(self, graph: IdeaDag, node_id: str) -> bool:
        node = graph.get_node(node_id)
        if not node:
            return False
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor
        action = NodeDetailsExtractor.get_action(node.details)
        return action == IdeaActionType.VISIT.value
    
    def _extract_url_from_path_context_with_source(self, graph: IdeaDag, node_id: str, candidate_title: str = "") -> tuple[Optional[str], Optional[str]]:
        url = self._extract_url_from_path_context(graph, node_id, candidate_title)
        if not url:
            return None, None
        
        path = graph.path_to_root(node_id)
        candidate_keywords = set()
        if candidate_title:
            import re
            words = re.findall(r'\b\w+\b', candidate_title.lower())
            candidate_keywords = {w for w in words if len(w) > 3}
        
        best_match = None
        best_score = 0
        best_source = None
        first_url = None
        first_source = None
        
        for path_node in reversed(path):
            details = path_node.details or {}
            action_result = details.get(DetailKey.ACTION_RESULT.value)
            if not isinstance(action_result, dict):
                continue
            
            from agent.app.idea_policies.action_constants import ActionResultKey
            action_type = action_result.get(ActionResultKey.ACTION.value)
            
            if action_type == IdeaActionType.SEARCH.value:
                results = action_result.get(ActionResultKey.RESULTS.value) or []
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict):
                            result_url = result.get("url") or result.get("link")
                            if result_url and isinstance(result_url, str) and result_url.startswith(("http://", "https://")):
                                if not first_url:
                                    first_url = result_url
                                    first_source = path_node.node_id
                                
                                if candidate_keywords:
                                    result_text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
                                    score = sum(1 for kw in candidate_keywords if kw in result_text)
                                    if score > best_score:
                                        best_score = score
                                        best_match = result_url
                                        best_source = path_node.node_id
            
            elif action_type == IdeaActionType.VISIT.value:
                links_inline = action_result.get("_links_inline")
                if links_inline and isinstance(links_inline, str):
                    for line in links_inline.split('\n'):
                        if '[link:' in line:
                            extracted_url = self._extract_url_from_text(line)
                            if extracted_url:
                                if not first_url:
                                    first_url = extracted_url
                                    first_source = path_node.node_id
                                
                                if candidate_keywords:
                                    line_lower = line.lower()
                                    score = sum(1 for kw in candidate_keywords if kw in line_lower)
                                    if score > best_score:
                                        best_score = score
                                        best_match = extracted_url
                                        best_source = path_node.node_id
        
        if best_match and best_source:
            return best_match, best_source
        if first_url and first_source:
            return first_url, first_source
        return None, None
    
    def _extract_url_from_path_context(self, graph: IdeaDag, node_id: str, candidate_title: str = "") -> Optional[str]:
        node = graph.get_node(node_id)
        if not node:
            return None
        
        path = graph.path_to_root(node_id)
        candidate_keywords = set()
        if candidate_title:
            import re
            words = re.findall(r'\b\w+\b', candidate_title.lower())
            candidate_keywords = {w for w in words if len(w) > 3}
        
        best_match = None
        best_score = 0
        first_url = None
        
        for path_node in reversed(path):
            details = path_node.details or {}
            action_result = details.get(DetailKey.ACTION_RESULT.value)
            if not isinstance(action_result, dict):
                continue
            
            from agent.app.idea_policies.action_constants import ActionResultKey
            action_type = action_result.get(ActionResultKey.ACTION.value)
            
            if action_type == IdeaActionType.SEARCH.value:
                results = action_result.get(ActionResultKey.RESULTS.value) or []
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict):
                            result_url = result.get("url") or result.get("link")
                            if result_url and isinstance(result_url, str) and result_url.startswith(("http://", "https://")):
                                if not first_url:
                                    first_url = result_url
                                
                                if candidate_keywords:
                                    result_text = (result.get("title", "") + " " + result.get("snippet", "")).lower()
                                    score = sum(1 for kw in candidate_keywords if kw in result_text)
                                    if score > best_score:
                                        best_score = score
                                        best_match = result_url
            
            elif action_type == IdeaActionType.VISIT.value:
                links_inline = action_result.get("_links_inline")
                if links_inline and isinstance(links_inline, str):
                    for line in links_inline.split('\n'):
                        if '[link:' in line:
                            url = self._extract_url_from_text(line)
                            if url:
                                if not first_url:
                                    first_url = url
                                
                                if candidate_keywords:
                                    line_lower = line.lower()
                                    score = sum(1 for kw in candidate_keywords if kw in line_lower)
                                    if score > best_score:
                                        best_score = score
                                        best_match = url
        
        return best_match if best_match else first_url
    
    def _parse_candidates(self, content: Optional[str], graph: Optional[IdeaDag] = None, parent_node_id: Optional[str] = None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not content:
            return [], {}
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            self._logger.error(f"[EXPANSION] JSON PARSE ERROR: {e}")
            self._logger.error(f"[EXPANSION] Content preview (first 500 chars): {content[:500] if content else 'None'}")
            # Salvage a fence-wrapped / prose-wrapped / TRUNCATED plan instead of silently
            # planning nothing (see ``_repair_json_object``); still fails safe when unrepairable.
            data = _repair_json_object(content, required_key="candidates")
            if data is None:
                self._logger.error("[EXPANSION] JSON repair failed - degrading to an empty plan")
                return [], {}
            self._logger.info(
                f"[EXPANSION] Repaired malformed JSON (recovered {len(data.get('candidates') or [])} candidates)"
            )
        except Exception as e:
            self._logger.error(f"[EXPANSION] PARSE EXCEPTION: {e}", exc_info=True)
            self._logger.error(f"[EXPANSION] Content preview (first 500 chars): {content[:500] if content else 'None'}")
            return [], {}

        if not isinstance(data, dict):
            # A bare top-level array is the one non-object shape worth accepting as the plan.
            if isinstance(data, list):
                data = {"candidates": data}
            else:
                self._logger.error(f"[EXPANSION] Response is not a JSON object ({type(data).__name__})")
                return [], {}

        candidates = data.get("candidates", [])
        if not candidates:
            self._logger.error(f"[EXPANSION] NO CANDIDATES IN RESPONSE!")
            self._logger.error(f"[EXPANSION] Response data keys: {list(data.keys())}")
            self._logger.error(f"[EXPANSION] Full response data: {json.dumps(data, indent=2, ensure_ascii=True)[:1000]}")
        meta = data.get("meta")
        if not isinstance(meta, dict):
            # A weak model can emit a truthy-but-wrong-shaped "meta" (seen live: `true`, and
            # separately a flat list) — `meta or {}` only substitutes on FALSY values, so a
            # malformed truthy meta survives to `dict(meta)` below and throws TypeError, killing
            # the whole expansion step instead of just ignoring the bad field.
            meta = {}
        cleaned: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            action = candidate.get(DetailKey.ACTION.value)
            title = candidate.get("title") or ""
            details = candidate.get("details") or {}
            if action:
                details = dict(details)
                details[DetailKey.ACTION.value] = action
            
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            justification = NodeDetailsExtractor.get_justification(candidate)
            if justification:
                details[DetailKey.JUSTIFICATION.value] = str(justification)

            # Opt-in: capture a leaf's measurable output contract. The model may place
            # ``expect`` at the candidate's top level or inside ``details``; both are
            # accepted. No-op when the flag is off or the field is absent (optional).
            if self._cfg.expansion.expect_contract_enabled:
                expect = candidate.get("expect")
                if expect is None:
                    expect = details.get(DetailKey.EXPECT.value)
                if isinstance(expect, str) and expect.strip():
                    details[DetailKey.EXPECT.value] = expect.strip()
                elif DetailKey.EXPECT.value in details:
                    # Drop a non-string/empty expect so it never reaches execution.
                    details.pop(DetailKey.EXPECT.value, None)

            candidate_goal = candidate.get("goal")
            local_goal: Optional[str] = None
            if isinstance(candidate_goal, str) and candidate_goal.strip():
                local_goal = candidate_goal.strip()
            else:
                existing_goal = details.get(DetailKey.GOAL.value) or details.get(DetailKey.ORIGINAL_GOAL.value)
                if isinstance(existing_goal, str) and existing_goal.strip():
                    local_goal = existing_goal.strip()
                elif isinstance(title, str) and title.strip():
                    local_goal = title.strip()

            if local_goal:
                details[DetailKey.GOAL.value] = details.get(DetailKey.GOAL.value) or local_goal
                if not details.get(DetailKey.ORIGINAL_GOAL.value):
                    details[DetailKey.ORIGINAL_GOAL.value] = local_goal
            
            if action == IdeaActionType.VISIT.value:
                url = (
                    details.get(DetailKey.URL.value)
                    or details.get(DetailKey.LINK.value)
                    or details.get("url")
                    or details.get("link")
                    or details.get("optional_url")
                )
                if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    extracted_url = None
                    source_node_id = None
                    
                    if title:
                        extracted_url = self._extract_url_from_text(title)
                    if not extracted_url and justification:
                        extracted_url = self._extract_url_from_text(str(justification))
                    if not extracted_url and graph and parent_node_id:
                        extracted_url, source_node_id = self._extract_url_from_path_context_with_source(graph, parent_node_id, candidate_title=title)
                    
                    if extracted_url:
                        details[DetailKey.URL.value] = extracted_url
                        # A source resolving to `parent_node_id` itself means the URL came from
                        # the node CURRENTLY being expanded (e.g. re-expanding a just-completed
                        # search leaf into a visit follow-up) — its result is already in hand
                        # right here, not something to defer. Wiring `requires_data` to it anyway
                        # deadlocks the engine: that node can only reach DONE once this very child
                        # completes, so `IdeaDagEngine.step()`'s "wait for required data" gate
                        # would return the same node_id forever (silently exhausting the step
                        # budget with no error — the `good_adaptive` self-loop bug).
                        if source_node_id and source_node_id != parent_node_id:
                            source_node = graph.get_node(source_node_id)
                            if source_node:
                                from agent.app.idea_policies.action_constants import NodeDetailsExtractor
                                source_action = NodeDetailsExtractor.get_action(source_node.details)
                                if source_action == IdeaActionType.THINK.value:
                                    details[DetailKey.REQUIRES_DATA.value] = {
                                        "type": "url_from_think",
                                        "source_node_id": source_node_id
                                    }
                                else:
                                    details[DetailKey.REQUIRES_DATA.value] = {
                                        "type": "urls_from_visit" if self._is_url_from_visit(graph, source_node_id) else "urls_from_search",
                                        "source_node_id": source_node_id
                                    }
                            self._logger.info(f"[EXPANSION] Visit candidate requires data from node {source_node_id}: {extracted_url[:60]}...")
                        self._logger.info(f"[EXPANSION] Proactively extracted URL for visit candidate '{title[:50]}...': {extracted_url[:60]}...")
                    else:
                        # Last resort: recover from a URL named in the mandate (explicit-URL
                        # tasks otherwise fail — the planner names the page but drops the URL).
                        recovered = self._match_mandate_url(title, self._mandate_urls(graph))
                        if recovered:
                            details[DetailKey.URL.value] = recovered
                            self._logger.info(f"[EXPANSION] Recovered visit URL from mandate for '{title[:50]}...': {recovered[:60]}...")
                        else:
                            # No URL yet — KEEP the node. In a search-driven task the visit's
                            # URL is resolved at execution time from a sibling search's results
                            # (VisitLeafAction._extract_urls_from_parent_search_results); dropping
                            # it here would break the search->visit pipeline (visits=0).
                            self._logger.warning(f"[EXPANSION] Visit candidate has no URL yet (search-driven?); keeping for runtime resolution: title='{title[:60]}...'")

            if action == IdeaActionType.SEARCH.value:
                details[DetailKey.PROVIDES_DATA.value] = {"type": "urls_from_search"}
            
            cleaned.append(
                {
                    "title": str(title),
                    "details": details,
                    "score": candidate.get("score"),
                }
            )
        return cleaned, dict(meta)
    
    def _create_fallback_candidate(self, node: "IdeaNode", graph: Optional["IdeaDag"] = None) -> Optional[Dict[str, Any]]:
        import re
        title = node.title or ""
        mandate = node.details.get("mandate") or ""
        text_to_search = f"{title} {mandate}"
        
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text_to_search)
        
        if urls:
            url = urls[0]
            self._logger.info(f"[EXPANSION] Fallback: Creating visit candidate for URL found in mandate: {url[:60]}...")
            return {
                "title": f"Visit {url}",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                    DetailKey.URL.value: url,
                    "optional_url": url,
                    DetailKey.JUSTIFICATION.value: "Fallback candidate: URL extracted from mandate",
                },
                "score": None,
            }
        
        text_lower = text_to_search.lower()
        if any(keyword in text_lower for keyword in ["visit", "go to", "fetch", "open", "navigate"]):
            if urls:
                url = urls[0]
                return {
                    "title": f"Visit {url}",
                    "details": {
                        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                        DetailKey.URL.value: url,
                        "optional_url": url,
                        DetailKey.JUSTIFICATION.value: "Fallback candidate: Visit action inferred from mandate",
                    },
                    "score": None,
                }
        
        if any(keyword in text_lower for keyword in ["search", "find", "look for", "query"]):
            query = title[:100] if title else "Search"
            return {
                "title": f"Search: {query}",
                "details": {
                    DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                    DetailKey.QUERY.value: query,
                    DetailKey.JUSTIFICATION.value: "Fallback candidate: Search action inferred from mandate",
                },
                "score": None,
            }
        
        self._logger.warning(f"[EXPANSION] Fallback: Creating generic think node (no URL or search query found)")
        return {
            "title": "Analyze and plan next steps",
            "details": {
                DetailKey.ACTION.value: IdeaActionType.THINK.value,
                DetailKey.JUSTIFICATION.value: "Fallback candidate: Generic think node",
            },
            "score": None,
        }