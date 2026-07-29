"""
Slot filling — turning a live query into the concrete values a retrieved template needs.

:mod:`agent.app.plan_library.retrieval` can say *which* composition strategy fits a node's
intent; :mod:`agent.app.plan_library.schema` can bind values that are already known. This
module is the missing middle: given a matched :class:`~agent.app.plan_library.schema.PlanTemplate`
and the task text, produce the plain ``{slot_name: value}`` mapping ``fill_template`` /
``bind_template`` consume. Without it retrieval is inert — it can recognise an
``argmax_over_n_page_field`` task but has no way to know WHO the candidates are or WHAT field
to read.

One extractor per :class:`~agent.app.plan_library.schema.ExtractionStrategy`, dispatched off
each slot's declared ``extraction``, in three passes:

1. **deterministic, free** — ``REGEX_CANDIDATE_LIST`` (reuses
   ``candidate_coverage.extract_named_candidates``, the proven enumerated-candidate extractor),
   ``REGEX_SHAPE_HINT`` (reuses ``shape_classifier``), and ``DEFAULT`` (the declared default).
2. **one batched LLM call** — every remaining ``LLM`` slot in a SINGLE JSON-mode request built
   from the ``SlotSpec``s themselves (each field's instruction is its ``description``, verbatim),
   never one call per slot, and never speculatively: the caller only gets here after retrieval
   has already qualified a match.
3. **``DERIVED``** — computed from slots the first two passes filled, via a per-template hook
   registry (see :data:`_DERIVED_HOOKS`).

A required slot that no pass could fill raises :class:`TemplateValidationError`, which the
caller (``PlanLibrary.fill_from_query``) turns into "no match" — fail toward silence, exactly
``contract_satisfaction.py``'s stated philosophy and the same downgrade a ``bind_template``
failure already gets.

The LLM interface is duck-typed on :class:`~agent.app.agent_io.AgentIO`'s two methods
(``build_llm_payload`` / ``query_llm_with_fallback``) rather than the class, so a stub can stand
in offline; model/timeout knobs are parameters, because this module deliberately has no
dependency on the engine's config layer (that wiring is the engine's business, not the library's).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from agent.app.idea_policies.action_constants import PromptBuilder
from agent.app.idea_policies.candidate_coverage import extract_named_candidates
from agent.app.idea_policies.shape_classifier import classify_answer_shape, classify_shape
from agent.app.llm_backends import json_instruction_from_response_format
from agent.app.plan_library.schema import (
    LIST_SLOT_KINDS,
    ExtractionStrategy,
    PlanTemplate,
    SlotKind,
    SlotSpec,
    TemplateValidationError,
    validate_template,
)

logger = logging.getLogger(__name__)

#: Deterministic by default: slot extraction is a transcription job, not a creative one.
DEFAULT_TEMPERATURE = 0.0
#: One small JSON object — a template's whole slot set is a few dozen tokens of answer.
DEFAULT_MAX_TOKENS = 1200
DEFAULT_TIMEOUT_SECONDS = 90.0
#: The task statement is pasted verbatim into the prompt; clip a pathological one rather
#: than letting a single retrieval hit blow the context budget.
MAX_MANDATE_CHARS = 8000

_SCHEMA_NAME = "plan_template_slots"

_SYSTEM_PROMPT = (
    "You extract the PARAMETERS of an already-chosen plan template from a task statement.\n"
    "The strategy is fixed and is NOT your concern: your only job is to say what THIS task's "
    "values are, so the plan can be instantiated.\n"
    "Rules:\n"
    "- Copy the task's own wording and units; do not paraphrase into your own terms.\n"
    "- Never invent entities, numbers or fields the task does not state.\n"
    "- Do NOT solve the task, do not answer its question, and do not add commentary.\n"
    "- If the task genuinely does not state a value for a field, use null for that field.\n"
    "Respond with a single JSON object and nothing else."
)

#: Superlative direction words, used ONLY by the ``REGEX_SHAPE_HINT`` extractor to turn the
#: shape classifier's ``argmax`` label into an argmax/argmin choice. Kept local (and tiny)
#: rather than importing ``shape_classifier``'s private lists.
_MIN_WORDS = (
    "lowest", "smallest", "shortest", "fewest", "least", "minimum", "closest", "nearest",
    "youngest", "narrowest", "lightest", "argmin",
)
_MAX_WORDS = (
    "highest", "largest", "biggest", "tallest", "deepest", "longest", "greatest", "most",
    "maximum", "widest", "heaviest", "oldest", "furthest", "farthest", "argmax",
)
_SUM_WORDS = ("sum of", "add up", "added together", "combined total", "total number of", "in total")


# --------------------------------------------------------------------------------------
# DERIVED — per-template hooks
# --------------------------------------------------------------------------------------

#: ``{template_id: {slot_name: fn(values, template) -> value}}``.
#:
#: EMPTY on purpose: none of the six seed templates declares ``ExtractionStrategy.DERIVED``
#: (the closest thing is ``computed_ratio_argmax``'s ``fields``, which reads as the
#: concatenation of ``numerator_field`` and ``denominator_field`` but is declared ``llm``,
#: because the mandate's own phrasing of the pair is better prose than anything a join
#: produces). The registry exists so a mined or hand-authored template CAN declare a derived
#: slot without this module growing a special case; an unregistered derived slot simply stays
#: unfilled and falls through to its default / the required-slot check.
_DERIVED_HOOKS: Dict[str, Dict[str, Callable[[Mapping[str, Any], PlanTemplate], Any]]] = {}


def register_derived(
    template_id: str,
    slot_name: str,
    fn: Optional[Callable[[Mapping[str, Any], PlanTemplate], Any]] = None,
) -> Any:
    """Register (or, used bare, decorate) the hook that computes one template's derived slot."""

    def _register(func: Callable[[Mapping[str, Any], PlanTemplate], Any]):
        _DERIVED_HOOKS.setdefault(str(template_id), {})[str(slot_name)] = func
        return func

    return _register if fn is None else _register(fn)


def unregister_derived(template_id: str, slot_name: Optional[str] = None) -> None:
    """Drop one derived hook (or a whole template's) — used by tests to stay isolated."""
    hooks = _DERIVED_HOOKS.get(str(template_id))
    if hooks is None:
        return
    if slot_name is None:
        _DERIVED_HOOKS.pop(str(template_id), None)
    else:
        hooks.pop(str(slot_name), None)
        if not hooks:
            _DERIVED_HOOKS.pop(str(template_id), None)


def derived_hook(
    template_id: str, slot_name: str
) -> Optional[Callable[[Mapping[str, Any], PlanTemplate], Any]]:
    return _DERIVED_HOOKS.get(str(template_id), {}).get(str(slot_name))


# --------------------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------------------


async def extract_slot_values(
    template: Union[PlanTemplate, Mapping[str, Any]],
    query_text: str = "",
    mandate: str = "",
    io: Any = None,
    *,
    model_name: Optional[str] = None,
    fallback_model: Optional[str] = None,
    timeout_seconds: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """Best-effort: fill every declared slot of ``template`` via its ``ExtractionStrategy``.

    :param query_text: the node's intent text — what retrieval matched on.
    :param mandate: the full task statement, RAW. The deterministic extractors need it: an
        enumerated candidate list lives in the mandate rather than in a child node's goal, and
        ``extract_named_candidates`` matches at line starts, so a whitespace-collapsed copy
        (which is what ``retrieve()`` keeps as ``query_text``) has no enumeration left to find.
        Falls back to ``query_text`` when the caller has only that — still correct, just no
        longer free: the candidate list then rides along in the LLM request.
    :param io: an ``AgentIO``-like object (``build_llm_payload`` + ``query_llm_with_fallback``).
        ``None`` disables the LLM pass entirely, which is exactly what an offline caller wants:
        deterministic slots still fill, and a template needing more is simply not applicable.
    :returns: ``{slot_name: value}`` ready for ``fill_template`` / ``bind_template``.
    :raises TemplateValidationError: on an invalid template, or a required slot that nothing
        could fill — the caller treats that as "no match", matching
        ``contract_satisfaction.py``'s fail-toward-silence philosophy, same as a
        ``bind_template()`` failure already does.
    """
    tpl = validate_template(template)
    mandate_text = _clean(mandate) or _clean(query_text)
    goal_text = _clean(query_text)

    values: Dict[str, Any] = deterministic_slot_values(tpl, mandate_text)

    pending = [spec for spec in tpl.slots if _needs_llm(spec, values)]
    if pending:
        supplied = await llm_slot_values(
            tpl,
            pending,
            query_text=goal_text,
            mandate=mandate_text,
            io=io,
            model_name=model_name,
            fallback_model=fallback_model,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for spec in pending:
            value = _coerce(spec, supplied.get(spec.name))
            if _is_filled(value):
                values[spec.name] = value

    _apply_derived(tpl, values)

    missing = [
        spec.name
        for spec in tpl.slots
        # Mirrors ``schema._resolve_slot_values``: a declared default satisfies a required
        # slot, so only a required slot with NO default is genuinely unfillable.
        if spec.required and spec.default is None and not _is_filled(values.get(spec.name))
    ]
    if missing:
        raise TemplateValidationError(
            f"template '{tpl.template_id}': could not extract required slot(s) "
            f"{', '.join(missing)} from the query"
        )
    return values


def deterministic_slot_values(
    template: Union[PlanTemplate, Mapping[str, Any]], mandate: str
) -> Dict[str, Any]:
    """The free pass: every slot a regex or a declared default can fill. No I/O, no LLM."""
    tpl = template if isinstance(template, PlanTemplate) else validate_template(template)
    text = _clean(mandate)
    out: Dict[str, Any] = {}
    for spec in tpl.slots:
        value = _coerce(spec, _deterministic_value(spec, text))
        if _is_filled(value):
            out[spec.name] = value
    return out


def _deterministic_value(spec: SlotSpec, mandate: str) -> Any:
    if spec.extraction is ExtractionStrategy.REGEX_CANDIDATE_LIST:
        return candidate_list_value(spec, mandate)
    if spec.extraction is ExtractionStrategy.REGEX_SHAPE_HINT:
        return shape_hint_value(spec, mandate)
    if spec.extraction is ExtractionStrategy.DEFAULT:
        # ``schema._resolve_slot_values`` would apply this default anyway; echoing it here
        # costs nothing and makes the returned mapping — which the retrieval log records and
        # a human reviews — show the value that will actually be used instead of a hole.
        return spec.default
    return None


def _apply_derived(template: PlanTemplate, values: Dict[str, Any]) -> None:
    """Third pass: slots computed from already-filled ones. Never fatal on its own."""
    for spec in template.slots:
        if spec.extraction is not ExtractionStrategy.DERIVED or _is_filled(values.get(spec.name)):
            continue
        hook = derived_hook(template.template_id, spec.name)
        if hook is None:
            logger.debug(
                "Plan library: template '%s' slot '%s' is DERIVED but no hook is registered",
                template.template_id, spec.name,
            )
            continue
        try:
            value = _coerce(spec, hook(values, template))
        except Exception as exc:  # noqa: BLE001 — a derived hook never breaks slot fill
            logger.warning(
                "Plan library: derived slot '%s' of '%s' failed: %s",
                spec.name, template.template_id, exc,
            )
            continue
        if _is_filled(value):
            values[spec.name] = value


# --------------------------------------------------------------------------------------
# REGEX_CANDIDATE_LIST
# --------------------------------------------------------------------------------------


def candidate_list_value(spec: SlotSpec, mandate: str) -> Optional[List[Dict[str, Any]]]:
    """The mandate's enumerated candidate list, shaped for an ``ENTITY_LIST`` slot.

    Reuses ``candidate_coverage.extract_named_candidates`` unchanged — the proven extractor
    that already distinguishes a candidate enumeration ("1. Jengish Chokusu (Kyrgyzstan)")
    from an INSTRUCTION list ("1. Identify the poet ...") and fails open to ``[]``.

    Elements are ``{"name": ...}`` mappings: ``schema._normalize_item`` accepts bare strings
    too, but a mapping is the shape the slot descriptions declare (``[{name, qualifier?}, ...]``),
    it is what the retrieval log records for review, and it leaves room for a later pass to
    attach a ``qualifier`` without changing the contract. ``key`` is deliberately NOT set:
    ``_normalize_item`` slugs it from the name, and duplicating that here would be a second
    place for leaf ids to be decided.

    Restricted to ``ENTITY_LIST``: a ``HOP_LIST`` element also needs a per-hop ``field`` that
    a flat enumeration cannot supply, so filling one from this extractor would only produce a
    template that fails to bind.
    """
    if spec.kind is not SlotKind.ENTITY_LIST:
        return None
    names = extract_named_candidates(mandate or "")
    out: List[Dict[str, Any]] = []
    seen = set()
    for name in names:
        cleaned = _clean(name)
        marker = cleaned.casefold()
        if not cleaned or marker in seen:
            continue
        seen.add(marker)
        out.append({"name": cleaned})
    return out or None


# --------------------------------------------------------------------------------------
# REGEX_SHAPE_HINT
# --------------------------------------------------------------------------------------


def shape_hint_value(spec: SlotSpec, mandate: str) -> Optional[str]:
    """A slot value derived from the deterministic shape classifiers, or ``None``.

    Deliberately narrow, and a WEAKER signal than the other extractors: ``shape_classifier``
    knows a mandate's *shape*, not its parameters, so the only slots it can speak to are
    ``AGGREGATION_OP`` ones (which operator: argmax / argmin / count_where / sum) — for
    anything else it can offer no more than the archetype label ``classify_shape`` already
    hands ``retrieval.archetype_hint_for`` for the rerank.

    NO seed template declares this strategy (all six get their operator wording from the LLM
    pass or a default), so this is dispatch completeness for future/mined templates rather
    than a live path. Fails open to ``None`` exactly like the classifiers it wraps.
    """
    text = (mandate or "").lower()
    if not text.strip():
        return None
    if spec.kind is SlotKind.AGGREGATION_OP:
        try:
            answer_shape = classify_answer_shape(mandate or "")
        except Exception:  # noqa: BLE001 — a hint never breaks slot fill
            return None
        if answer_shape == "argmax":
            return "argmin" if _wants_minimum(text) else "argmax"
        if answer_shape == "count":
            return "count_where"
        if answer_shape == "computation" and any(word in text for word in _SUM_WORDS):
            return "sum"
        return None
    try:
        return classify_shape(mandate or "")
    except Exception:  # noqa: BLE001
        return None


def _wants_minimum(text: str) -> bool:
    """True when the mandate asks for the LOW end and nothing asks for the high end."""
    return any(w in text for w in _MIN_WORDS) and not any(w in text for w in _MAX_WORDS)


# --------------------------------------------------------------------------------------
# LLM — one batched, schema-driven JSON call
# --------------------------------------------------------------------------------------


def _needs_llm(spec: SlotSpec, values: Mapping[str, Any]) -> bool:
    """Which slots ride along in the single LLM request.

    Every unfilled ``LLM`` slot, plus any REQUIRED slot whose deterministic extractor came up
    empty (e.g. a mandate that names its candidates inline in prose rather than as a numbered
    list, which ``extract_named_candidates`` correctly refuses to parse). The call is already
    being made for the ``LLM`` slots, so letting a required hole ride along costs nothing and
    saves an otherwise-good match from being discarded. ``DERIVED`` slots never ride along:
    they are computed after this pass, from its output.
    """
    if _is_filled(values.get(spec.name)) or spec.extraction is ExtractionStrategy.DERIVED:
        return False
    return spec.extraction is ExtractionStrategy.LLM or spec.required


async def llm_slot_values(
    template: PlanTemplate,
    specs: Sequence[SlotSpec],
    *,
    query_text: str = "",
    mandate: str = "",
    io: Any = None,
    model_name: Optional[str] = None,
    fallback_model: Optional[str] = None,
    timeout_seconds: Optional[float] = DEFAULT_TIMEOUT_SECONDS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """ONE JSON-mode call for ALL of ``specs``. Returns ``{}`` on any failure.

    Follows the established convention for a schema whose fields are not all required
    (``expansion.py`` / ``MergeLeafAction``): the JSON Schema is conveyed as a text hint
    appended to the system message and the request goes out in plain ``json_object`` mode,
    because strict structured output demands that ``required`` enumerate every property —
    which would forbid a template from having an optional slot.
    """
    if not specs:
        return {}
    if io is None or not hasattr(io, "query_llm_with_fallback"):
        logger.debug(
            "Plan library: no LLM available for %d slot(s) of '%s'", len(specs), template.template_id
        )
        return {}

    schema = build_slot_schema(template, specs)
    system = _SYSTEM_PROMPT
    hint = json_instruction_from_response_format({"type": "json_schema", "json_schema": schema})
    if hint:
        system = f"{system}\n\n{hint}"
    messages = PromptBuilder.build_messages(
        system_content=system,
        user_content=_user_prompt(template, specs, query_text=query_text, mandate=mandate),
    )

    try:
        payload = io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=None,
        )
        content = await io.query_llm_with_fallback(
            payload,
            model_name=model_name,
            fallback_model=fallback_model,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 — an unfilled slot is a "no match", not a crash
        logger.warning("Plan library: slot-fill LLM call failed for '%s': %s", template.template_id, exc)
        return {}

    data = _parse_json_object(content)
    if data is None:
        logger.info(
            "Plan library: slot-fill response for '%s' was not usable JSON (%d chars)",
            template.template_id, len(content or ""),
        )
        return {}
    return data


def build_slot_schema(template: PlanTemplate, specs: Sequence[SlotSpec]) -> Dict[str, Any]:
    """The JSON Schema for one batched slot-fill call, built FROM the ``SlotSpec``s.

    Schema-driven, not template-specific: each field's type comes from its ``SlotKind`` and
    its instruction is the ``SlotSpec.description`` verbatim, so a new template (or a mined
    one) needs no code here.
    """
    properties: Dict[str, Any] = {}
    required: List[str] = []
    for spec in specs:
        properties[spec.name] = _property_schema(spec)
        if spec.required:
            required.append(spec.name)
    return {
        "name": _SCHEMA_NAME,
        "schema": {
            "type": "object",
            "description": f"Parameters of the '{template.template_id}' plan template.",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def _property_schema(spec: SlotSpec) -> Dict[str, Any]:
    """One slot's JSON Schema fragment, keyed off its ``SlotKind``."""
    node: Dict[str, Any]
    if spec.kind is SlotKind.ENTITY_LIST:
        node = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The entity's name as the task writes it."},
                    "qualifier": {
                        "type": "string",
                        "description": "Any disambiguating detail the task gives (country, region, "
                                       "parenthetical). Empty string when there is none.",
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        }
    elif spec.kind is SlotKind.HOP_LIST:
        node = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The entity this hop opens, named as the previous hop reveals it.",
                    },
                    "field": {
                        "type": "string",
                        "description": "What to read on that entity's page at this hop.",
                    },
                },
                "required": ["label", "field"],
                "additionalProperties": False,
            },
        }
    elif spec.kind is SlotKind.THRESHOLD:
        node = {
            "type": "object",
            "properties": {
                "comparator": {"type": "string"},
                "value": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["comparator", "value"],
            "additionalProperties": False,
        }
    else:
        node = {"type": "string"}
    if spec.kind in LIST_SLOT_KINDS:
        if spec.min_arity is not None:
            node["minItems"] = spec.min_arity
        if spec.max_arity is not None:
            node["maxItems"] = spec.max_arity
    if spec.description:
        node["description"] = spec.description
    return node


def _user_prompt(
    template: PlanTemplate, specs: Sequence[SlotSpec], *, query_text: str, mandate: str
) -> str:
    """The task text plus a field-by-field brief written from the ``SlotSpec``s."""
    lines = [f"TEMPLATE: {template.title or template.template_id} ({template.template_id})"]
    if template.description:
        lines.append(f"WHAT IT DOES: {template.description}")
    lines.append("")
    lines.append("TASK STATEMENT:")
    lines.append(_clip(mandate, MAX_MANDATE_CHARS) or "(not supplied)")
    if query_text and query_text != mandate:
        lines.append("")
        lines.append("CURRENT GOAL (the step this plan is being instantiated for):")
        lines.append(_clip(query_text, MAX_MANDATE_CHARS))
    lines.append("")
    lines.append("FIELDS TO EXTRACT — return a JSON object with exactly these keys:")
    for spec in specs:
        arity = ""
        if spec.kind in LIST_SLOT_KINDS and (spec.min_arity or spec.max_arity):
            bounds = []
            if spec.min_arity:
                bounds.append(f"at least {spec.min_arity}")
            if spec.max_arity:
                bounds.append(f"at most {spec.max_arity}")
            arity = f" [{' and '.join(bounds)} items]"
        lines.append(
            f'- "{spec.name}" ({"required" if spec.required else "optional"}{arity}): '
            f"{spec.description or spec.kind.value}"
        )
    return "\n".join(lines)


def _parse_json_object(content: Any) -> Optional[Dict[str, Any]]:
    """``json.loads`` with the repair fallback the expansion stage already uses.

    A cheap model fences, truncates or pads its JSON in exactly the same ways here as it does
    when planning, so this reuses ``expansion._repair_json_object`` rather than growing a
    second, differently-buggy repair. The import is deferred (not module scope) to keep this
    module free of a static edge into ``idea_policies.expansion``: the on-demand plan-library
    action will later be imported FROM ``idea_policies.actions``, and a static edge back would
    thread that cycle through the package ``__init__``.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        data = json.loads(content)
    except ValueError:
        data = None
    if isinstance(data, dict):
        return data
    try:
        from agent.app.idea_policies.expansion import _repair_json_object
    except Exception:  # noqa: BLE001 - pragma: no cover
        return None
    repaired = _repair_json_object(content)
    return repaired if isinstance(repaired, dict) else None


# --------------------------------------------------------------------------------------
# value shaping
# --------------------------------------------------------------------------------------


def _coerce(spec: SlotSpec, raw: Any) -> Any:
    """Shape one extracted value into what ``schema`` expects for that ``SlotKind``.

    Tolerant of the shapes a weak model actually emits (a bare string where a list was asked
    for, a list of strings where a list of objects was), because the alternative — rejecting
    the whole match on a formatting slip — throws away a correct retrieval.
    """
    if raw is None:
        return None
    if spec.kind in LIST_SLOT_KINDS:
        items = raw if isinstance(raw, (list, tuple)) else [raw]
        out: List[Dict[str, Any]] = []
        for element in items:
            item = _coerce_item(spec, element)
            if item:
                out.append(item)
        return out or None
    if spec.kind is SlotKind.THRESHOLD:
        return dict(raw) if isinstance(raw, Mapping) else _text(raw) or None
    return _text(raw) or None


def _coerce_item(spec: SlotSpec, raw: Any) -> Optional[Dict[str, Any]]:
    """One list element -> the ``{name|label, ...}`` mapping ``schema._normalize_item`` reads."""
    if isinstance(raw, Mapping):
        item = {str(k): v for k, v in raw.items() if v is not None and _text(v)}
        # ``_normalize_item`` resolves an item's name from name|label|entity_hint, so a hop
        # element carrying only ``label`` is already well-formed.
        return item if any(item.get(k) for k in ("name", "label", "entity_hint")) else None
    text = _text(raw)
    if not text:
        return None
    return {"label": text} if spec.kind is SlotKind.HOP_LIST else {"name": text}


def _is_filled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def _text(value: Any) -> str:
    if value is None or isinstance(value, (list, tuple, dict)):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return _clean(value if isinstance(value, str) else str(value))


def _clean(value: Any) -> str:
    """Collapse whitespace runs but keep line structure — a numbered candidate list is
    line-oriented, and ``extract_named_candidates`` matches at line starts."""
    text = str(value or "")
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


def _clip(text: str, limit: int) -> str:
    out = str(text or "")
    if len(out) <= limit:
        return out
    return out[:limit].rsplit(" ", 1)[0] + " ... [truncated]"
