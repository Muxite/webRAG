"""
Plan-template schema — the parameterized generalization of a compiled plan.

A *plan template* is a hand-authored (later: mined) composition strategy for one task shape
(argmax over N page reads, an N-hop entity chain, a computed-ratio argmax, ...). It is exactly
:mod:`agent.app.testing.compiled_plan`'s ``{leaves: [{id, instruction, expect, depends_on}],
aggregation}`` shape with two generalizations:

  * entities are ``<<slot>>`` placeholders instead of hardcoded names, and
  * a leaf may declare ``for_each`` to repeat itself once per element of a *list* slot.

:func:`fill_template` binds concrete slot values and returns compiled-plan's shape **exactly**,
so a filled template is consumable by ``compiled_plan.normalize_plan / validate_plan /
topological_waves / substitute_deps`` with zero changes there. :func:`bind_template` returns the
same thing plus the structured bindings each leaf was filled from (the current ``for_each`` item,
the derived search query) — that richer view is what
:mod:`agent.app.idea_policies.plan_library` turns into native Graph-of-Thought candidates,
because a plain compiled-plan dict has already thrown the slot structure away.

**Placeholder syntax is ``<<slot>>``, deliberately NOT ``{slot}``**: ``compiled_plan.substitute_deps``
already owns ``{dep_id}`` at *execution* time, so a template may write a literal ``{hop1}`` into an
instruction to have a prior hop's resolved fact substituted at run time, and filling can never
clobber it.

Pure module: no I/O, no LLM. Retrieval and slot extraction live elsewhere; ``fill_template``
takes slot values as a plain mapping the caller has already produced.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from agent.app.testing import compiled_plan


class TemplateValidationError(ValueError):
    """Raised when a plan template is structurally invalid or cannot be filled."""


class SlotKind(str, Enum):
    """What a slot holds — drives arity checks and the derived search query."""

    ENTITY_LIST = "entity_list"        # ordered [{name, qualifier?, key?}, ...]
    ENTITY = "entity"
    FIELD = "field"                    # the attribute/quantity to read, free text
    CRITERION = "criterion"            # binary/categorical property being tested
    THRESHOLD = "threshold"            # {comparator, value, unit}
    AGGREGATION_OP = "aggregation_op"  # argmax|argmin|sum|count_where|negation_select
    SOURCE_TYPE = "source_type"        # e.g. "the authoritative Wikipedia page", has a default
    HOP_LIST = "hop_list"              # ordered [{label, entity_hint?, field}, ...] for chains


class ExtractionStrategy(str, Enum):
    """How a slot's value is obtained from a mandate (consumed by the slot-fill layer)."""

    REGEX_CANDIDATE_LIST = "regex_candidate_list"  # reuses idea_policies/candidate_coverage.py
    REGEX_SHAPE_HINT = "regex_shape_hint"          # reuses idea_policies/shape_classifier.py
    LLM = "llm"          # one JSON-mode call, fired only on a retrieval hit
    DERIVED = "derived"  # computed from another already-filled slot
    DEFAULT = "default"


#: Slot kinds whose value is an ordered list — the only legal ``for_each`` targets.
LIST_SLOT_KINDS = frozenset({SlotKind.ENTITY_LIST, SlotKind.HOP_LIST})

#: The loop variable a ``for_each`` blueprint may reference as ``<<item.*>>``.
ITEM_REF = "item"

_PLACEHOLDER_RE = re.compile(r"<<\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*>>")


@dataclass
class SlotSpec:
    """One parameter of a template."""

    name: str
    kind: SlotKind
    description: str = ""
    required: bool = True
    extraction: ExtractionStrategy = ExtractionStrategy.LLM
    default: Any = None
    min_arity: Optional[int] = None
    max_arity: Optional[int] = None


@dataclass
class LeafBlueprint:
    """One leaf of a template — or, with ``for_each``, one leaf *per list element*."""

    id_pattern: str          # e.g. "<<item.key>>_field"; must be unique after filling
    instruction: str         # "<<slot>>" / "<<item.field>>" placeholders
    expect: str              # MUST re-embed the concrete filled entity name: contract
                             # satisfaction keys off a real capitalized subject token plus
                             # quantity-cue wording, exactly like hand-authored leaves do.
    depends_on: List[str] = field(default_factory=list)  # other blueprints' ``id_pattern``s
    for_each: Optional[str] = None  # name of a list slot this blueprint repeats over


@dataclass
class PlanTemplate:
    """A retrievable, parameterized composition strategy."""

    template_id: str
    schema_version: int = 1
    # Open string, not a closed enum, so a future mining pass can introduce new shapes
    # with no schema migration.
    archetype: str = ""
    title: str = ""
    description: str = ""
    # What actually gets embedded: description + concrete trigger phrases harvested from the
    # real task statements this template generalizes (a manual HyDE anchor).
    embedding_text: str = ""
    # {source: "hand_authored"|"mined", based_on_tasks: [...], mined_from_trace_ids: [...]}
    provenance: Dict[str, Any] = field(default_factory=dict)
    slots: List[SlotSpec] = field(default_factory=list)
    leaves: List[LeafBlueprint] = field(default_factory=list)
    aggregation: str = ""
    min_candidates: Optional[int] = None
    max_candidates: Optional[int] = None

    def slot(self, name: str) -> Optional[SlotSpec]:
        for spec in self.slots:
            if spec.name == name:
                return spec
        return None


@dataclass(frozen=True)
class FilledLeaf:
    """One concrete leaf produced by binding a :class:`LeafBlueprint`."""

    id: str
    instruction: str
    expect: str
    depends_on: List[str]
    #: Structured web-search query, built from the bound slot VALUES (never re-parsed out of
    #: the filled instruction prose) — see :func:`_build_query`.
    query: str
    #: ``id_pattern`` of the blueprint this leaf came from.
    blueprint_id: str
    #: The normalized ``for_each`` item this instance was bound to (``None`` when the
    #: blueprint does not repeat).
    item: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class FilledPlan:
    """A fully bound template: compiled-plan leaves plus the bindings they came from."""

    template_id: str
    leaves: List[FilledLeaf]
    aggregation: str

    def to_plan(self) -> Dict[str, Any]:
        """Return ``compiled_plan``'s ``{"leaves": [...], "aggregation": "..."}`` shape exactly."""
        return {
            "leaves": [
                {
                    "id": leaf.id,
                    "instruction": leaf.instruction,
                    "expect": leaf.expect,
                    "depends_on": list(leaf.depends_on),
                }
                for leaf in self.leaves
            ],
            "aggregation": self.aggregation,
        }


# --------------------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------------------


def normalize_template(raw: Union[PlanTemplate, Mapping[str, Any]]) -> PlanTemplate:
    """Return a :class:`PlanTemplate` from a JSON-ish mapping (or another template).

    Forgiving about missing optional fields (mirroring ``compiled_plan.normalize_plan``);
    strictness lives in :func:`validate_template`. Unknown ``kind`` / ``extraction`` values
    are the one exception — they cannot be defaulted meaningfully, so they raise.
    """
    if isinstance(raw, PlanTemplate):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        raise TemplateValidationError(f"template must be a mapping, got {type(raw).__name__}")

    slots = [_normalize_slot(idx, spec) for idx, spec in enumerate(_as_list(raw.get("slots"), "slots"))]
    leaves = [_normalize_leaf(idx, leaf) for idx, leaf in enumerate(_as_list(raw.get("leaves"), "leaves"))]

    return PlanTemplate(
        template_id=str(raw.get("template_id") or "").strip(),
        schema_version=_as_int(raw.get("schema_version"), default=1),
        archetype=str(raw.get("archetype") or "").strip(),
        title=str(raw.get("title") or "").strip(),
        description=str(raw.get("description") or "").strip(),
        embedding_text=str(raw.get("embedding_text") or "").strip(),
        provenance=dict(raw.get("provenance") or {}),
        slots=slots,
        leaves=leaves,
        aggregation=str(raw.get("aggregation") or "").strip(),
        min_candidates=_as_opt_int(raw.get("min_candidates")),
        max_candidates=_as_opt_int(raw.get("max_candidates")),
    )


def _normalize_slot(idx: int, raw: Any) -> SlotSpec:
    if isinstance(raw, SlotSpec):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        raise TemplateValidationError(f"slot #{idx} must be a mapping, got {type(raw).__name__}")
    name = str(raw.get("name") or "").strip()
    return SlotSpec(
        name=name,
        kind=_coerce_enum(SlotKind, raw.get("kind"), f"slot '{name or idx}'", "kind"),
        description=str(raw.get("description") or "").strip(),
        required=bool(raw.get("required", True)),
        extraction=_coerce_enum(
            ExtractionStrategy,
            raw.get("extraction") or ExtractionStrategy.LLM,
            f"slot '{name or idx}'",
            "extraction",
        ),
        default=raw.get("default"),
        min_arity=_as_opt_int(raw.get("min_arity")),
        max_arity=_as_opt_int(raw.get("max_arity")),
    )


def _normalize_leaf(idx: int, raw: Any) -> LeafBlueprint:
    if isinstance(raw, LeafBlueprint):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        raise TemplateValidationError(f"leaf #{idx} must be a mapping, got {type(raw).__name__}")
    deps_raw = raw.get("depends_on") or []
    if isinstance(deps_raw, str):
        deps_raw = [deps_raw]
    if not isinstance(deps_raw, list):
        raise TemplateValidationError(f"leaf #{idx}: depends_on must be a list")
    for_each = str(raw.get("for_each") or "").strip() or None
    return LeafBlueprint(
        id_pattern=str(raw.get("id_pattern") or "").strip(),
        instruction=str(raw.get("instruction") or "").strip(),
        expect=str(raw.get("expect") or "").strip(),
        depends_on=[str(dep).strip() for dep in deps_raw if str(dep).strip()],
        for_each=for_each,
    )


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------


def validate_template(raw: Union[PlanTemplate, Mapping[str, Any]]) -> PlanTemplate:
    """Normalize then strictly validate a template; return it or raise.

    Checks: identity/aggregation present; unique non-empty slot names; unique non-empty
    blueprint ``id_pattern``s with non-empty instruction/expect; every ``<<ref>>`` in a
    blueprint resolves to a declared slot (or to ``item`` inside a ``for_each``); every
    ``<<ref>>`` in ``aggregation`` resolves to a declared slot (``item`` is per-leaf only);
    ``for_each`` names a declared LIST slot and the blueprint's ``id_pattern`` varies per item;
    every ``depends_on`` entry names a declared blueprint; and the blueprint dependency graph
    is acyclic (via ``compiled_plan.topological_waves``).

    A ``depends_on`` self-edge is legal **only** on a ``for_each`` blueprint, where it means
    "each instance depends on the previous instance" (the N-hop chain shape); it is therefore
    stripped before the cycle check. Concrete per-instance cycles are impossible by
    construction, and :func:`bind_template` re-checks the filled plan with
    ``compiled_plan.validate_plan`` regardless.
    """
    template = normalize_template(raw)

    if not template.template_id:
        raise TemplateValidationError("template_id is required")
    if not template.aggregation:
        raise TemplateValidationError(f"template '{template.template_id}': aggregation is required")
    if not template.leaves:
        raise TemplateValidationError(f"template '{template.template_id}': no leaves")

    slots_by_name: Dict[str, SlotSpec] = {}
    for spec in template.slots:
        if not spec.name:
            raise TemplateValidationError(f"template '{template.template_id}': a slot has an empty name")
        if spec.name in slots_by_name:
            raise TemplateValidationError(f"template '{template.template_id}': duplicate slot '{spec.name}'")
        if spec.name == ITEM_REF:
            raise TemplateValidationError(
                f"template '{template.template_id}': '{ITEM_REF}' is reserved for the for_each loop variable"
            )
        slots_by_name[spec.name] = spec

    seen_patterns: set = set()
    for idx, leaf in enumerate(template.leaves):
        where = f"template '{template.template_id}' leaf #{idx}"
        if not leaf.id_pattern:
            raise TemplateValidationError(f"{where}: id_pattern is required")
        if leaf.id_pattern in seen_patterns:
            raise TemplateValidationError(f"{where}: duplicate id_pattern '{leaf.id_pattern}'")
        seen_patterns.add(leaf.id_pattern)
        if not leaf.instruction:
            raise TemplateValidationError(f"{where} ('{leaf.id_pattern}'): instruction is required")
        if not leaf.expect:
            raise TemplateValidationError(f"{where} ('{leaf.id_pattern}'): expect is required")

        if leaf.for_each is not None:
            spec = slots_by_name.get(leaf.for_each)
            if spec is None:
                raise TemplateValidationError(
                    f"{where} ('{leaf.id_pattern}'): for_each references undeclared slot '{leaf.for_each}'"
                )
            if spec.kind not in LIST_SLOT_KINDS:
                raise TemplateValidationError(
                    f"{where} ('{leaf.id_pattern}'): for_each slot '{leaf.for_each}' has kind "
                    f"'{spec.kind.value}', which is not a list kind "
                    f"({', '.join(sorted(k.value for k in LIST_SLOT_KINDS))})"
                )
            if not any(ref.split(".")[0] == ITEM_REF for ref in _refs(leaf.id_pattern)):
                raise TemplateValidationError(
                    f"{where} ('{leaf.id_pattern}'): a for_each id_pattern must reference "
                    f"<<{ITEM_REF}.*>> so each instance gets a unique leaf id"
                )

        # ``depends_on`` entries are other blueprints' id_patterns, not free text: they are
        # checked against ``by_pattern`` below (a stricter check), and may legally carry the
        # TARGET's ``<<item.*>>`` refs even when this blueprint does not repeat.
        allowed = set(slots_by_name) | ({ITEM_REF} if leaf.for_each else set())
        for text in (leaf.id_pattern, leaf.instruction, leaf.expect):
            _check_refs(text, allowed, where=f"{where} ('{leaf.id_pattern}')")

    _check_refs(template.aggregation, set(slots_by_name), where=f"template '{template.template_id}' aggregation")

    by_pattern = {leaf.id_pattern: leaf for leaf in template.leaves}
    dummy: List[Dict[str, Any]] = []
    for leaf in template.leaves:
        deps: List[str] = []
        for dep in leaf.depends_on:
            if dep not in by_pattern:
                raise TemplateValidationError(
                    f"template '{template.template_id}' leaf '{leaf.id_pattern}': depends_on "
                    f"references unknown blueprint '{dep}'"
                )
            if dep == leaf.id_pattern:
                if not leaf.for_each:
                    raise TemplateValidationError(
                        f"template '{template.template_id}' leaf '{leaf.id_pattern}': depends on itself"
                    )
                continue  # for_each self-edge == chain over successive instances
            deps.append(dep)
        dummy.append({"id": leaf.id_pattern, "instruction": leaf.instruction, "depends_on": deps})

    try:
        compiled_plan.topological_waves(dummy)
    except compiled_plan.PlanValidationError as exc:
        raise TemplateValidationError(f"template '{template.template_id}': {exc}") from exc

    return template


# --------------------------------------------------------------------------------------
# filling
# --------------------------------------------------------------------------------------


def fill_template(
    template: Union[PlanTemplate, Mapping[str, Any]],
    slot_values: Mapping[str, Any],
) -> Dict[str, Any]:
    """Bind ``slot_values`` into ``template``.

    :returns: ``compiled_plan.py``'s ``{"leaves": [...], "aggregation": "..."}`` shape exactly.
    :raises TemplateValidationError: on an invalid template, a missing required slot, an
        unresolvable reference, a bad arity, or a filled plan compiled-plan rejects.
    """
    return bind_template(template, slot_values).to_plan()


def bind_template(
    template: Union[PlanTemplate, Mapping[str, Any]],
    slot_values: Mapping[str, Any],
) -> FilledPlan:
    """Like :func:`fill_template`, but keeps the structured bindings (``for_each`` item and the
    derived search query) each leaf was filled from — what the GoT adapter needs."""
    tpl = validate_template(template)
    if not isinstance(slot_values, Mapping):
        raise TemplateValidationError(f"slot_values must be a mapping, got {type(slot_values).__name__}")
    values = _resolve_slot_values(tpl, slot_values)

    # Pass 1: materialize every instance (ids + filled text), remembering its binding context.
    produced: Dict[str, List[str]] = {leaf.id_pattern: [] for leaf in tpl.leaves}
    staged: List[Tuple[LeafBlueprint, Dict[str, Any], int, str, str, str, str]] = []
    for blueprint in tpl.leaves:
        for index, item in enumerate(_items_for(tpl, blueprint, values)):
            ctx = {ITEM_REF: item} if item is not None else {}
            where = f"template '{tpl.template_id}' leaf '{blueprint.id_pattern}'"
            leaf_id = _fill_text(blueprint.id_pattern, values, ctx, where=where)
            if not leaf_id.strip():
                raise TemplateValidationError(f"{where}: id_pattern filled to an empty leaf id")
            instruction = _fill_text(blueprint.instruction, values, ctx, where=where)
            expect = _fill_text(blueprint.expect, values, ctx, where=where)
            query = _build_query(tpl, values, item, instruction)
            produced[blueprint.id_pattern].append(leaf_id)
            staged.append((blueprint, ctx, index, leaf_id, instruction, expect, query))

    # Pass 2: resolve blueprint-level dependencies down to concrete leaf ids.
    leaves: List[FilledLeaf] = []
    for blueprint, ctx, index, leaf_id, instruction, expect, query in staged:
        deps = _resolve_depends_on(tpl, blueprint, ctx, index, values, produced)
        leaves.append(
            FilledLeaf(
                id=leaf_id,
                instruction=instruction,
                expect=expect,
                depends_on=deps,
                query=query,
                blueprint_id=blueprint.id_pattern,
                item=ctx.get(ITEM_REF),
            )
        )

    aggregation = _fill_text(
        tpl.aggregation, values, {}, where=f"template '{tpl.template_id}' aggregation"
    )
    filled = FilledPlan(template_id=tpl.template_id, leaves=leaves, aggregation=aggregation)

    # Final guarantee: the filled plan is a legal compiled plan (unique ids, resolvable deps,
    # acyclic) — the same gate execution_compiled.py applies to a hand-authored plan.
    try:
        compiled_plan.validate_plan(filled.to_plan())
    except compiled_plan.PlanValidationError as exc:
        raise TemplateValidationError(f"template '{tpl.template_id}' filled to an invalid plan: {exc}") from exc
    return filled


def _resolve_slot_values(template: PlanTemplate, slot_values: Mapping[str, Any]) -> Dict[str, Any]:
    """Bind declared slots to caller-supplied values, applying defaults and arity checks."""
    resolved: Dict[str, Any] = {}
    for spec in template.slots:
        value = slot_values.get(spec.name)
        if value is None or (isinstance(value, str) and not value.strip()):
            if spec.default is not None:
                value = spec.default
            elif spec.required:
                raise TemplateValidationError(
                    f"template '{template.template_id}': required slot '{spec.name}' has no value"
                )
            else:
                value = ""
        if spec.kind in LIST_SLOT_KINDS:
            if not isinstance(value, (list, tuple)):
                raise TemplateValidationError(
                    f"template '{template.template_id}': slot '{spec.name}' (kind "
                    f"'{spec.kind.value}') expects a list, got {type(value).__name__}"
                )
            value = list(value)
            if spec.min_arity is not None and len(value) < spec.min_arity:
                raise TemplateValidationError(
                    f"template '{template.template_id}': slot '{spec.name}' needs at least "
                    f"{spec.min_arity} items, got {len(value)}"
                )
            if spec.max_arity is not None and len(value) > spec.max_arity:
                value = value[: spec.max_arity]
        resolved[spec.name] = value
    return resolved


def _items_for(
    template: PlanTemplate,
    blueprint: LeafBlueprint,
    values: Mapping[str, Any],
) -> List[Optional[Dict[str, Any]]]:
    """The bound items a blueprint expands over — ``[None]`` when it does not repeat."""
    if not blueprint.for_each:
        return [None]
    raw_items = values.get(blueprint.for_each) or []
    where = f"template '{template.template_id}' leaf '{blueprint.id_pattern}'"
    if not raw_items:
        raise TemplateValidationError(f"{where}: for_each slot '{blueprint.for_each}' is empty")
    return [_normalize_item(raw, index, where) for index, raw in enumerate(raw_items)]


def _normalize_item(raw: Any, index: int, where: str) -> Dict[str, Any]:
    """Normalize one list-slot element, injecting the synthetic ``key``/``index``/``ordinal``
    attributes an ``id_pattern`` needs to stay unique."""
    if isinstance(raw, str):
        item: Dict[str, Any] = {"name": raw.strip()}
    elif isinstance(raw, Mapping):
        item = dict(raw)
    else:
        raise TemplateValidationError(
            f"{where}: for_each item #{index} must be a string or mapping, got {type(raw).__name__}"
        )
    name = str(item.get("name") or item.get("label") or item.get("entity_hint") or "").strip()
    item.setdefault("name", name)
    item["index"] = index
    item["ordinal"] = index + 1
    key = str(item.get("key") or "").strip() or _slug(name) or f"item_{index}"
    item["key"] = key
    return item


def _resolve_depends_on(
    template: PlanTemplate,
    blueprint: LeafBlueprint,
    ctx: Mapping[str, Any],
    index: int,
    values: Mapping[str, Any],
    produced: Mapping[str, List[str]],
) -> List[str]:
    """Map a blueprint's ``depends_on`` patterns onto concrete leaf ids.

    Three cases, all keyed off the target blueprint (``depends_on`` entries are blueprint
    ``id_pattern``s, checked by :func:`validate_template`):

      * **self-edge on a ``for_each`` blueprint** -> the *previous* instance (a chain);
      * **another blueprint repeating over the SAME list slot** -> the same-index instance;
      * **anything else** -> every instance the target produced (fan-in).
    """
    by_pattern = {leaf.id_pattern: leaf for leaf in template.leaves}
    where = f"template '{template.template_id}' leaf '{blueprint.id_pattern}'"
    deps: List[str] = []
    for dep in blueprint.depends_on:
        target = by_pattern[dep]
        if target is blueprint:
            if index > 0:
                deps.append(produced[dep][index - 1])
            continue
        if target.for_each and blueprint.for_each and target.for_each == blueprint.for_each:
            dep_id = _fill_text(dep, values, ctx, where=where)
            if dep_id not in produced[dep]:
                raise TemplateValidationError(
                    f"{where}: depends_on '{dep}' filled to unknown leaf id '{dep_id}'"
                )
            deps.append(dep_id)
            continue
        deps.extend(produced[dep])
    return list(dict.fromkeys(deps))


def _build_query(
    template: PlanTemplate,
    values: Mapping[str, Any],
    item: Optional[Mapping[str, Any]],
    instruction: str,
) -> str:
    """Build a leaf's web-search query from the bound slot VALUES, never from the filled prose.

    ``SlotKind`` makes this deterministic: the subject is the bound ``for_each`` item (or the
    template's ENTITY slot), the field is the item's own ``field`` (HOP_LIST) or the template's
    FIELD slot. This is why the plan library does not need ``execution_compiled.py``'s
    ``_THIN_QUERY_SYS`` LLM call to derive a query from a compiled leaf's free text.
    """
    if item is not None:
        subject = str(item.get("name") or "")
        qualifier = str(item.get("qualifier") or "")
        field_text = str(item.get("field") or "")
    else:
        subject = _first_slot_value(template, values, SlotKind.ENTITY)
        qualifier = ""
        field_text = ""
    if not field_text.strip():
        field_text = _first_slot_value(template, values, SlotKind.FIELD)
    query = _compact(" ".join(p for p in (subject, qualifier, field_text) if p.strip()))
    return query or _compact(instruction)


def _first_slot_value(template: PlanTemplate, values: Mapping[str, Any], kind: SlotKind) -> str:
    for spec in template.slots:
        if spec.kind == kind:
            return _render(values.get(spec.name))
    return ""


# --------------------------------------------------------------------------------------
# placeholder plumbing
# --------------------------------------------------------------------------------------


def _refs(text: str) -> List[str]:
    """Every ``<<ref>>`` path in ``text``, in order of appearance."""
    return _PLACEHOLDER_RE.findall(text or "")


def _check_refs(text: str, allowed_roots: set, where: str) -> None:
    for ref in _refs(text):
        root = ref.split(".")[0]
        if root not in allowed_roots:
            raise TemplateValidationError(
                f"{where}: placeholder '<<{ref}>>' references undeclared slot '{root}' "
                f"(declared: {', '.join(sorted(allowed_roots)) or 'none'})"
            )


def _fill_text(text: str, values: Mapping[str, Any], ctx: Mapping[str, Any], where: str) -> str:
    """Substitute every ``<<ref>>`` in ``text``. Non-``<<>>`` braces (``{dep_id}`` runtime
    placeholders owned by ``compiled_plan.substitute_deps``) are left untouched."""
    if not text:
        return ""

    def _sub(match: "re.Match[str]") -> str:
        return _render(_resolve_ref(match.group(1), values, ctx, where))

    return _PLACEHOLDER_RE.sub(_sub, text)


def _resolve_ref(ref: str, values: Mapping[str, Any], ctx: Mapping[str, Any], where: str) -> Any:
    head, *rest = ref.split(".")
    if head == ITEM_REF:
        if ITEM_REF not in ctx:
            raise TemplateValidationError(
                f"{where}: '<<{ref}>>' used outside a for_each blueprint"
            )
        current: Any = ctx[ITEM_REF]
    elif head in values:
        current = values[head]
    else:
        raise TemplateValidationError(f"{where}: placeholder '<<{ref}>>' references undeclared slot '{head}'")

    for attr in rest:
        if isinstance(current, Mapping) and attr in current:
            current = current[attr]
        else:
            raise TemplateValidationError(f"{where}: placeholder '<<{ref}>>' has no attribute '{attr}'")
    return current


def _render(value: Any) -> str:
    """Render a slot value as template text (lists become readable enumerations)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("name", "label", "value"):
            if value.get(key) is not None:
                return str(value[key])
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(_render(v) for v in value if v is not None)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _compact(text: str) -> str:
    """Collapse whitespace and clip to Brave's query ceiling on a word boundary.

    ``connector_search._sanitize_brave_query`` is the real enforcement point; clipping here
    keeps the query readable in logs and telemetry rather than silently truncated later.

    Duplicate-constant hazard (2026-08-08): the ``400`` below is a second, independent copy of
    Brave's char limit — now that Serper is the default provider (whose real limit is unverified,
    see ``connector_search_serper.py``'s module docstring) this clip may be tighter than the
    active provider actually needs. Left as-is (harmless either way: it only affects log/telemetry
    readability, not the real request), but if Serper's limit ever needs enforcing for real,
    fix the duplication rather than adding a third copy.
    """
    out = " ".join(str(text or "").split())
    if len(out) > 400:
        clipped = out[:400]
        out = clipped.rsplit(" ", 1)[0] if " " in clipped else clipped
    return out.strip()


def _slug(text: str) -> str:
    """Lowercase underscore slug from free text (first ~6 words) — mirrors ``compiled_plan._slug``."""
    words = re.sub(r"[^a-z0-9\s]+", " ", str(text or "").lower()).split()
    return "_".join(words[:6]).strip("_")


def _as_list(value: Any, field_name: str) -> Sequence[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TemplateValidationError(f"template '{field_name}' must be a list, got {type(value).__name__}")
    return value


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_opt_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_enum(enum_cls: Any, value: Any, where: str, field_name: str) -> Any:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_cls)
        raise TemplateValidationError(
            f"{where}: unknown {field_name} '{value}' (expected one of: {allowed})"
        ) from exc
