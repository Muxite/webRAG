"""
Offline unit tests for plan-library slot filling — free, no LLM, no Chroma, no network.

Slot filling is the piece that makes retrieval more than a lookup: it turns a live query into
the concrete ``{slot: value}`` mapping ``schema.fill_template`` needs, so "this looks like an
``argmax_over_n_page_field`` task" becomes "...over THESE six peaks, reading THIS field".

Each :class:`ExtractionStrategy` is pinned in isolation:

* ``REGEX_CANDIDATE_LIST`` against the REAL task-062 statement (the corpus this generalizes),
  including the fail-open cases ``candidate_coverage`` already guarantees;
* ``REGEX_SHAPE_HINT`` over the deterministic classifiers, plus the fact that NO seed template
  declares it (so the extractor exists for dispatch completeness, not for a live path);
* ``LLM`` through a canned-completion stub (``expansion_json_telemetry_test``'s ``_StubIO``
  pattern), pinning the property that actually costs money: ONE call per fill attempt, for
  every LLM slot at once, and never speculatively;
* ``DERIVED`` through its hook registry — also unused by all six seed templates;
* ``DEFAULT``, which ``schema._resolve_slot_values`` already applies, so the test pins that the
  echoed value is the SAME one the filled plan uses rather than a second source of truth.

Then the end-to-end path this whole layer exists for: real template + real task statement ->
``extract_slot_values`` -> ``fill_template``/``bind_template`` -> a valid filled plan, and
through ``PlanLibrary.fill_from_query`` -> real GoT candidates.
"""
from __future__ import annotations

import json

import pytest

from agent.app.idea_policies import plan_library as adapter
from agent.app.idea_policies.base import DetailKey
from agent.app.idea_tests import test_051_tier4_dependent_chain as t051
from agent.app.idea_tests import test_062_tier5_prominence_argmax as t062
from agent.app.plan_library import retrieval as R
from agent.app.plan_library import slot_fill as SF
from agent.app.plan_library.schema import (
    ExtractionStrategy,
    LeafBlueprint,
    PlanTemplate,
    SlotKind,
    SlotSpec,
    TemplateValidationError,
    bind_template,
    fill_template,
)
from agent.app.testing import compiled_plan

MANDATE_062 = t062.get_task_statement()
MANDATE_051 = t051.get_task_statement()

PEAKS = [
    "Jengish Chokusu", "Mount Gongga", "Kongur Tagh",
    "Ismoil Somoni Peak", "Muztagh Ata", "Noshaq",
]


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------


class StubIO:
    """Minimal ``AgentIO`` stand-in: a canned completion, no connector, no network."""

    def __init__(self, response: str = "{}", *, raises: Exception = None):
        self.response = response
        self.raises = raises
        self.built = []      # every build_llm_payload kwargs
        self.payloads = []   # every payload actually sent

    def build_llm_payload(self, **kwargs):
        self.built.append(kwargs)
        return dict(kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs):
        self.payloads.append((payload, kwargs))
        if self.raises is not None:
            raise self.raises
        return self.response

    # -- assertions helpers -------------------------------------------------------------

    @property
    def calls(self) -> int:
        return len(self.payloads)

    def system(self, index: int = 0) -> str:
        return self.payloads[index][0]["messages"][0]["content"]

    def user(self, index: int = 0) -> str:
        return self.payloads[index][0]["messages"][1]["content"]


class FakeChroma:
    """A ``ConnectorChroma``-shaped stand-in, ranking whatever it was handed."""

    def __init__(self, hits=()):
        self.hits = list(hits)

    async def get_or_create_collection(self, collection, metadata=None):
        return None

    async def query_chroma(self, collection, query_texts, n_results=3, where=None):
        hits = self.hits[:n_results]
        return {
            "ids": [[h[0] for h in hits]],
            "distances": [[h[1] for h in hits]],
            "metadatas": [[{"template_id": h[0], "archetype": h[2]} for h in hits]],
        }


def _shipped(template_id: str) -> PlanTemplate:
    template = R.PlanLibrary(warn_on_drift=False).get(template_id)
    assert template is not None, template_id
    return template


#: What a competent model returns for task 062's three LLM-strategy slots.
LLM_062 = json.dumps(
    {
        "field": "TOPOGRAPHIC PROMINENCE in metres",
        "entity_noun": "mountain",
        "superlative": "HIGHEST",
    }
)


# --------------------------------------------------------------------------------------
# REGEX_CANDIDATE_LIST
# --------------------------------------------------------------------------------------


def test_candidate_list_extracts_every_named_candidate_from_the_real_062_statement():
    spec = SlotSpec(name="candidates", kind=SlotKind.ENTITY_LIST)
    items = SF.candidate_list_value(spec, MANDATE_062)

    assert [item["name"] for item in items] == PEAKS
    # The shape schema._normalize_item declares: a name-bearing mapping. ``key`` is left to
    # _normalize_item so leaf ids have exactly one source of truth.
    assert all(set(item) == {"name"} for item in items)


def test_candidate_list_output_binds_without_any_renormalization():
    """The point of the shape contract: bind_template accepts it as-is."""
    spec = SlotSpec(name="candidates", kind=SlotKind.ENTITY_LIST)
    items = SF.candidate_list_value(spec, MANDATE_062)
    filled = bind_template(_shipped("argmax_over_n_page_field"), {"candidates": items, "field": "x"})
    assert [leaf.item["name"] for leaf in filled.leaves] == PEAKS
    assert filled.leaves[0].id == "jengish_chokusu_field"  # key slugged by _normalize_item


def test_candidate_list_fails_open_on_an_instruction_list():
    """Task 051's numbered steps are INSTRUCTIONS, not candidates — extracting them would
    fabricate entities. ``candidate_coverage`` already refuses; this pins that it still does."""
    spec = SlotSpec(name="candidates", kind=SlotKind.ENTITY_LIST)
    assert SF.candidate_list_value(spec, MANDATE_051) is None
    assert SF.candidate_list_value(spec, "which of these is deepest?") is None
    assert SF.candidate_list_value(spec, "") is None


def test_candidate_list_only_fills_entity_list_slots():
    """A HOP_LIST element also needs a per-hop ``field`` a flat enumeration cannot supply, so
    filling one from this extractor would only produce a template that fails to bind."""
    for kind in (SlotKind.HOP_LIST, SlotKind.FIELD, SlotKind.ENTITY):
        assert SF.candidate_list_value(SlotSpec(name="s", kind=kind), MANDATE_062) is None


def test_candidate_list_dedupes_repeated_names():
    mandate = "Compare these:\n1. Lake Baikal\n2. Lake Tanganyika\n3. lake baikal\n"
    items = SF.candidate_list_value(SlotSpec(name="c", kind=SlotKind.ENTITY_LIST), mandate)
    assert [item["name"] for item in items] == ["Lake Baikal", "Lake Tanganyika"]


# --------------------------------------------------------------------------------------
# REGEX_SHAPE_HINT
# --------------------------------------------------------------------------------------


def _agg_slot() -> SlotSpec:
    return SlotSpec(
        name="op", kind=SlotKind.AGGREGATION_OP, extraction=ExtractionStrategy.REGEX_SHAPE_HINT
    )


def test_shape_hint_derives_the_operator_direction_from_the_mandate():
    assert SF.shape_hint_value(_agg_slot(), MANDATE_062) == "argmax"
    assert (
        SF.shape_hint_value(_agg_slot(), "Which of these five rivers has the LOWEST elevation?")
        == "argmin"
    )
    assert (
        SF.shape_hint_value(_agg_slot(), "Which of these five islands is the smallest?") == "argmin"
    )
    assert (
        SF.shape_hint_value(_agg_slot(), "How many of these seven lakes exceed 500 metres?")
        == "count_where"
    )


def test_shape_hint_fails_open_on_an_unrecognised_mandate():
    assert SF.shape_hint_value(_agg_slot(), "Write an essay about lakes.") is None
    assert SF.shape_hint_value(_agg_slot(), "   ") is None


def test_shape_hint_on_a_non_operator_slot_is_only_the_archetype_label():
    """The classifier knows a mandate's SHAPE, not its parameters — for anything but an
    operator slot it can offer no more than the label retrieval already reranks with."""
    spec = SlotSpec(name="x", kind=SlotKind.FIELD, extraction=ExtractionStrategy.REGEX_SHAPE_HINT)
    assert SF.shape_hint_value(spec, MANDATE_051) == "chain"
    # 062 is a six-mountain fan-out: the label is "breadth" (it classified as None until
    # the shape classifier learned that shape), still just the label and no parameters.
    assert SF.shape_hint_value(spec, MANDATE_062) == "breadth"
    assert SF.shape_hint_value(spec, "Write an essay about lakes.") is None


def test_no_seed_template_declares_the_shape_hint_or_derived_strategies():
    """Documents the finding: both strategies are dispatch completeness for future/mined
    templates. If a seed template ever adopts one, this test is the reminder to cover it."""
    declared = {
        (template.template_id, slot.name): slot.extraction
        for template in R.PlanLibrary(warn_on_drift=False).templates.values()
        for slot in template.slots
    }
    assert declared, "the shipped corpus must not be empty"
    assert ExtractionStrategy.REGEX_SHAPE_HINT not in declared.values()
    assert ExtractionStrategy.DERIVED not in declared.values()
    assert set(declared.values()) == {
        ExtractionStrategy.REGEX_CANDIDATE_LIST,
        ExtractionStrategy.LLM,
        ExtractionStrategy.DEFAULT,
    }


# --------------------------------------------------------------------------------------
# DEFAULT — already applied by the schema; echoed here so the mapping is self-describing
# --------------------------------------------------------------------------------------


def test_default_slot_is_echoed_and_matches_what_the_schema_would_apply():
    template = _shipped("argmax_over_n_page_field")
    values = SF.deterministic_slot_values(template, MANDATE_062)
    assert values["source_type"] == "the authoritative Wikipedia page"

    complete = fill_template(template, {**values, "field": "prominence"})
    without = fill_template(
        template, {k: v for k, v in values.items() if k != "source_type"} | {"field": "prominence"}
    )
    assert complete == without  # echoing it changes nothing; it only makes the log honest


# --------------------------------------------------------------------------------------
# LLM — ONE batched, schema-driven call
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_llm_call_covers_every_llm_slot_of_the_template():
    io = StubIO(LLM_062)
    values = await SF.extract_slot_values(_shipped("argmax_over_n_page_field"), MANDATE_062, MANDATE_062, io)

    assert io.calls == 1  # batched, not one call per slot
    assert values["field"] == "TOPOGRAPHIC PROMINENCE in metres"
    assert values["entity_noun"] == "mountain"
    assert values["superlative"] == "HIGHEST"
    # the regex- and default-filled slots never went to the model
    assert '- "field"' in io.user() and '- "candidates"' not in io.user()
    assert '- "source_type"' not in io.user()


@pytest.mark.asyncio
async def test_the_llm_prompt_uses_each_slot_description_verbatim_and_the_task_text():
    template = _shipped("argmax_over_n_page_field")
    io = StubIO(LLM_062)
    await SF.extract_slot_values(template, "compare six peaks", MANDATE_062, io)

    user = io.user()
    for name in ("field", "entity_noun", "superlative"):
        assert template.slot(name).description in user
    assert "Jengish Chokusu (Kyrgyzstan / China)" in user   # the raw statement, not a summary
    assert "compare six peaks" in user                       # the node's own goal, when it differs
    assert template.template_id in user


@pytest.mark.asyncio
async def test_the_call_follows_the_json_mode_convention_and_carries_the_schema_hint():
    io = StubIO(LLM_062)
    await SF.extract_slot_values(_shipped("argmax_over_n_page_field"), MANDATE_062, MANDATE_062, io)

    kwargs = io.built[0]
    assert kwargs["json_mode"] is True
    # A template may declare OPTIONAL slots, which strict structured output forbids (``required``
    # must enumerate every property) — so the schema rides as a text hint, exactly like expansion.
    assert kwargs["json_schema"] is None
    assert kwargs["temperature"] == SF.DEFAULT_TEMPERATURE
    assert "conforms to this JSON Schema" in io.system()
    assert '"superlative"' in io.system()


def test_the_json_schema_is_built_from_the_slot_specs():
    template = _shipped("argmax_over_n_page_field")
    schema = SF.build_slot_schema(template, template.slots)["schema"]

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"candidates", "field"}   # exactly the required slots
    candidates = schema["properties"]["candidates"]
    assert candidates["type"] == "array"
    assert candidates["minItems"] == 2 and candidates["maxItems"] == 12
    assert candidates["items"]["required"] == ["name"]
    assert candidates["description"] == template.slot("candidates").description
    assert schema["properties"]["field"]["type"] == "string"


def test_the_json_schema_describes_a_hop_list_as_label_plus_field():
    template = _shipped("entity_chain_resolution")
    hops = SF.build_slot_schema(template, template.slots)["schema"]["properties"]["hops"]
    assert hops["type"] == "array"
    assert sorted(hops["items"]["required"]) == ["field", "label"]
    assert hops["minItems"] == 1 and hops["maxItems"] == 4


@pytest.mark.asyncio
async def test_a_fenced_or_truncated_completion_is_repaired_not_discarded():
    """A weak model fences its JSON here exactly as it does when planning, so the same
    ``expansion._repair_json_object`` fallback applies."""
    io = StubIO("```json\n" + LLM_062 + "\n```")
    values = await SF.extract_slot_values(_shipped("argmax_over_n_page_field"), MANDATE_062, MANDATE_062, io)
    assert values["field"] == "TOPOGRAPHIC PROMINENCE in metres"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "io",
    [
        StubIO(raises=RuntimeError("model is down")),
        StubIO("I'm sorry, I cannot help with that."),
        StubIO(""),
        None,  # no LLM available at all
    ],
)
async def test_an_unusable_llm_pass_raises_for_the_missing_required_slot(io):
    """Fail toward silence: the caller turns this into "no match", never a half-filled plan."""
    with pytest.raises(TemplateValidationError) as exc:
        await SF.extract_slot_values(_shipped("argmax_over_n_page_field"), MANDATE_062, MANDATE_062, io)
    assert "field" in str(exc.value)


@pytest.mark.asyncio
async def test_optional_slots_are_left_to_their_defaults_when_the_model_omits_them():
    io = StubIO(json.dumps({"field": "maximum depth in metres", "superlative": None}))
    values = await SF.extract_slot_values(_shipped("argmax_over_n_page_field"), MANDATE_062, MANDATE_062, io)
    assert "superlative" not in values and "entity_noun" not in values
    plan = fill_template(_shipped("argmax_over_n_page_field"), values)
    assert "HIGHEST" in plan["aggregation"]  # the declared default, applied by the schema


@pytest.mark.asyncio
async def test_no_llm_call_at_all_when_every_slot_is_deterministic():
    """The one call is per FILL ATTEMPT, and only when something actually needs a model."""
    template = PlanTemplate(
        template_id="all_deterministic",
        slots=[
            SlotSpec(
                name="candidates", kind=SlotKind.ENTITY_LIST,
                extraction=ExtractionStrategy.REGEX_CANDIDATE_LIST, min_arity=2,
            ),
            SlotSpec(
                name="source_type", kind=SlotKind.SOURCE_TYPE, required=False,
                extraction=ExtractionStrategy.DEFAULT, default="the Wikipedia page",
            ),
        ],
        leaves=[
            LeafBlueprint(
                id_pattern="<<item.key>>_leaf", for_each="candidates",
                instruction="Open <<source_type>> for <<item.name>>.",
                expect="<<item.name>>: the value -- source URL",
            )
        ],
        aggregation="Compare them all.",
    )
    io = StubIO(LLM_062)
    values = await SF.extract_slot_values(template, MANDATE_062, MANDATE_062, io)
    assert io.calls == 0 and io.built == []
    assert len(values["candidates"]) == 6


@pytest.mark.asyncio
async def test_a_required_regex_slot_falls_back_to_the_llm_when_the_list_is_inline_prose():
    """The real-world gap: ``extract_named_candidates`` only parses ENUMERATED lists, and
    correctly returns nothing for an inline "A, B and C" phrasing. The batched call is already
    being made for the other slots, so a required hole rides along instead of sinking the match."""
    mandate = (
        "Compare the topographic prominence of these three peaks: Muztagh Ata, Jengish Chokusu "
        "and Noshaq, and report which is highest."
    )
    io = StubIO(
        json.dumps(
            {
                "candidates": [
                    {"name": "Muztagh Ata", "qualifier": "China"},
                    "Jengish Chokusu",
                    {"name": "Noshaq"},
                ],
                "field": "topographic prominence in metres",
            }
        )
    )
    values = await SF.extract_slot_values(_shipped("argmax_over_n_page_field"), mandate, mandate, io)

    assert io.calls == 1
    assert '- "candidates"' in io.user()   # it rode along in the same request
    assert [c["name"] for c in values["candidates"]] == [
        "Muztagh Ata", "Jengish Chokusu", "Noshaq",
    ]
    assert values["candidates"][0]["qualifier"] == "China"
    plan = fill_template(_shipped("argmax_over_n_page_field"), values)
    assert len(plan["leaves"]) == 3


@pytest.mark.asyncio
async def test_a_bare_string_where_a_list_was_asked_for_is_coerced_not_rejected():
    """Tolerate a weak model's shape slips: rejecting the match on a formatting slip throws
    away a correct retrieval."""
    io = StubIO(json.dumps({"hops": "the poet's birth town", "seed": "the 1924 collection",
                            "seed_field": "the poet who wrote it"}))
    values = await SF.extract_slot_values(_shipped("entity_chain_resolution"), MANDATE_051, MANDATE_051, io)
    assert values["hops"] == [{"label": "the poet's birth town"}]


# --------------------------------------------------------------------------------------
# DERIVED — a per-template hook registry (empty for the seed corpus)
# --------------------------------------------------------------------------------------


def _ratio_template() -> PlanTemplate:
    return PlanTemplate(
        template_id="derived_probe",
        slots=[
            SlotSpec(name="numerator_field", kind=SlotKind.FIELD),
            SlotSpec(name="denominator_field", kind=SlotKind.FIELD),
            SlotSpec(
                name="fields", kind=SlotKind.FIELD, required=False,
                extraction=ExtractionStrategy.DERIVED,
                default="both figures",
            ),
        ],
        leaves=[
            LeafBlueprint(
                id_pattern="read", instruction="Read <<fields>>.",
                expect="<<numerator_field>> and <<denominator_field>> -- source URL",
            )
        ],
        aggregation="Divide <<numerator_field>> by <<denominator_field>>.",
    )


@pytest.fixture
def derived_hook():
    def _join(values, template):
        return f"{values['numerator_field']} and {values['denominator_field']}"

    SF.register_derived("derived_probe", "fields", _join)
    yield _join
    SF.unregister_derived("derived_probe", "fields")


@pytest.mark.asyncio
async def test_a_derived_slot_is_computed_from_the_slots_already_filled(derived_hook):
    io = StubIO(json.dumps({"numerator_field": "water volume (km^3)",
                            "denominator_field": "surface area (km^2)"}))
    values = await SF.extract_slot_values(_ratio_template(), "a ratio task", "a ratio task", io)

    assert values["fields"] == "water volume (km^3) and surface area (km^2)"
    # and it was NOT asked of the model: derived slots are computed from that call's output
    assert '- "fields"' not in io.user()
    assert "Read water volume (km^3) and surface area (km^2)." == fill_template(
        _ratio_template(), values
    )["leaves"][0]["instruction"]


@pytest.mark.asyncio
async def test_an_unregistered_or_failing_derived_slot_falls_through_to_its_default():
    io = StubIO(json.dumps({"numerator_field": "a", "denominator_field": "b"}))
    values = await SF.extract_slot_values(_ratio_template(), "q", "q", io)
    assert "fields" not in values
    assert "Read both figures." == fill_template(_ratio_template(), values)["leaves"][0]["instruction"]

    def _boom(values, template):
        raise RuntimeError("hook is broken")

    SF.register_derived("derived_probe", "fields", _boom)
    try:
        values = await SF.extract_slot_values(_ratio_template(), "q", "q", StubIO(io.response))
        assert "fields" not in values  # swallowed, never fatal
    finally:
        SF.unregister_derived("derived_probe", "fields")


# --------------------------------------------------------------------------------------
# end to end — the point of the whole layer
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_real_template_real_statement_produces_a_valid_filled_plan():
    template = _shipped("argmax_over_n_page_field")
    io = StubIO(LLM_062)

    values = await SF.extract_slot_values(template, MANDATE_062, MANDATE_062, io)
    plan = fill_template(template, values)

    compiled_plan.validate_plan(plan)                       # a legal compiled plan, unchanged
    assert [leaf["id"] for leaf in plan["leaves"]] == [
        "jengish_chokusu_field", "mount_gongga_field", "kongur_tagh_field",
        "ismoil_somoni_peak_field", "muztagh_ata_field", "noshaq_field",
    ]
    blob = json.dumps(plan)
    assert "<<" not in blob and ">>" not in blob           # every placeholder was bound
    for peak in PEAKS:
        assert peak in blob
    first = plan["leaves"][0]
    assert first["depends_on"] == []                       # independent per-candidate reads
    assert "the authoritative Wikipedia page" in first["instruction"]   # DEFAULT slot
    assert "TOPOGRAPHIC PROMINENCE in metres" in first["instruction"]   # LLM slot
    assert first["expect"].startswith("Jengish Chokusu:")               # REGEX slot, per leaf
    assert "HIGHEST" in plan["aggregation"] and "mountain" in plan["aggregation"]


@pytest.mark.asyncio
async def test_end_to_end_through_the_adapter_yields_native_got_candidates():
    template = _shipped("argmax_over_n_page_field")
    values = await SF.extract_slot_values(template, MANDATE_062, MANDATE_062, StubIO(LLM_062))

    expansion = adapter.candidates_from_template(template, values)

    assert len(expansion.candidates) == 6
    details = expansion.candidates[0]["details"]
    assert details[DetailKey.ACTION.value] == "search"
    # the query comes from the bound slot VALUES, never re-parsed out of the filled prose
    assert details[DetailKey.QUERY.value] == "Jengish Chokusu TOPOGRAPHIC PROMINENCE in metres"
    assert "Jengish Chokusu" in details[DetailKey.EXPECT.value]
    assert expansion.parent_details[DetailKey.INTENT.value] == expansion.aggregation


@pytest.mark.asyncio
async def test_end_to_end_a_chain_template_fills_its_hops_and_keeps_the_dependency_order():
    template = _shipped("entity_chain_resolution")
    io = StubIO(
        json.dumps(
            {
                "seed": "the novel 'Things Fall Apart'",
                "seed_field": "the AUTHOR of the novel -- their full name",
                "hops": [
                    {"label": "that author", "field": "the UNIVERSITY they attended"},
                    {"label": "that university", "field": "the YEAR it was founded"},
                ],
            }
        )
    )
    values = await SF.extract_slot_values(template, MANDATE_051, MANDATE_051, io)
    filled = bind_template(template, values)

    assert [leaf.id for leaf in filled.leaves] == [
        "chain_start", "that_author_hop", "that_university_hop",
    ]
    assert filled.leaves[1].depends_on == ["chain_start"]
    assert filled.leaves[2].depends_on == ["chain_start", "that_author_hop"]
    assert "the YEAR it was founded" in filled.leaves[2].instruction


# --------------------------------------------------------------------------------------
# PlanLibrary.fill_from_query — retrieve() then fill_from_query() is the whole caller API
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_from_query_extracts_and_fills_in_one_step():
    library = R.PlanLibrary(warn_on_drift=False)
    chroma = FakeChroma([("argmax_over_n_page_field", 0.30, "argmax"),
                         ("bounded_subset_sum", 0.70, "subset_sum")])
    io = StubIO(LLM_062)

    result = await library.retrieve(chroma, MANDATE_062)
    assert result.decision == R.DECISION_AUTO_APPLY
    outcome = await library.fill_from_query(result, mandate=MANDATE_062, io=io)

    assert outcome.ok and io.calls == 1
    assert len(outcome.expansion.candidates) == 6
    assert outcome.expansion.template_id == "argmax_over_n_page_field"
    # the values are carried so the retrieval log can record what was actually bound
    assert [c["name"] for c in outcome.slot_values["candidates"]] == PEAKS
    assert outcome.slot_values["field"] == "TOPOGRAPHIC PROMINENCE in metres"


@pytest.mark.asyncio
async def test_fill_from_query_downgrades_to_no_match_when_a_slot_cannot_be_extracted():
    library = R.PlanLibrary(warn_on_drift=False)
    chroma = FakeChroma([("argmax_over_n_page_field", 0.30, "argmax")])
    result = await library.retrieve(chroma, MANDATE_062)

    outcome = await library.fill_from_query(result, mandate=MANDATE_062, io=None)

    assert not outcome.ok and outcome.expansion is None
    assert outcome.retrieval.decision == R.DECISION_NO_MATCH
    assert outcome.retrieval.template_id is None
    assert "slot extraction failed" in outcome.retrieval.reason
    assert outcome.retrieval.candidates                       # ranking kept for the log
    assert outcome.retrieval.retrieval_id == result.retrieval_id


@pytest.mark.asyncio
async def test_fill_from_query_never_calls_the_model_without_a_qualifying_match():
    """The cost property: no similarity-qualifying hit, no LLM traffic at all."""
    library = R.PlanLibrary(warn_on_drift=False)
    chroma = FakeChroma([("argmax_over_n_page_field", 0.99, "argmax")])   # far below threshold
    io = StubIO(LLM_062)

    result = await library.retrieve(chroma, "what is the capital of France")
    assert result.decision == R.DECISION_NO_MATCH

    outcome = await library.fill_from_query(result, mandate="what is the capital of France", io=io)
    assert not outcome.ok and io.calls == 0 and io.built == []


@pytest.mark.asyncio
async def test_fill_from_query_on_an_unknown_template_never_reaches_slot_fill():
    library = R.PlanLibrary(warn_on_drift=False)
    io = StubIO(LLM_062)
    ghost = R.RetrievalResult(
        retrieval_id="x" * 32, query_text="q", decision=R.DECISION_AUTO_APPLY,
        template_id="not_on_disk", similarity=0.9,
    )
    outcome = await library.fill_from_query(ghost, mandate=MANDATE_062, io=io)
    assert not outcome.ok and io.calls == 0
    assert "not in the library" in outcome.retrieval.reason


@pytest.mark.asyncio
async def test_without_a_mandate_the_candidate_list_falls_back_to_the_llm():
    """A caller that passes no mandate still fills — but pays for it, which is why the engine
    should pass the RAW statement.

    ``retrieve()`` whitespace-collapses its query text, so the fallback text has no line
    structure left and ``extract_named_candidates`` (deliberately line-oriented) sees no
    enumeration. The required slot then rides along in the LLM request instead of being free.
    """
    library = R.PlanLibrary(warn_on_drift=False)
    chroma = FakeChroma([("argmax_over_n_page_field", 0.30, "argmax")])
    result = await library.retrieve(chroma, MANDATE_062)
    io = StubIO(json.dumps({"field": "topographic prominence in metres",
                            "candidates": [{"name": p} for p in PEAKS]}))

    outcome = await library.fill_from_query(result, io=io)

    assert outcome.ok
    assert '- "candidates"' in io.user()   # not free any more: the model had to supply them
    assert [c["name"] for c in outcome.slot_values["candidates"]] == PEAKS


@pytest.mark.asyncio
async def test_the_synchronous_fill_still_takes_caller_supplied_values():
    """``fill()`` is unchanged for a caller that already HAS the values (the on-demand path)."""
    library = R.PlanLibrary(warn_on_drift=False)
    chroma = FakeChroma([("argmax_over_n_page_field", 0.30, "argmax")])
    result = await library.retrieve(chroma, MANDATE_062)

    outcome = library.fill(result, {"candidates": [{"name": "A"}, {"name": "B"}], "field": "depth"})

    assert outcome.ok and len(outcome.expansion.candidates) == 2
    assert outcome.slot_values["field"] == "depth"
