"""
Offline unit tests for the plan-library template schema + its GoT adapter — free, no I/O.

Covers the one real design fork of the plan-library work: a hand-authored ``PlanTemplate`` is
filled into ``compiled_plan.py``'s exact shape (so the existing compiled-plan machinery accepts
it unchanged), and the adapter turns that filled plan into native Graph-of-Thought candidate
dicts structurally identical to what ``LlmExpansionPolicy._parse_candidates`` produces — the
shape ``IdeaDag.expand()`` consumes.
"""
import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies import plan_library as adapter
from agent.app.idea_policies.action_constants import ActionResultBuilder
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.plan_library.schema import (
    ExtractionStrategy,
    LeafBlueprint,
    PlanTemplate,
    SlotKind,
    SlotSpec,
    TemplateValidationError,
    bind_template,
    fill_template,
    normalize_template,
    validate_template,
)
from agent.app.testing import compiled_plan


class _StubIO:
    """Raises on any I/O attempt — nothing in this module may touch the network."""

    telemetry = None

    def set_telemetry(self, t):
        return None


def _argmax_template() -> PlanTemplate:
    """A minimal 2-slot argmax template: one page-read leaf per candidate, no deps."""
    return PlanTemplate(
        template_id="argmax_over_n_page_field",
        archetype="argmax",
        title="Argmax over N page fields",
        description="Read one field per named candidate, then pick the extreme.",
        embedding_text="which of these has the largest / deepest / highest value",
        provenance={"source": "hand_authored", "based_on_tasks": ["062"]},
        slots=[
            SlotSpec(name="candidates", kind=SlotKind.ENTITY_LIST, min_arity=2),
            SlotSpec(name="field", kind=SlotKind.FIELD),
            SlotSpec(
                name="source_type",
                kind=SlotKind.SOURCE_TYPE,
                required=False,
                extraction=ExtractionStrategy.DEFAULT,
                default="the authoritative Wikipedia page",
            ),
        ],
        leaves=[
            LeafBlueprint(
                id_pattern="<<item.key>>_field",
                for_each="candidates",
                instruction=(
                    "Open <<source_type>> for <<item.name>> and read, directly from the page, "
                    "its <<field>>. Report ONLY that single value and the source URL."
                ),
                expect="The <<field>> of <<item.name>> as a number with its unit, plus the source URL.",
            ),
        ],
        aggregation=(
            "Write out every candidate's <<field>> explicitly before naming the winner. "
            "Candidates: <<candidates>>."
        ),
    )


def _chain_template() -> PlanTemplate:
    """A 2+-hop chain: each hop depends on the previous instance of the same blueprint."""
    return PlanTemplate(
        template_id="entity_chain_resolution",
        archetype="chain",
        embedding_text="who / what did the thing that the previous page names",
        slots=[SlotSpec(name="hops", kind=SlotKind.HOP_LIST, min_arity=2)],
        leaves=[
            LeafBlueprint(
                id_pattern="hop_<<item.ordinal>>",
                for_each="hops",
                instruction="Hop <<item.ordinal>>: find the <<item.field>> of <<item.label>>.",
                expect="The <<item.field>> named on the page for <<item.label>>, plus its URL.",
                depends_on=["hop_<<item.ordinal>>"],  # self-edge == previous instance
            ),
        ],
        aggregation="State each hop's resolved fact in order, then give the final answer.",
    )


_SLOT_VALUES = {
    "candidates": ["Lake Baikal", {"name": "Lake Tanganyika", "qualifier": "Africa"}],
    "field": "maximum depth",
}


# --------------------------------------------------------------------------------------
# fill_template -> compiled_plan shape
# --------------------------------------------------------------------------------------


def test_fill_template_returns_compiled_plan_shape():
    filled = fill_template(_argmax_template(), _SLOT_VALUES)

    assert set(filled) == {"leaves", "aggregation"}
    assert isinstance(filled["aggregation"], str) and filled["aggregation"]
    assert len(filled["leaves"]) == 2  # one leaf per for_each item
    for leaf in filled["leaves"]:
        assert set(leaf) == {"id", "instruction", "expect", "depends_on"}
        assert leaf["id"] and leaf["instruction"] and leaf["expect"]
        assert leaf["depends_on"] == []


def test_filled_plan_passes_compiled_plan_validate_unchanged():
    """The filled template must be a legal compiled plan with zero changes to compiled_plan.py.

    ``validate_plan`` now also returns ``agg_mode``/``composition`` (additive schema fields, see
    ``compiled_plan.normalize_plan``) — ``None`` for both here since no template sets them yet, so
    the leaves/aggregation shape (what this test actually guards) stays unchanged and doesn't need
    ``fill_template`` to know about those two keys at all.
    """
    filled = fill_template(_argmax_template(), _SLOT_VALUES)
    validated = compiled_plan.validate_plan(filled)
    assert validated["leaves"] == filled["leaves"]
    assert validated["aggregation"] == filled["aggregation"]
    assert validated["agg_mode"] is None
    assert validated["composition"] is None
    assert compiled_plan.plan_structure(filled)["is_pure_fanout"] is True


def test_fill_substitutes_slots_defaults_and_item_attributes():
    filled = fill_template(_argmax_template(), _SLOT_VALUES)
    ids = [leaf["id"] for leaf in filled["leaves"]]
    assert ids == ["lake_baikal_field", "lake_tanganyika_field"]  # <<item.key>> slugged from name

    first = filled["leaves"][0]
    assert "the authoritative Wikipedia page" in first["instruction"]  # optional slot default
    assert "Lake Baikal" in first["instruction"] and "maximum depth" in first["instruction"]
    # ``expect`` re-embeds the concrete entity name, which contract satisfaction keys off.
    assert "Lake Baikal" in first["expect"]
    # A list slot renders as a readable enumeration in the aggregation.
    assert "Lake Baikal, Lake Tanganyika" in filled["aggregation"]


def test_fill_leaves_runtime_dep_placeholders_untouched():
    """``{dep_id}`` belongs to compiled_plan.substitute_deps at EXECUTION time; filling a
    ``<<slot>>`` template must never clobber it."""
    template = _argmax_template()
    template.leaves.append(
        LeafBlueprint(
            id_pattern="compare",
            instruction="Compare {lake_baikal_field} against the other <<field>> values.",
            expect="A comparison of every <<field>> value.",
            depends_on=["<<item.key>>_field"],
        )
    )
    filled = fill_template(template, _SLOT_VALUES)
    compare = filled["leaves"][-1]
    assert "{lake_baikal_field}" in compare["instruction"]
    # A depends_on entry naming a for_each blueprint fans in over ALL its instances.
    assert compare["depends_on"] == ["lake_baikal_field", "lake_tanganyika_field"]


def test_chain_template_self_edge_chains_successive_instances():
    filled = fill_template(
        _chain_template(),
        {"hops": [{"label": "the novel", "field": "author"}, {"label": "the author", "field": "birthplace"}]},
    )
    assert [leaf["id"] for leaf in filled["leaves"]] == ["hop_1", "hop_2"]
    assert filled["leaves"][0]["depends_on"] == []
    assert filled["leaves"][1]["depends_on"] == ["hop_1"]
    assert compiled_plan.topological_waves(filled["leaves"]) == [["hop_1"], ["hop_2"]]


def test_fill_rejects_missing_required_slot():
    with pytest.raises(TemplateValidationError):
        fill_template(_argmax_template(), {"candidates": ["A", "B"]})  # no ``field``


def test_fill_enforces_list_slot_arity():
    with pytest.raises(TemplateValidationError):
        fill_template(_argmax_template(), {"candidates": ["A"], "field": "depth"})  # min_arity=2


def test_fill_rejects_non_list_value_for_list_slot():
    with pytest.raises(TemplateValidationError):
        fill_template(_argmax_template(), {"candidates": "A and B", "field": "depth"})


def test_normalize_template_round_trips_json_shape():
    template = _argmax_template()
    raw = {
        "template_id": template.template_id,
        "archetype": template.archetype,
        "aggregation": template.aggregation,
        "slots": [
            {"name": "candidates", "kind": "entity_list", "min_arity": 2},
            {"name": "field", "kind": "field"},
            {
                "name": "source_type",
                "kind": "source_type",
                "required": False,
                "extraction": "default",
                "default": "the authoritative Wikipedia page",
            },
        ],
        "leaves": [
            {
                "id_pattern": template.leaves[0].id_pattern,
                "for_each": "candidates",
                "instruction": template.leaves[0].instruction,
                "expect": template.leaves[0].expect,
            }
        ],
    }
    parsed = normalize_template(raw)
    assert parsed.slots[0].kind is SlotKind.ENTITY_LIST
    assert parsed.slots[2].extraction is ExtractionStrategy.DEFAULT
    assert parsed.leaves[0].for_each == "candidates"
    assert parsed.leaves[0].depends_on == []
    assert fill_template(parsed, _SLOT_VALUES) == fill_template(template, _SLOT_VALUES)


# --------------------------------------------------------------------------------------
# validate_template rejections
# --------------------------------------------------------------------------------------


def test_validate_rejects_unresolvable_slot_reference():
    template = _argmax_template()
    template.leaves[0].instruction = "Read the <<undeclared_slot>> of <<item.name>>."
    with pytest.raises(TemplateValidationError, match="undeclared_slot"):
        validate_template(template)


def test_validate_rejects_unresolvable_reference_in_aggregation():
    template = _argmax_template()
    template.aggregation = "Rank by <<nope>>."
    with pytest.raises(TemplateValidationError, match="nope"):
        validate_template(template)


def test_validate_rejects_item_reference_without_for_each():
    template = _argmax_template()
    template.leaves[0].for_each = None
    with pytest.raises(TemplateValidationError, match="item"):
        validate_template(template)


def test_validate_rejects_for_each_on_non_list_slot():
    template = _argmax_template()
    template.leaves[0].for_each = "field"  # a FIELD slot, not a list kind
    with pytest.raises(TemplateValidationError, match="not a list kind"):
        validate_template(template)


def test_validate_rejects_for_each_on_undeclared_slot():
    template = _argmax_template()
    template.leaves[0].for_each = "ghosts"
    with pytest.raises(TemplateValidationError, match="ghosts"):
        validate_template(template)


def test_validate_rejects_for_each_id_pattern_without_item_ref():
    template = _argmax_template()
    template.leaves[0].id_pattern = "one_fixed_id"
    with pytest.raises(TemplateValidationError, match="unique leaf id"):
        validate_template(template)


def test_validate_rejects_cyclic_depends_on():
    template = _argmax_template()
    template.leaves = [
        LeafBlueprint(id_pattern="a", instruction="a", expect="a", depends_on=["b"]),
        LeafBlueprint(id_pattern="b", instruction="b", expect="b", depends_on=["a"]),
    ]
    template.aggregation = "Combine a and b."
    with pytest.raises(TemplateValidationError, match="cycle"):
        validate_template(template)


def test_validate_rejects_self_dependency_without_for_each():
    template = _argmax_template()
    template.leaves = [LeafBlueprint(id_pattern="a", instruction="a", expect="a", depends_on=["a"])]
    template.aggregation = "Report a."
    with pytest.raises(TemplateValidationError, match="depends on itself"):
        validate_template(template)


def test_validate_rejects_unknown_depends_on_target():
    template = _argmax_template()
    template.leaves[0].depends_on = ["not_a_blueprint"]
    with pytest.raises(TemplateValidationError, match="not_a_blueprint"):
        validate_template(template)


def test_validate_rejects_missing_expect_and_aggregation():
    template = _argmax_template()
    template.leaves[0].expect = ""
    with pytest.raises(TemplateValidationError, match="expect is required"):
        validate_template(template)

    template = _argmax_template()
    template.aggregation = ""
    with pytest.raises(TemplateValidationError, match="aggregation is required"):
        validate_template(template)


def test_validate_rejects_unknown_slot_kind():
    with pytest.raises(TemplateValidationError, match="unknown kind"):
        normalize_template(
            {"template_id": "t", "slots": [{"name": "x", "kind": "wat"}], "leaves": [], "aggregation": "a"}
        )


# --------------------------------------------------------------------------------------
# adapter -> GoT candidates
# --------------------------------------------------------------------------------------


def test_adapter_emits_structurally_valid_got_candidates():
    expansion = adapter.candidates_from_template(_argmax_template(), _SLOT_VALUES)
    assert len(expansion.candidates) == 2
    action_values = {a.value for a in IdeaActionType}

    for candidate in expansion.candidates:
        # Exactly the keys IdeaDag.expand() reads off a candidate dict.
        assert set(candidate) == {"title", "details", "score"}
        assert isinstance(candidate["title"], str) and candidate["title"]
        assert candidate["score"] is None
        details = candidate["details"]
        assert details[DetailKey.ACTION.value] in action_values
        assert details[DetailKey.ACTION.value] == IdeaActionType.SEARCH.value
        assert details[DetailKey.EXPECT.value].strip()
        assert details[DetailKey.QUERY.value].strip()
        assert details[DetailKey.INTENT.value].strip()
        assert details[DetailKey.GOAL.value] and details[DetailKey.ORIGINAL_GOAL.value]
        # Same auto-tag _parse_candidates puts on every search candidate.
        assert details[DetailKey.PROVIDES_DATA.value] == {"type": "urls_from_search"}
        assert details[adapter.PLAN_LIBRARY_TEMPLATE_ID] == "argmax_over_n_page_field"
        assert details[adapter.PLAN_LIBRARY_ORIGIN] == adapter.ORIGIN_AUTO


def test_adapter_candidates_are_expandable_into_the_graph():
    """End-to-end shape check: the candidates go straight into ``IdeaDag.expand()``."""
    expansion = adapter.candidates_from_template(_argmax_template(), _SLOT_VALUES)
    graph = IdeaDag(root_title="Which lake is deeper?")
    created = graph.expand(graph.root_id(), expansion.candidates)

    assert [n.title for n in created] == [c["title"] for c in expansion.candidates]
    assert all(n.status is IdeaNodeStatus.PENDING for n in created)
    assert all(n.details[DetailKey.ACTION.value] == IdeaActionType.SEARCH.value for n in created)


def test_adapter_query_comes_from_slot_values_not_instruction_prose():
    expansion = adapter.candidates_from_template(_argmax_template(), _SLOT_VALUES)
    queries = [c["details"][DetailKey.QUERY.value] for c in expansion.candidates]
    assert queries == ["Lake Baikal maximum depth", "Lake Tanganyika Africa maximum depth"]


def test_adapter_expect_is_carried_through_unchanged():
    filled = bind_template(_argmax_template(), _SLOT_VALUES)
    expansion = adapter.candidates_from_filled_plan(filled)
    assert [c["details"][DetailKey.EXPECT.value] for c in expansion.candidates] == [
        leaf.expect for leaf in filled.leaves
    ]


def test_adapter_aggregation_becomes_parent_merge_guidance():
    expansion = adapter.candidates_from_template(_argmax_template(), _SLOT_VALUES)
    # MergeLeafAction reads the parent's INTENT as ``parent_intent``.
    assert expansion.parent_details[DetailKey.INTENT.value] == expansion.aggregation
    assert "explicitly before naming the winner" in expansion.aggregation
    # No leaf has deps -> the siblings may run in parallel, like meta.execute_all_children.
    meta = expansion.parent_details[DetailKey.EXPANSION_META.value]
    assert meta == {DetailKey.EXECUTE_ALL_CHILDREN.value: True}


def test_adapter_marks_dependent_batch_as_sequential():
    expansion = adapter.candidates_from_template(
        _chain_template(),
        {"hops": [{"label": "the novel", "field": "author"}, {"label": "the author", "field": "birthplace"}]},
        origin=adapter.ORIGIN_ACTION,
    )
    meta = expansion.parent_details[DetailKey.EXPANSION_META.value]
    assert meta == {DetailKey.EXECUTE_ALL_CHILDREN.value: False}
    assert expansion.candidates[1]["details"][adapter.PLAN_LIBRARY_DEPENDS_ON] == ["hop_1"]
    assert expansion.candidates[0]["details"][adapter.PLAN_LIBRARY_ORIGIN] == adapter.ORIGIN_ACTION


def test_adapter_exposes_the_filled_plan_for_logging():
    expansion = adapter.candidates_from_template(_argmax_template(), _SLOT_VALUES)
    assert expansion.plan == fill_template(_argmax_template(), _SLOT_VALUES)
    assert bool(expansion) is True


def test_adapter_rejects_a_bare_plan_dict():
    """The bare compiled-plan dict has thrown the structured slots away, so it cannot build a
    query — the adapter takes the FilledPlan instead of silently re-parsing prose."""
    with pytest.raises(TemplateValidationError):
        adapter.candidates_from_filled_plan(fill_template(_argmax_template(), _SLOT_VALUES))


# --------------------------------------------------------------------------------------
# depends_on -> the engine's existing sequential-execution mechanism
# --------------------------------------------------------------------------------------


def _expand_chain_into_graph():
    expansion = adapter.candidates_from_template(
        _chain_template(),
        {"hops": [{"label": "the novel", "field": "author"}, {"label": "the author", "field": "birthplace"}]},
    )
    graph = IdeaDag(root_title="Where was the author born?")
    created = graph.expand(graph.root_id(), expansion.candidates)
    return graph, created


def test_link_dependencies_writes_requires_data_between_siblings():
    graph, created = _expand_chain_into_graph()
    hop1, hop2 = created

    assert adapter.link_dependencies(created) == 1
    assert DetailKey.REQUIRES_DATA.value not in hop1.details
    assert hop2.details[DetailKey.REQUIRES_DATA.value] == {
        "type": "urls_from_search",
        "source_node_id": hop1.node_id,
    }
    assert hop2.details[adapter.PLAN_LIBRARY_DEPENDS_ON_NODES] == [hop1.node_id]


def test_linked_dependency_actually_gates_engine_execution():
    """The load-bearing claim: ``depends_on`` rides the engine's EXISTING gate, no new code."""
    graph, created = _expand_chain_into_graph()
    hop1, hop2 = created
    adapter.link_dependencies(created)
    engine = IdeaDagEngine(io=_StubIO(), settings={})

    assert engine._has_required_data(graph, hop1) is True
    assert engine._has_required_data(graph, hop2) is False  # blocked: source not done

    hop1.status = IdeaNodeStatus.DONE
    hop1.details[DetailKey.ACTION_RESULT.value] = ActionResultBuilder.success(
        action=IdeaActionType.SEARCH.value,
        results=[{"url": "https://example.org/novel", "title": "The novel"}],
    )
    assert engine._has_required_data(graph, hop2) is True


def test_link_dependencies_points_a_fan_in_leaf_at_its_deepest_source():
    template = _argmax_template()
    template.leaves.append(
        LeafBlueprint(
            id_pattern="compare",
            instruction="Compare every <<field>> value gathered above.",
            expect="A comparison of every <<field>> value.",
            depends_on=["<<item.key>>_field"],
        )
    )
    template.leaves.append(
        LeafBlueprint(
            id_pattern="conclude",
            instruction="Name the winner for <<field>>.",
            expect="The winning candidate for <<field>> with its value and URL.",
            depends_on=["<<item.key>>_field", "compare"],
        )
    )
    expansion = adapter.candidates_from_template(template, _SLOT_VALUES)
    graph = IdeaDag(root_title="Which lake is deeper?")
    created = graph.expand(graph.root_id(), expansion.candidates)

    assert adapter.link_dependencies(created) == 2
    conclude = created[-1]
    compare = created[-2]
    # ``requires_data`` names a single source: the DEEPEST dependency (wave 1), not a wave-0 leaf.
    assert conclude.details[DetailKey.REQUIRES_DATA.value]["source_node_id"] == compare.node_id
    assert len(conclude.details[adapter.PLAN_LIBRARY_DEPENDS_ON_NODES]) == 3


def test_link_dependencies_ignores_untagged_nodes():
    graph = IdeaDag(root_title="root")
    created = graph.expand(
        graph.root_id(),
        [{"title": "organic", "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}, "score": None}],
    )
    assert adapter.link_dependencies(created) == 0
    assert DetailKey.REQUIRES_DATA.value not in created[0].details


# --------------------------------------------------------------------------------------
# every leaf gets its OWN page visit (the grounding follow-through)
# --------------------------------------------------------------------------------------


def _expand_argmax_into_graph():
    """The N-way fan-out — the shape where a single shared visit loses N-1 entities."""
    expansion = adapter.candidates_from_template(_argmax_template(), _SLOT_VALUES)
    graph = IdeaDag(root_title="Which lake is deeper?")
    created = graph.expand(graph.root_id(), expansion.candidates)
    return graph, created


def test_every_search_leaf_gets_its_own_visit_sibling():
    """One visit per leaf — not zero (search-only, ungrounded) and not one shared."""
    graph, created = _expand_argmax_into_graph()

    visits = adapter.link_page_visits(graph, created)

    assert len(visits) == len(created) == 2
    root = graph.get_node(graph.root_id())
    # siblings, not children: `step()` routes any node with an action to the leaf handler, so a
    # visit hung UNDER a search would never be reached.
    assert [v.parent_id for v in visits] == [root.node_id] * 2
    assert root.children == [c.node_id for c in created] + [v.node_id for v in visits]
    for visit in visits:
        assert visit.details[DetailKey.ACTION.value] == IdeaActionType.VISIT.value
        assert visit.details[DetailKey.IS_LEAF.value] is True
        assert visit.details["link_count"] == 1


def test_each_visit_requires_data_from_its_own_search_not_an_arbitrary_one():
    """The entity affinity: leaf i's page read is fed by leaf i's search, not by whichever
    sibling happened to finish first (which is all the generic grounding hook can manage)."""
    graph, created = _expand_argmax_into_graph()
    baikal, tanganyika = created

    visits = adapter.link_page_visits(graph, created)

    assert [v.details[DetailKey.REQUIRES_DATA.value]["source_node_id"] for v in visits] == [
        baikal.node_id, tanganyika.node_id
    ]
    assert {v.details[DetailKey.REQUIRES_DATA.value]["type"] for v in visits} == {"urls_from_search"}
    # the pairing is walkable from either end, for telemetry and for idempotence
    assert baikal.details[adapter.PLAN_LIBRARY_VISIT_NODE] == visits[0].node_id
    assert visits[0].details[adapter.PLAN_LIBRARY_VISIT_FOR] == "lake_baikal_field"
    assert visits[1].details[adapter.PLAN_LIBRARY_VISIT_FOR] == "lake_tanganyika_field"


def test_each_visits_link_idea_names_its_own_entity():
    """``link_idea`` is what steers ``VisitLeafAction``'s URL pick. The generic hook's
    "URL from search results or mandate" cannot discriminate between six peaks' pages; a
    leaf's own filled instruction can."""
    graph, created = _expand_argmax_into_graph()

    baikal_visit, tanganyika_visit = adapter.link_page_visits(graph, created)

    baikal_idea = baikal_visit.details["link_idea"]
    tanganyika_idea = tanganyika_visit.details["link_idea"]
    assert "Lake Baikal" in baikal_idea and "Lake Tanganyika" not in baikal_idea
    assert "Lake Tanganyika" in tanganyika_idea and "Lake Baikal" not in tanganyika_idea
    # ...and it says what KIND of page, which is how a wiki page beats a mirror/aggregator
    assert "Wikipedia" in baikal_idea
    assert baikal_idea != "URL from search results or mandate"
    assert len(baikal_idea) <= 200  # same clip VisitLeafAction applies to its own fallback


def test_the_visit_carries_the_leafs_extraction_contract():
    """``_effective_intent`` builds the extraction target out of intent + expect, so the page
    read aims at the same contract the template authored for that leaf."""
    graph, created = _expand_argmax_into_graph()
    baikal = created[0]

    visit = adapter.link_page_visits(graph, created)[0]

    assert visit.details[DetailKey.INTENT.value] == baikal.details[DetailKey.INTENT.value]
    assert visit.details[DetailKey.EXPECT.value] == baikal.details[DetailKey.EXPECT.value]
    assert visit.details[adapter.PLAN_LIBRARY_TEMPLATE_ID] == "argmax_over_n_page_field"
    assert visit.details[adapter.PLAN_LIBRARY_ORIGIN] == adapter.ORIGIN_AUTO


def test_a_visit_waits_on_its_own_search_only():
    """The claim in engine terms: a sibling's completed search does not unblock this visit —
    only its own does. Same gate ``link_dependencies`` rides, no new engine code."""
    graph, created = _expand_argmax_into_graph()
    baikal, tanganyika = created
    baikal_visit, tanganyika_visit = adapter.link_page_visits(graph, created)
    engine = IdeaDagEngine(io=_StubIO(), settings={})

    baikal.status = IdeaNodeStatus.DONE
    baikal.details[DetailKey.ACTION_RESULT.value] = ActionResultBuilder.success(
        action=IdeaActionType.SEARCH.value,
        results=[{"url": "https://en.wikipedia.org/wiki/Lake_Baikal", "title": "Lake Baikal"}],
    )

    assert engine._has_required_data(graph, baikal_visit) is True
    assert engine._has_required_data(graph, tanganyika_visit) is False


def test_link_page_visits_is_idempotent():
    """Re-running the pass (a re-expansion of the same parent) must not double the visits."""
    graph, created = _expand_argmax_into_graph()

    first = adapter.link_page_visits(graph, created)
    second = adapter.link_page_visits(graph, created)

    assert len(first) == 2 and second == []
    assert len(graph.get_node(graph.root_id()).children) == 4


def test_link_page_visits_ignores_organic_children():
    """Same safety as ``link_dependencies``: LLM-invented children carry no marker, so an
    unconditional call is a no-op and the generic hooks keep owning the organic path."""
    graph = IdeaDag(root_title="root")
    created = graph.expand(
        graph.root_id(),
        [{"title": "organic", "details": {DetailKey.ACTION.value: IdeaActionType.SEARCH.value}, "score": None}],
    )
    assert adapter.link_page_visits(graph, created) == []
    assert len(graph.get_node(graph.root_id()).children) == 1
