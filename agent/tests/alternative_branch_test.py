"""Phase 2 of the DAG v2 shape-spectrum work: sequential A->B fallback and concurrent
race-and-merge (opt-in, ``expansion_alternative_branch_enabled`` + its two consumers).

Every mechanism here is a status/contract comparison, so the whole design is offline
testable with no live LLM call. Layers pinned, in the order they run:

  1. schema + prompt + candidate parse (the two optional authored fields);
  2. ``alternative_branch.link_alternatives`` / ``link_race_groups`` resolution;
  3. the four eligibility filters a parked fallback must be invisible to — the panel's
     blocking finding, and the reason ``ALTERNATIVE_PENDING`` exists at all;
  4. ``IdeaDagEngine._maybe_promote_alternative_branch``: both trigger sub-conditions,
     at BOTH call sites (the sequential ``_apply_action_result`` chain and the
     auto-parallel batch loop, which bypasses it entirely);
  5. ``idea_sequencing.siblings_are_independent``'s new race-group rule;
  6. ``SimpleMergePolicy.select_winner``'s four-step mechanical chain.
"""
from __future__ import annotations

import json
import logging

import pytest
from unittest.mock import AsyncMock, patch

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_schemas import (
    EXPANSION_JSON_SCHEMA,
    EXPANSION_JSON_SCHEMA_WITH_BRANCHING,
    EXPANSION_JSON_SCHEMA_WITH_EXPECT,
)
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_node_state import get_pending_executable_nodes, is_action_ready
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies import alternative_branch as alt
from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.base import (
    DecompositionPolicy,
    DetailKey,
    EvaluationPolicy,
    ExpansionPolicy,
    IdeaActionType,
    IdeaNodeStatus,
)
from agent.app.idea_policies.expansion import (
    LlmExpansionPolicy,
    _ALTERNATIVE_BRANCH_ADDENDUM,
)
from agent.app.idea_sequencing import siblings_are_independent
from agent.app.llm_backends import LLMBackend

_LOGGER = logging.getLogger("alternative_branch_test")

_MANDATE = "Report the main span of the Hardanger Bridge in metres. Do not guess."
_GOAL = "Find the main span of the Hardanger Bridge"
#: A goal naming only an ENTITY: its contract asks for no measurable datum, so opening any
#: page that repeats the entity's name satisfies it without verifying anything (F35).
_SUBJECT_GOAL = "Open the Hardanger Bridge page"
_DATUM_PAGE = "The Hardanger Bridge has a main span of 1310 m and opened in 2013."
_SUBJECT_ONLY_PAGE = "The Hardanger Bridge is a suspension bridge in Vestland, Norway."


# ======================================================================================
# layer 1 — schema, prompt addendum, candidate parse
# ======================================================================================


def _candidate_props(schema):
    return schema["schema"]["properties"]["candidates"]["items"]


def test_branching_schema_is_an_additive_superset_of_the_expect_variant():
    branching = _candidate_props(EXPANSION_JSON_SCHEMA_WITH_BRANCHING)
    expect = _candidate_props(EXPANSION_JSON_SCHEMA_WITH_EXPECT)
    assert set(expect["properties"]) < set(branching["properties"])
    assert set(branching["properties"]) - set(expect["properties"]) == {
        "alternative_of", "race_group",
    }
    # Both new fields are OPTIONAL: most candidates carry neither.
    assert branching["required"] == expect["required"] == ["title", "action", "details"]
    for field in ("alternative_of", "race_group"):
        assert branching["properties"][field]["type"] == "string"
        assert branching["properties"][field]["description"]


def test_default_schema_is_untouched_by_the_new_variant():
    """Regression: the branching fields must not leak into the shipped default schema."""
    assert set(_candidate_props(EXPANSION_JSON_SCHEMA)["properties"]) == {
        "title", "action", "details",
    }


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for k in (
        "LLM_PROVIDER", "LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY", "MODEL_API_URL", "OPENAI_BASE_URL", "OPENROUTER_BASE_URL",
        "MODEL_NAME",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_NAME", "gpt-4.1-nano")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")


def _make_connector_with_mock_backend():
    from shared.connector_config import ConnectorConfig

    cfg = ConnectorConfig()
    mock_backend = AsyncMock(spec=LLMBackend)
    mock_backend.normalize_payload.side_effect = lambda p, *_, **__: p
    mock_backend.simplify_payload.side_effect = lambda p: dict(p)
    with patch("agent.app.connector_llm.create_llm_backend", return_value=mock_backend):
        from agent.app.connector_llm import ConnectorLLM

        return ConnectorLLM(cfg)


class RecordingIO:
    def __init__(self, connector, response: str):
        self._connector = connector
        self._response = response
        self.payload_kwargs: dict = {}

    def set_telemetry(self, telemetry):
        return None

    def build_llm_payload(self, **kwargs):
        self.payload_kwargs = kwargs
        return self._connector.build_payload(**kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs):
        return self._response


def _prompt_text(io) -> str:
    return "\n".join(str(m.get("content", "")) for m in io.payload_kwargs["messages"])


_BRANCHING_RESPONSE = json.dumps({
    "candidates": [
        {
            "title": "Read the bridge's encyclopedia page",
            "action": "visit",
            "details": {"optional_url": "https://en.wikipedia.org/wiki/Hardanger_Bridge"},
            "race_group": "span",
        },
        {
            "title": "Search the roads authority for the span",
            "action": "search",
            "details": {"query": "Hardanger Bridge main span"},
            "race_group": "span",
        },
        {
            "title": "Fall back to the bridge database",
            "action": "search",
            "details": {"query": "Hardanger bru hovedspenn"},
            "alternative_of": "Read the bridge's encyclopedia page",
        },
    ]
})


async def _expand_with(settings):
    connector = _make_connector_with_mock_backend()
    io = RecordingIO(connector, _BRANCHING_RESPONSE)
    policy = LlmExpansionPolicy(io=io, settings=settings)
    graph = IdeaDag(root_title="Research", root_details={"mandate": _MANDATE})
    candidates = await policy.expand(graph, graph.root_id())
    return candidates, _prompt_text(io)


@pytest.mark.asyncio
async def test_flag_off_neither_addendum_nor_parsed_fields():
    candidates, prompt = await _expand_with({})
    assert _ALTERNATIVE_BRANCH_ADDENDUM not in prompt
    assert "MULTIPLE APPROACHES" not in prompt
    assert '"race_group"' not in prompt
    for candidate in candidates:
        assert DetailKey.RACE_GROUP.value not in candidate["details"]
        assert alt.ALTERNATIVE_OF_HINT not in candidate["details"]


@pytest.mark.asyncio
async def test_flag_on_round_trips_both_fields_onto_the_candidates():
    candidates, prompt = await _expand_with({"expansion_alternative_branch_enabled": True})
    assert _ALTERNATIVE_BRANCH_ADDENDUM in prompt
    assert "alternative_of" in prompt and "race_group" in prompt
    assert candidates[0]["details"][DetailKey.RACE_GROUP.value] == "span"
    assert candidates[1]["details"][DetailKey.RACE_GROUP.value] == "span"
    assert candidates[2]["details"][alt.ALTERNATIVE_OF_HINT] == (
        "Read the bridge's encyclopedia page"
    )
    assert DetailKey.RACE_GROUP.value not in candidates[2]["details"]


@pytest.mark.asyncio
async def test_flag_on_drops_a_non_string_hint():
    bad = json.dumps({"candidates": [{
        "title": "t", "action": "search", "details": {"query": "q"},
        "alternative_of": 17, "race_group": "   ",
    }]})
    connector = _make_connector_with_mock_backend()
    io = RecordingIO(connector, bad)
    policy = LlmExpansionPolicy(
        io=io, settings={"expansion_alternative_branch_enabled": True}
    )
    graph = IdeaDag(root_title="Research", root_details={"mandate": _MANDATE})
    candidates = await policy.expand(graph, graph.root_id())
    assert alt.ALTERNATIVE_OF_HINT not in candidates[0]["details"]
    assert DetailKey.RACE_GROUP.value not in candidates[0]["details"]


# ======================================================================================
# layer 2 — resolution
# ======================================================================================


def _graph_with_children(*specs):
    """(graph, parent, [children]) for ``specs`` of ``(title, details)``."""
    graph = IdeaDag(root_title="root", root_details={"mandate": _MANDATE})
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    children = [graph.add_child(parent.node_id, title, details=dict(det))
                for title, det in specs]
    return graph, parent, children


def _leaf(action=IdeaActionType.SEARCH.value, **extra):
    details = {DetailKey.ACTION.value: action, DetailKey.IS_LEAF.value: True}
    details.update(extra)
    return details


def test_link_alternatives_pairs_and_parks_the_fallback():
    _g, _p, (primary, fallback) = _graph_with_children(
        ("Route A", _leaf()),
        ("Route B", _leaf(**{alt.ALTERNATIVE_OF_HINT: "route a"})),
    )
    assert alt.link_alternatives([primary, fallback]) == 1

    assert primary.details[DetailKey.HAS_ALTERNATIVE_NODE.value] == fallback.node_id
    assert fallback.details[DetailKey.ALTERNATIVE_OF_NODE.value] == primary.node_id
    assert fallback.details[alt.ALTERNATIVE_PENDING] is True
    assert fallback.status == IdeaNodeStatus.BLOCKED
    # The authored hint is consumed, so a second pass cannot re-link it.
    assert alt.ALTERNATIVE_OF_HINT not in fallback.details
    assert primary.status == IdeaNodeStatus.PENDING


def test_link_alternatives_ignores_an_unresolvable_or_self_referential_hint():
    _g, _p, (a, b, c) = _graph_with_children(
        ("Route A", _leaf(**{alt.ALTERNATIVE_OF_HINT: "a candidate that does not exist"})),
        ("Route B", _leaf(**{alt.ALTERNATIVE_OF_HINT: "Route B"})),
        ("Route C", _leaf()),
    )
    assert alt.link_alternatives([a, b, c]) == 0
    for node in (a, b, c):
        assert node.status == IdeaNodeStatus.PENDING
        assert alt.ALTERNATIVE_PENDING not in node.details


def test_link_alternatives_gives_a_primary_at_most_one_fallback():
    """A promotion cursor for a chain of fallbacks does not exist, and two fallbacks have
    no defined promotion order — so the second declared one stays an ordinary sibling."""
    _g, _p, (primary, first, second) = _graph_with_children(
        ("Route A", _leaf()),
        ("Route B", _leaf(**{alt.ALTERNATIVE_OF_HINT: "Route A"})),
        ("Route C", _leaf(**{alt.ALTERNATIVE_OF_HINT: "Route A"})),
    )
    assert alt.link_alternatives([primary, first, second]) == 1
    assert primary.details[DetailKey.HAS_ALTERNATIVE_NODE.value] == first.node_id
    assert alt.ALTERNATIVE_PENDING not in second.details
    assert second.status == IdeaNodeStatus.PENDING


def test_link_race_groups_tags_members_and_the_shared_ancestor():
    _g, parent, (a, b, c) = _graph_with_children(
        ("Route A", _leaf(**{DetailKey.RACE_GROUP.value: "Span"})),
        ("Route B", _leaf(**{DetailKey.RACE_GROUP.value: "span"})),
        ("Route C", _leaf()),
    )
    assert alt.link_race_groups(parent, [a, b, c]) == 2

    assert a.details[DetailKey.RACE_GROUP.value] == "span"
    assert b.details[DetailKey.RACE_GROUP.value] == "span"
    assert DetailKey.RACE_GROUP.value not in c.details
    # Membership lives on the ANCESTOR (not on graph topology): that is what lets the
    # merge node keep its ordinary single-parent shape.
    assert parent.details[alt.RACE_GROUPS] == {"span": [a.node_id, b.node_id]}
    # Racers dispatch normally — nothing is parked.
    assert a.status == b.status == IdeaNodeStatus.PENDING


def test_link_race_groups_drops_a_group_of_one():
    _g, parent, (a, b) = _graph_with_children(
        ("Route A", _leaf(**{DetailKey.RACE_GROUP.value: "span"})),
        ("Route B", _leaf(**{DetailKey.RACE_GROUP.value: "opening year"})),
    )
    assert alt.link_race_groups(parent, [a, b]) == 0
    assert DetailKey.RACE_GROUP.value not in a.details
    assert DetailKey.RACE_GROUP.value not in b.details
    assert alt.RACE_GROUPS not in parent.details


# ======================================================================================
# layer 3 — the four eligibility filters (the panel's blocking finding)
# ======================================================================================


class DummyIO:
    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    async def expand(self, graph, node_id, memories=None):
        return []


class FakeEvaluation(EvaluationPolicy):
    async def evaluate(self, graph, node_id):
        graph.evaluate(node_id, 0.6)
        return 0.6

    async def evaluate_batch(self, graph, parent_id, candidate_ids):
        return {nid: (graph.evaluate(nid, 0.6) or 0.6) for nid in candidate_ids}


class FakeDecomposition(DecompositionPolicy):
    def should_decompose(self, graph, node_id):
        return False


class ScriptedAction(LeafAction):
    """Returns whatever the current test scripted for the node it is executing."""

    RESULTS: dict = {}

    async def execute(self, graph, node_id, io):
        return dict(ScriptedAction.RESULTS[node_id])


class FakeRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type):
        return ScriptedAction(settings=self.settings)


def _make_engine(**overrides):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "got_reexpand_enabled": False,
        "auto_parallel_siblings": False,
        "allow_execute_all_children": True,
        "expansion_alternative_branch_enabled": True,
        "got_contract_reexpand_enabled": True,
        "got_contract_veto_requires_datum_enabled": True,
    }
    settings.update(overrides)
    engine = IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=FakeExpansion(settings),
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeRegistry(settings),
    )
    return engine


def _parked_pair():
    graph, parent, (primary, fallback) = _graph_with_children(
        ("Route A", _leaf(**{DetailKey.QUERY.value: "a"})),
        ("Route B", _leaf(**{DetailKey.QUERY.value: "b",
                             alt.ALTERNATIVE_OF_HINT: "Route A"})),
    )
    alt.link_alternatives([primary, fallback])
    return graph, parent, primary, fallback


@pytest.mark.asyncio
async def test_parked_fallback_is_excluded_from_all_four_eligibility_paths():
    """The panel's blocking finding: BLOCKED alone is NOT a dispatch gate anywhere in this
    engine, so without the explicit marker check the 'sequential' fallback races its own
    primary the first time both clear ``siblings_are_independent``."""
    engine = _make_engine(auto_parallel_siblings=True)
    graph, parent, primary, fallback = _parked_pair()
    executed = []

    async def _spy(graph_, parent_id, node_id):
        # Records the dispatch and leaves the node untouched, so the primary never reaches
        # a terminal status and the promote/retire step below stays out of the way.
        executed.append(node_id)
        return None

    engine._execute_action = _spy
    engine._execute_action_guarded = _spy

    # site 2: the shared readiness predicate
    assert is_action_ready(fallback, 0) is False
    assert is_action_ready(primary, 0) is True

    # sites 1, 3 and 4 all funnel through the intermediate-node handler: whichever branch
    # it takes (batch or single-select), only the primary may run.
    await engine._handle_intermediate_node(graph, parent.node_id, 0, None)
    assert executed == [primary.node_id]
    assert fallback.status == IdeaNodeStatus.BLOCKED

    # site 3 explicitly: the score loop's BLOCKED special-case would otherwise wave an
    # unscored BLOCKED node straight into selection.
    fallback.score = None
    scored = []
    for child_id in parent.children:
        child = graph.get_node(child_id)
        if child.status == IdeaNodeStatus.BLOCKED and alt.is_alternative_pending(child):
            continue
        scored.append(child_id)
    assert fallback.node_id not in scored

    # and it is not counted as work the run still owes at finalize time.
    assert fallback.node_id not in [n.node_id for n in get_pending_executable_nodes(graph)]


def test_promoted_fallback_becomes_eligible_again():
    graph, parent, primary, fallback = _parked_pair()
    fallback.details.pop(alt.ALTERNATIVE_PENDING)
    fallback.status = IdeaNodeStatus.PENDING
    assert is_action_ready(fallback, 0) is True
    assert alt.is_alternative_pending(fallback) is False


def test_parked_fallback_does_not_stall_its_parents_merge():
    """A child left BLOCKED forever would block ``are_children_ready_to_merge``, so a
    successful primary's parent could never merge at all."""
    graph, parent, primary, fallback = _parked_pair()
    primary.status = IdeaNodeStatus.DONE
    primary.details[DetailKey.ACTION_RESULT.value] = {"success": True, "content": "x"}
    policy = SimpleMergePolicy(settings={})
    assert policy.are_children_ready_to_merge(graph, parent.node_id) is True


# ======================================================================================
# layer 4 — the sequential trigger, both sub-conditions, both call sites
# ======================================================================================


def _visit_result(content):
    return {
        "action": IdeaActionType.VISIT.value, "success": True,
        "url": "https://en.wikipedia.org/wiki/Hardanger_Bridge",
        "title": "Hardanger Bridge", "content": content,
    }


def _visit_pair(goal=_GOAL, graph_mandate=_MANDATE):
    graph = IdeaDag(root_title="root", root_details={"mandate": graph_mandate})
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    primary = graph.add_child(parent.node_id, goal, details=_leaf(
        IdeaActionType.VISIT.value, **{DetailKey.GOAL.value: goal}))
    fallback = graph.add_child(parent.node_id, "Fallback route", details=_leaf(
        IdeaActionType.VISIT.value,
        **{DetailKey.GOAL.value: goal, alt.ALTERNATIVE_OF_HINT: goal}))
    alt.link_alternatives([primary, fallback])
    return graph, parent, primary, fallback


@pytest.mark.asyncio
async def test_sequential_site_promotes_on_a_failed_primary():
    engine = _make_engine(got_alternative_branch_promote_on_fail_enabled=True)
    graph, parent, primary, fallback = _visit_pair()
    primary.details[DetailKey.ACTION_RESULT.value] = {
        "action": "visit", "success": False, "error": "404",
    }

    await engine._apply_action_result(graph, primary.node_id, 3)

    assert primary.status == IdeaNodeStatus.FAILED
    assert fallback.status == IdeaNodeStatus.PENDING
    assert alt.ALTERNATIVE_PENDING not in fallback.details
    assert fallback.details[alt.ALTERNATIVE_TRIGGER_REASON] == "primary_failed"


@pytest.mark.asyncio
async def test_sequential_site_promotes_on_a_done_but_unverified_primary():
    """F35's distinction: 'the page repeats the words of my own goal' is true of every
    wrong-but-plausible route, so it is not enough to abandon the fallback."""
    engine = _make_engine(got_alternative_branch_promote_on_unverified_enabled=True)
    graph, parent, primary, fallback = _visit_pair(goal=_SUBJECT_GOAL)
    primary.details[DetailKey.ACTION_RESULT.value] = _visit_result(_SUBJECT_ONLY_PAGE)

    await engine._apply_action_result(graph, primary.node_id, 3)

    assert primary.status == IdeaNodeStatus.DONE
    assert fallback.status == IdeaNodeStatus.PENDING
    assert fallback.details[alt.ALTERNATIVE_TRIGGER_REASON] == "primary_unverified_datum"


@pytest.mark.asyncio
async def test_unverified_trigger_needs_the_f35_datum_signal_armed():
    """Without ``contract_veto_requires_datum_enabled`` there is no datum half of the
    verdict for anything else in the run to trust, so this trigger stays silent."""
    engine = _make_engine(
        got_alternative_branch_promote_on_unverified_enabled=True,
        got_contract_veto_requires_datum_enabled=False,
    )
    graph, _p, primary, fallback = _visit_pair(goal=_SUBJECT_GOAL)
    primary.details[DetailKey.ACTION_RESULT.value] = _visit_result(_SUBJECT_ONLY_PAGE)

    await engine._apply_action_result(graph, primary.node_id, 3)

    assert fallback.status == IdeaNodeStatus.SKIPPED
    assert alt.ALTERNATIVE_TRIGGER_REASON not in fallback.details


@pytest.mark.asyncio
async def test_a_datum_verified_primary_retires_its_fallback_unused():
    engine = _make_engine(
        got_alternative_branch_promote_on_fail_enabled=True,
        got_alternative_branch_promote_on_unverified_enabled=True,
    )
    graph, parent, primary, fallback = _visit_pair()
    primary.details[DetailKey.ACTION_RESULT.value] = _visit_result(_DATUM_PAGE)

    await engine._apply_action_result(graph, primary.node_id, 3)

    assert primary.status == IdeaNodeStatus.DONE
    assert fallback.status == IdeaNodeStatus.SKIPPED
    assert fallback.details[alt.ALTERNATIVE_RETIRED] is True
    assert alt.ALTERNATIVE_PENDING not in fallback.details


@pytest.mark.asyncio
async def test_the_two_sub_conditions_are_independently_switchable():
    engine = _make_engine(got_alternative_branch_promote_on_unverified_enabled=True)
    graph, _p, primary, fallback = _visit_pair()
    primary.details[DetailKey.ACTION_RESULT.value] = {"success": False, "error": "404"}
    await engine._apply_action_result(graph, primary.node_id, 3)
    # promote-on-fail is OFF, so a FAILED primary retires the fallback instead.
    assert fallback.status == IdeaNodeStatus.SKIPPED

    engine = _make_engine(got_alternative_branch_promote_on_fail_enabled=True)
    graph, _p, primary, fallback = _visit_pair(goal=_SUBJECT_GOAL)
    primary.details[DetailKey.ACTION_RESULT.value] = _visit_result(_SUBJECT_ONLY_PAGE)
    await engine._apply_action_result(graph, primary.node_id, 3)
    # promote-on-unverified is OFF, so a subject-only DONE primary is accepted.
    assert fallback.status == IdeaNodeStatus.SKIPPED


@pytest.mark.asyncio
async def test_flags_off_leaves_every_existing_arm_untouched():
    engine = _make_engine(expansion_alternative_branch_enabled=False)
    graph, _p, primary, fallback = _visit_pair()
    # No authored alternative at all -> the trigger is a single dict lookup and a False.
    primary.details.pop(DetailKey.HAS_ALTERNATIVE_NODE.value)
    primary.details[DetailKey.ACTION_RESULT.value] = {"success": False, "error": "404"}
    await engine._apply_action_result(graph, primary.node_id, 3)
    assert engine._maybe_promote_alternative_branch(graph, primary.node_id) is False
    assert fallback.status == IdeaNodeStatus.BLOCKED


@pytest.mark.asyncio
async def test_batch_site_promotes_for_a_failed_batched_sibling():
    """The auto-parallel loop bypasses ``_apply_action_result`` entirely and its
    FAILED/timeout children ``continue`` with no follow-up of any kind — so without its own
    call site the trigger would never fire for a batched sibling."""
    engine = _make_engine(
        auto_parallel_siblings=True,
        got_alternative_branch_promote_on_fail_enabled=True,
    )
    graph = IdeaDag(root_title="root", root_details={"mandate": _MANDATE})
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    primary = graph.add_child(parent.node_id, _GOAL, details=_leaf(
        IdeaActionType.VISIT.value,
        **{DetailKey.GOAL.value: _GOAL, "optional_url": "https://a.example/x"}))
    sibling = graph.add_child(parent.node_id, "Other fact", details=_leaf(
        IdeaActionType.VISIT.value,
        **{DetailKey.GOAL.value: "other", "optional_url": "https://b.example/y"}))
    fallback = graph.add_child(parent.node_id, "Fallback route", details=_leaf(
        IdeaActionType.VISIT.value,
        **{DetailKey.GOAL.value: _GOAL, alt.ALTERNATIVE_OF_HINT: _GOAL,
           "optional_url": "https://c.example/z"}))
    alt.link_alternatives([primary, sibling, fallback])

    ScriptedAction.RESULTS = {
        primary.node_id: {"action": "visit", "success": False, "error": "404"},
        sibling.node_id: _visit_result(_DATUM_PAGE),
    }
    await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    assert primary.status == IdeaNodeStatus.FAILED
    assert sibling.status == IdeaNodeStatus.DONE
    assert fallback.status == IdeaNodeStatus.PENDING
    assert fallback.details[alt.ALTERNATIVE_TRIGGER_REASON] == "primary_failed"


# ======================================================================================
# layer 5 — the new independence rule
# ======================================================================================


def test_race_group_pairing_is_independent_even_when_no_heuristic_matches():
    """A realistic race pairs (say) a SEARCH with a THINK, which matches none of the four
    URL/query heuristics — the group would wrongly serialize without this rule."""
    graph, _p, (a, b) = _graph_with_children(
        ("Route A", _leaf(IdeaActionType.SEARCH.value, **{
            DetailKey.QUERY.value: "hardanger bridge span",
            DetailKey.RACE_GROUP.value: "span"})),
        ("Route B", _leaf(IdeaActionType.THINK.value, **{
            DetailKey.PROMPT.value: "recall the span",
            DetailKey.RACE_GROUP.value: "span"})),
    )
    assert siblings_are_independent(
        graph, [a.node_id, b.node_id], graph.get_node(graph.root_id()), _LOGGER,
    ) == (True, "race_group")


def test_a_mixed_batch_is_not_waved_through_on_the_racers_evidence():
    graph, _p, (a, b, c) = _graph_with_children(
        ("Route A", _leaf(IdeaActionType.SEARCH.value, **{
            DetailKey.QUERY.value: "q", DetailKey.RACE_GROUP.value: "span"})),
        ("Route B", _leaf(IdeaActionType.THINK.value, **{
            DetailKey.PROMPT.value: "p", DetailKey.RACE_GROUP.value: "span"})),
        ("Unrelated", _leaf(IdeaActionType.THINK.value, **{DetailKey.PROMPT.value: "p2"})),
    )
    independent, reason = siblings_are_independent(
        graph, [a.node_id, b.node_id, c.node_id], graph.get_node(graph.root_id()), _LOGGER,
    )
    assert (independent, reason) == (False, "no_independence_evidence")


def test_a_state_dependency_still_beats_a_race_label():
    """The label is an authoring hint; a dependency is a fact about dispatch."""
    graph, _p, (search, visit) = _graph_with_children(
        ("Find URLs", _leaf(IdeaActionType.SEARCH.value, **{
            DetailKey.QUERY.value: "q", DetailKey.RACE_GROUP.value: "span"})),
        ("Open one", _leaf(IdeaActionType.VISIT.value, **{
            DetailKey.RACE_GROUP.value: "span"})),
    )
    independent, reason = siblings_are_independent(
        graph, [search.node_id, visit.node_id], graph.get_node(graph.root_id()), _LOGGER,
    )
    assert (independent, reason) == (False, "state_dependency")


def test_unlabelled_siblings_keep_their_previous_verdict():
    graph, _p, (a, b) = _graph_with_children(
        ("Route A", _leaf(IdeaActionType.SEARCH.value, **{DetailKey.QUERY.value: "one"})),
        ("Route B", _leaf(IdeaActionType.SEARCH.value, **{DetailKey.QUERY.value: "two"})),
    )
    assert siblings_are_independent(
        graph, [a.node_id, b.node_id], graph.get_node(graph.root_id()), _LOGGER,
    ) == (True, "disjoint_searches")


# ======================================================================================
# layer 6 — the winner-selection chain
# ======================================================================================


def _race_graph(*members, mandate=_MANDATE):
    """``members`` are ``(status, content_or_None, completed_step)`` triples."""
    graph = IdeaDag(root_title="root", root_details={"mandate": mandate})
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    nodes = []
    for index, (status, content, step) in enumerate(members):
        details = _leaf(IdeaActionType.VISIT.value, **{
            DetailKey.GOAL.value: _GOAL, DetailKey.RACE_GROUP.value: "span"})
        if content is not None:
            details[DetailKey.ACTION_RESULT.value] = _visit_result(content)
        if step is not None:
            details[alt.RACE_COMPLETED_STEP] = step
        node = graph.add_child(parent.node_id, f"Route {index}", details=details)
        node.status = status
        nodes.append(node)
    alt.link_race_groups(parent, nodes)
    return graph, parent, nodes


def _policy(enabled=True):
    return SimpleMergePolicy(settings={"merge_race_winner_selection_enabled": enabled})


def test_step1_datum_verified_member_wins_outright():
    graph, parent, (a, b) = _race_graph(
        (IdeaNodeStatus.DONE, _SUBJECT_ONLY_PAGE, 1),
        (IdeaNodeStatus.DONE, _DATUM_PAGE, 4),
    )
    # b completed LATER, and still wins: verifying the datum beats finishing first.
    assert _policy().select_winner(graph, parent.node_id, "span") == b.node_id


def test_step2_earliest_completion_breaks_a_datum_verified_tie():
    graph, parent, (a, b) = _race_graph(
        (IdeaNodeStatus.DONE, _DATUM_PAGE, 5),
        (IdeaNodeStatus.DONE, _DATUM_PAGE, 2),
    )
    assert _policy().select_winner(graph, parent.node_id, "span") == b.node_id


def test_step2_falls_back_to_declared_order_within_one_batch():
    """Every member of a parallel batch shares a step index — the graph records no finer
    completion signal, so declared order decides."""
    graph, parent, (a, b) = _race_graph(
        (IdeaNodeStatus.DONE, _DATUM_PAGE, 3),
        (IdeaNodeStatus.DONE, _DATUM_PAGE, 3),
    )
    assert _policy().select_winner(graph, parent.node_id, "span") == a.node_id


def test_step3_any_done_beats_failed_and_blocked():
    graph, parent, (a, b, c) = _race_graph(
        (IdeaNodeStatus.FAILED, None, None),
        (IdeaNodeStatus.DONE, _SUBJECT_ONLY_PAGE, 2),
        (IdeaNodeStatus.BLOCKED, None, None),
    )
    assert _policy().select_winner(graph, parent.node_id, "span") == b.node_id


def test_step4_no_usable_member_returns_none():
    graph, parent, _nodes = _race_graph(
        (IdeaNodeStatus.FAILED, None, None),
        (IdeaNodeStatus.FAILED, None, None),
    )
    assert _policy().select_winner(graph, parent.node_id, "span") is None


def test_step4_no_winner_leaves_aggregate_everything_behavior_intact():
    graph, parent, (a, b) = _race_graph(
        (IdeaNodeStatus.FAILED, None, None),
        (IdeaNodeStatus.FAILED, None, None),
    )
    merged = _policy().merge(graph, parent.node_id, recursive=False)["merged"]
    assert {item["node_id"] for item in merged} == {a.node_id, b.node_id}


def test_merge_replaces_the_group_with_its_winner_and_skips_the_losers():
    graph, parent, (a, b) = _race_graph(
        (IdeaNodeStatus.DONE, _SUBJECT_ONLY_PAGE, 1),
        (IdeaNodeStatus.DONE, _DATUM_PAGE, 2),
    )
    merged = _policy().merge(graph, parent.node_id, recursive=False)["merged"]
    assert [item["node_id"] for item in merged] == [b.node_id]
    assert a.status == IdeaNodeStatus.SKIPPED
    assert a.details[alt.RACE_LOSER] is True
    assert b.status == IdeaNodeStatus.DONE


def test_flag_off_merge_aggregates_every_route_exactly_as_before():
    graph, parent, (a, b) = _race_graph(
        (IdeaNodeStatus.DONE, _SUBJECT_ONLY_PAGE, 1),
        (IdeaNodeStatus.DONE, _DATUM_PAGE, 2),
    )
    merged = _policy(enabled=False).merge(graph, parent.node_id, recursive=False)["merged"]
    assert {item["node_id"] for item in merged} == {a.node_id, b.node_id}
    assert a.status == IdeaNodeStatus.DONE


def test_non_race_siblings_are_unaffected_by_winner_selection():
    graph, parent, (a, b) = _race_graph(
        (IdeaNodeStatus.DONE, _SUBJECT_ONLY_PAGE, 1),
        (IdeaNodeStatus.DONE, _DATUM_PAGE, 2),
    )
    plain = graph.add_child(parent.node_id, "Unrelated fact", details=_leaf(
        IdeaActionType.VISIT.value, **{
            DetailKey.ACTION_RESULT.value: _visit_result("Some other page.")}))
    plain.status = IdeaNodeStatus.DONE

    merged = _policy().merge(graph, parent.node_id, recursive=False)["merged"]
    assert {item["node_id"] for item in merged} == {b.node_id, plain.node_id}
    assert plain.status == IdeaNodeStatus.DONE


def test_select_winner_is_a_noop_without_an_ancestor_registry():
    graph = IdeaDag(root_title="root", root_details={"mandate": _MANDATE})
    parent = graph.add_child(graph.root_id(), "parent goal", details={})
    graph.add_child(parent.node_id, "Route", details=_leaf())
    assert _policy().select_winner(graph, parent.node_id, "span") is None


@pytest.mark.asyncio
async def test_engine_stamps_a_completion_step_only_on_race_members():
    engine = _make_engine()
    graph, parent, (a, b) = _race_graph(
        (IdeaNodeStatus.PENDING, _DATUM_PAGE, None),
        (IdeaNodeStatus.PENDING, _DATUM_PAGE, None),
    )
    plain = graph.add_child(parent.node_id, "Unrelated", details=_leaf(
        IdeaActionType.VISIT.value,
        **{DetailKey.ACTION_RESULT.value: _visit_result(_DATUM_PAGE)}))

    for node in (a, plain):
        await engine._apply_action_result(graph, node.node_id, 7)

    assert a.details[alt.RACE_COMPLETED_STEP] == 7
    assert alt.RACE_COMPLETED_STEP not in plain.details
