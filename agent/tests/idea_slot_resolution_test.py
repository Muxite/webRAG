"""Tests for the resolved-value channel -- ``IdeaDagEngine._resolve_slot`` and its writers.

The gap this closes: nothing in the engine could hand a DISCOVERED value to a node waiting
for it. ``requires_data`` gated WHEN a dependent ran, and ``defer_unresolved_slots`` moved
that moment later, but neither ever rewrote the dependent's own fields -- so a chain hop came
back to the same unscoped sibling-URL scavenging inside ``VisitLeafAction``. A ``slot`` on the
``requires_data`` record names the field to fill; the engine fills it at dispatch from the
source's ``waypoint`` (primary) or the source contract's ``value_for`` reader (fallback).

Layers here:

  * the resolution itself: waypoint first, contract fallback, and every fail-toward-silence
    branch (no slot, no waypoint, source not DONE, already-filled field);
  * BOTH dispatch routes. ``_handle_leaf_node`` and the auto-parallel batch's ``_run_one``
    are separate paths -- the batch one bypasses ``_has_required_data`` entirely -- and a
    fan-out of dependent hops is this mechanism's core case, so the batch coverage is not
    optional;
  * the compositions that must not interfere: ``defer_unresolved_slots`` (deferred, then
    resolved on retry) and the join's per-branch waypoint key;
  * a regression lock that ``_has_required_data`` never learns to read ``slot`` -- widening
    the readiness condition is explicitly prohibited by the design doc (it would serialize
    batches whose dependency is already satisfied).
"""
from __future__ import annotations

import copy

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies.actions import LeafAction
from agent.app.idea_policies.base import (
    DecompositionPolicy,
    DetailKey,
    EvaluationPolicy,
    ExpansionPolicy,
    IdeaActionType,
    IdeaNodeStatus,
)
from agent.app.idea_policies.data_contracts import (
    CHUNK_FROM_VISIT,
    URLS_FROM_SEARCH,
    URLS_FROM_VISIT,
    URL_FROM_THINK,
)

_SOURCE_URL = "https://en.wikipedia.org/wiki/Kongur_Tagh"
_OTHER_URL = "https://en.wikipedia.org/wiki/Muztagh_Ata"
_WAYPOINT_URL = "https://en.wikipedia.org/wiki/John_A._Roebling"


# ------------------------------------------------------------------------------------------
# harness
# ------------------------------------------------------------------------------------------


class DummyIO:
    def set_telemetry(self, telemetry):
        return None


class FakeExpansion(ExpansionPolicy):
    async def expand(self, graph: IdeaDag, node_id: str, memories=None):
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


class RecordingVisitAction(LeafAction):
    """Records the URL fields as they stood AT DISPATCH -- which is what proves the channel
    fired before the action ran, rather than after it (or not at all)."""

    SEEN: list = []

    async def execute(self, graph, node_id, io):
        node = graph.get_node(node_id)
        RecordingVisitAction.SEEN.append(
            {
                "node_id": node_id,
                "url": node.details.get(DetailKey.URL.value),
                "optional_url": node.details.get("optional_url"),
            }
        )
        url = node.details.get(DetailKey.URL.value) or "https://example.invalid/scavenged"
        return {
            "action": IdeaActionType.VISIT.value,
            "success": True,
            "url": url,
            "content": "page body",
            "content_full": "page body",
        }


class FakeRegistry:
    def __init__(self, settings):
        self.settings = dict(settings or {})

    def get(self, action_type):
        return RecordingVisitAction(settings=self.settings)


def _make_engine(*, channel=True, waypoint=True, defer=False, auto_parallel=False):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_contract_reexpand_enabled": False,
        "got_step_confidence_judge_enabled": False,
        "got_step_confidence_reexpand_enabled": False,
        "got_reexpand_enabled": False,
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": auto_parallel,
        "evaluate_parallel_siblings": False,
        "waypoint_enabled": waypoint,
        "resolved_value_channel_enabled": channel,
        "defer_unresolved_slots": defer,
    }
    RecordingVisitAction.SEEN = []
    return IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        expansion=FakeExpansion(settings),
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeRegistry(settings),
    )


def _search_result(*urls):
    return {
        "action": IdeaActionType.SEARCH.value,
        "success": True,
        "results": [{"title": f"r{i}", "url": u} for i, u in enumerate(urls)],
    }


def _anchor_waypoint(url=_WAYPOINT_URL):
    return {
        "hop_goal": "who engineered it", "value": "John A. Roebling", "value_kind": "anchor",
        "evidence": "...", "source_url": _SOURCE_URL, "source_title": "Bridge",
        "node_id": "src", "step": 0, "method": "anchor_text", "value_url": url,
    }


def _cue_waypoint():
    """A quantity waypoint: a number, so it carries NO url for a ``url`` slot to take."""
    return {
        "hop_goal": "main span", "value": "1,057", "value_kind": "span", "evidence": "...",
        "source_url": _SOURCE_URL, "source_title": "Bridge", "node_id": "src", "step": 0,
        "method": "cue_window", "value_url": None,
    }


def _graph_with_source_and_dependent(
    *, source_status=IdeaNodeStatus.DONE, source_result=None, source_waypoint=None,
    dependent_details=None, slot="url", contract="urls_from_search",
):
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "parent goal")
    source = graph.add_child(
        parent.node_id, "search for the bridge",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.IS_LEAF.value: True,
        },
        status=source_status,
    )
    if source_result is not None:
        source.details[DetailKey.ACTION_RESULT.value] = source_result
    if source_waypoint is not None:
        source.details[DetailKey.WAYPOINT.value] = source_waypoint

    requires_data = {"type": contract, "source_node_id": source.node_id}
    if slot is not None:
        requires_data["slot"] = slot
    details = {
        DetailKey.ACTION.value: IdeaActionType.VISIT.value,
        DetailKey.IS_LEAF.value: True,
        "link_idea": "the engineer's page",
        DetailKey.REQUIRES_DATA.value: requires_data,
    }
    details.update(dependent_details or {})
    dependent = graph.add_child(parent.node_id, "open the engineer's page", details=details)
    return graph, parent, source, dependent


# ------------------------------------------------------------------------------------------
# layer 1 -- the resolution itself
# ------------------------------------------------------------------------------------------


def test_waypoint_value_url_fills_both_url_and_optional_url():
    """``optional_url`` is not decoration: ``VisitLeafAction`` reads it FIRST and only stands
    its own scavenging chain down when it already holds a valid URL."""
    engine = _make_engine()
    graph, _, source, dependent = _graph_with_source_and_dependent(
        source_result=_search_result(_OTHER_URL), source_waypoint=_anchor_waypoint(),
    )

    assert engine._resolve_slot(graph, dependent) == _WAYPOINT_URL
    assert dependent.details[DetailKey.URL.value] == _WAYPOINT_URL
    assert dependent.details["optional_url"] == _WAYPOINT_URL


def test_falls_back_to_the_contract_getter_when_the_waypoint_has_no_url_for_this_slot():
    """A ``cue_window`` waypoint is a NUMBER -- it can never fill a ``url`` slot, so the
    contract's own structured output takes over rather than the whole resolution failing."""
    engine = _make_engine()
    graph, _, source, dependent = _graph_with_source_and_dependent(
        source_result=_search_result(_SOURCE_URL, _OTHER_URL), source_waypoint=_cue_waypoint(),
    )

    assert engine._resolve_slot(graph, dependent) == _SOURCE_URL
    assert dependent.details["optional_url"] == _SOURCE_URL


def test_no_waypoint_at_all_still_resolves_from_the_contract():
    engine = _make_engine(waypoint=False)
    graph, _, _, dependent = _graph_with_source_and_dependent(
        source_result=_search_result(_SOURCE_URL),
    )

    assert engine._resolve_slot(graph, dependent) == _SOURCE_URL


def test_no_op_when_the_source_produced_nothing_usable(caplog):
    """channel ON + waypoint OFF + a source with no usable output: the node is left exactly as
    it was (today's fallback still owns it), and the miss is LOGGED rather than silent --
    ``dataflow.unresolved_slots`` has no visibility into ``slot`` to flag it for us."""
    engine = _make_engine(waypoint=False)
    graph, _, _, dependent = _graph_with_source_and_dependent(source_result=_search_result())
    before = copy.deepcopy(dependent.details)

    with caplog.at_level("WARNING"):
        assert engine._resolve_slot(graph, dependent) is None

    assert dependent.details == before
    assert any("[SLOT_RESOLVE]" in rec.message for rec in caplog.records)


def test_no_op_when_the_source_is_not_done_yet():
    """Mirrors ``_has_required_data``'s own guard: a source that has not completed has no
    structured output to hand over, and this node will be offered again next dispatch."""
    engine = _make_engine()
    graph, _, source, dependent = _graph_with_source_and_dependent(
        source_status=IdeaNodeStatus.PENDING, source_waypoint=_anchor_waypoint(),
    )
    before = copy.deepcopy(dependent.details)

    assert engine._resolve_slot(graph, dependent) is None
    assert dependent.details == before


def test_no_op_when_the_named_source_is_missing(caplog):
    engine = _make_engine()
    graph, _, _, dependent = _graph_with_source_and_dependent(
        source_waypoint=_anchor_waypoint(),
    )
    dependent.details[DetailKey.REQUIRES_DATA.value]["source_node_id"] = "does-not-exist"
    before = copy.deepcopy(dependent.details)

    with caplog.at_level("WARNING"):
        assert engine._resolve_slot(graph, dependent) is None
    assert dependent.details == before
    assert any("[SLOT_RESOLVE]" in rec.message for rec in caplog.records)


def test_never_overwrites_an_already_authored_url():
    engine = _make_engine()
    graph, _, _, dependent = _graph_with_source_and_dependent(
        source_waypoint=_anchor_waypoint(),
        dependent_details={DetailKey.URL.value: _OTHER_URL},
    )

    assert engine._resolve_slot(graph, dependent) is None
    assert dependent.details[DetailKey.URL.value] == _OTHER_URL
    assert "optional_url" not in dependent.details


def test_a_placeholder_url_does_not_count_as_authored():
    """The exact shape the slot census found executing unrepaired 19 times: a literal
    ``<...>`` template is an unfilled field, not a value worth protecting."""
    engine = _make_engine()
    graph, _, _, dependent = _graph_with_source_and_dependent(
        source_waypoint=_anchor_waypoint(),
        dependent_details={DetailKey.URL.value: "<URL from previous search>"},
    )

    assert engine._resolve_slot(graph, dependent) == _WAYPOINT_URL


@pytest.mark.parametrize("channel", [True, False])
@pytest.mark.parametrize("waypoint", [True, False])
@pytest.mark.parametrize("slot", [None, "__absent__"])
def test_true_no_op_when_no_slot_is_declared(channel, waypoint, slot):
    """Every writer that does NOT declare a slot must be untouched under every flag
    combination -- the mechanism is opt-in per WRITE SITE as well as per run."""
    engine = _make_engine(channel=channel, waypoint=waypoint)
    graph, _, _, dependent = _graph_with_source_and_dependent(
        source_result=_search_result(_SOURCE_URL), source_waypoint=_anchor_waypoint(),
        slot=None,
    )
    if slot is None:
        dependent.details[DetailKey.REQUIRES_DATA.value]["slot"] = None
    before = copy.deepcopy(dependent.details)

    assert engine._resolve_slot(graph, dependent) is None
    assert dependent.details == before


def test_flag_off_is_a_no_op_even_with_a_declared_slot():
    engine = _make_engine(channel=False)
    graph, _, _, dependent = _graph_with_source_and_dependent(
        source_result=_search_result(_SOURCE_URL), source_waypoint=_anchor_waypoint(),
    )
    before = copy.deepcopy(dependent.details)

    assert engine._resolve_slot(graph, dependent) is None
    assert dependent.details == before


def test_resolution_never_raises_on_a_malformed_record():
    engine = _make_engine()
    graph, _, _, dependent = _graph_with_source_and_dependent(
        source_waypoint=_anchor_waypoint(),
    )
    dependent.details[DetailKey.REQUIRES_DATA.value] = "not a dict"
    assert engine._resolve_slot(graph, dependent) is None

    dependent.details[DetailKey.REQUIRES_DATA.value] = {"slot": 17, "source_node_id": None}
    assert engine._resolve_slot(graph, dependent) is None


def test_engine_warns_once_when_the_channel_runs_without_waypoints(caplog):
    with caplog.at_level("WARNING"):
        _make_engine(channel=True, waypoint=False)
    warnings = [r for r in caplog.records if "[SLOT_RESOLVE]" in r.message]
    assert len(warnings) == 1
    assert "waypoint_enabled" in warnings[0].message


def test_no_startup_warning_for_the_supported_combinations(caplog):
    with caplog.at_level("WARNING"):
        _make_engine(channel=True, waypoint=True)
        _make_engine(channel=False, waypoint=False)
        _make_engine(channel=False, waypoint=True)
    assert not [r for r in caplog.records if "[SLOT_RESOLVE]" in r.message]


# ------------------------------------------------------------------------------------------
# layer 2 -- the contract getters
# ------------------------------------------------------------------------------------------


class _StubNode:
    def __init__(self, details):
        self.details = details


def test_contract_getters_read_only_structured_tool_output():
    node = _StubNode({})
    assert URLS_FROM_SEARCH.value_for(_search_result(_SOURCE_URL), node, "url") == _SOURCE_URL
    assert URLS_FROM_VISIT.value_for({"links": [_SOURCE_URL]}, node, "url") == _SOURCE_URL
    assert URLS_FROM_VISIT.value_for(
        {"links_full": [{"href": _SOURCE_URL}]}, node, "url"
    ) == _SOURCE_URL
    assert URL_FROM_THINK.value_for({"extracted_url": _SOURCE_URL}, node, "url") == _SOURCE_URL
    assert URL_FROM_THINK.value_for({}, _StubNode({"url": _SOURCE_URL}), "url") == _SOURCE_URL


def test_contract_getters_decline_non_url_slots_and_junk():
    node = _StubNode({})
    assert URLS_FROM_SEARCH.value_for(_search_result(_SOURCE_URL), node, "query") is None
    assert URLS_FROM_SEARCH.value_for({"results": [{"url": "not-a-url"}]}, node, "url") is None
    assert URLS_FROM_SEARCH.value_for({"results": "garbage"}, node, "url") is None
    assert URLS_FROM_VISIT.value_for({}, node, "url") is None
    # A page's body text carries no value a slot can be filled from deterministically.
    assert CHUNK_FROM_VISIT.value_for is None


# ------------------------------------------------------------------------------------------
# layer 3 -- BOTH dispatch routes
# ------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fires_on_the_sequential_leaf_dispatch_path():
    engine = _make_engine()
    graph, _, _, dependent = _graph_with_source_and_dependent(
        source_result=_search_result(_OTHER_URL), source_waypoint=_anchor_waypoint(),
    )

    await engine._handle_leaf_node(graph, dependent.node_id, 0, None)

    assert dependent.status == IdeaNodeStatus.DONE
    seen = [s for s in RecordingVisitAction.SEEN if s["node_id"] == dependent.node_id]
    assert seen and seen[0]["url"] == _WAYPOINT_URL, "resolved BEFORE the action ran"
    assert seen[0]["optional_url"] == _WAYPOINT_URL


@pytest.mark.asyncio
async def test_fires_on_the_auto_parallel_batch_dispatch_path():
    """The confirmed-critical hook. The batch path calls ``_execute_action`` directly, with no
    ``_has_required_data`` gate and no visit to ``_handle_leaf_node`` at all -- and a fan-out
    of dependent hops is precisely what this channel is for, so a hook only on the sequential
    route would miss the mechanism's own core case."""
    engine = _make_engine(auto_parallel=True)
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "parent goal")

    dependents = []
    for i, url in enumerate((_WAYPOINT_URL, _OTHER_URL)):
        source = graph.add_child(
            parent.node_id, f"search {i}",
            details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
            status=IdeaNodeStatus.DONE,
        )
        source.details[DetailKey.ACTION_RESULT.value] = _search_result(url)
        source.details[DetailKey.WAYPOINT.value] = _anchor_waypoint(url)
        dependents.append(
            graph.add_child(
                parent.node_id, f"open page {i}",
                details={
                    DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                    DetailKey.IS_LEAF.value: True,
                    "link_idea": f"page {i}",
                    DetailKey.REQUIRES_DATA.value: {
                        "type": "urls_from_search",
                        "source_node_id": source.node_id,
                        "slot": "url",
                    },
                },
            )
        )

    await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    assert all(d.status == IdeaNodeStatus.DONE for d in dependents)
    seen = {s["node_id"]: s for s in RecordingVisitAction.SEEN}
    assert len(seen) == 2, "both siblings dispatched through the batch path"
    assert seen[dependents[0].node_id]["url"] == _WAYPOINT_URL
    assert seen[dependents[1].node_id]["url"] == _OTHER_URL
    assert seen[dependents[1].node_id]["optional_url"] == _OTHER_URL


@pytest.mark.asyncio
async def test_batch_path_flag_off_leaves_the_dependents_unresolved():
    """The other half of the batch assertion: without the channel those same siblings reach
    their action with nothing in either URL field, i.e. straight into today's scavenging."""
    engine = _make_engine(channel=False, auto_parallel=True)
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "parent goal")
    for i, url in enumerate((_WAYPOINT_URL, _OTHER_URL)):
        source = graph.add_child(
            parent.node_id, f"search {i}",
            details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
            status=IdeaNodeStatus.DONE,
        )
        source.details[DetailKey.ACTION_RESULT.value] = _search_result(url)
        graph.add_child(
            parent.node_id, f"open page {i}",
            details={
                DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                DetailKey.IS_LEAF.value: True,
                "link_idea": f"page {i}",
                DetailKey.REQUIRES_DATA.value: {
                    "type": "urls_from_search",
                    "source_node_id": source.node_id,
                    "slot": "url",
                },
            },
        )

    await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    assert RecordingVisitAction.SEEN
    assert all(s["url"] is None and s["optional_url"] is None for s in RecordingVisitAction.SEEN)


@pytest.mark.asyncio
async def test_deferral_and_resolution_compose_deferred_first_then_filled():
    """``defer_unresolved_slots`` changes WHEN a node runs; the channel changes WHAT it runs
    on. Together: the placeholder-bearing sibling sits out the batch, and comes back with its
    URL actually filled instead of hitting the same scavenging a second time."""
    engine = _make_engine(defer=True, auto_parallel=True)
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "parent goal")
    source = graph.add_child(
        parent.node_id, "search",
        details={DetailKey.ACTION.value: IdeaActionType.SEARCH.value},
        status=IdeaNodeStatus.DONE,
    )
    source.details[DetailKey.ACTION_RESULT.value] = _search_result(_SOURCE_URL)
    source.details[DetailKey.WAYPOINT.value] = _anchor_waypoint()

    deferred = graph.add_child(
        parent.node_id, "open the engineer's page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.IS_LEAF.value: True,
            "optional_url": "<URL from previous search>",
            DetailKey.REQUIRES_DATA.value: {
                "type": "urls_from_search",
                "source_node_id": source.node_id,
                "slot": "url",
            },
        },
    )
    clean = graph.add_child(
        parent.node_id, "open a known page",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.IS_LEAF.value: True,
            DetailKey.URL.value: _OTHER_URL,
        },
    )

    await engine._handle_intermediate_node(graph, parent.node_id, 0, None)

    assert clean.status == IdeaNodeStatus.DONE
    assert deferred.status != IdeaNodeStatus.DONE, "deferred out of this batch"
    assert [s["node_id"] for s in RecordingVisitAction.SEEN] == [clean.node_id]

    await engine._handle_leaf_node(graph, deferred.node_id, 1, None)

    assert deferred.status == IdeaNodeStatus.DONE
    retried = RecordingVisitAction.SEEN[-1]
    assert retried["node_id"] == deferred.node_id
    assert retried["url"] == _WAYPOINT_URL
    assert retried["optional_url"] == _WAYPOINT_URL, "the placeholder was replaced, not kept"


# ------------------------------------------------------------------------------------------
# layer 4 -- the join
# ------------------------------------------------------------------------------------------


def _ragged_join_graph():
    graph = IdeaDag(root_title="root")
    parent = graph.add_child(graph.root_id(), "join parent")
    done = graph.add_child(
        parent.node_id, "branch A",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {"action": "visit", "success": True},
            DetailKey.WAYPOINT.value: _anchor_waypoint(),
        },
        status=IdeaNodeStatus.DONE,
    )
    failed = graph.add_child(
        parent.node_id, "branch B",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
        status=IdeaNodeStatus.FAILED,
    )
    pending = graph.add_child(
        parent.node_id, "branch C",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value},
        status=IdeaNodeStatus.PENDING,
    )
    return graph, parent, done, failed, pending


@pytest.mark.parametrize("channel", [True, False])
def test_join_carries_each_branchs_waypoint_only_when_the_channel_is_on(channel):
    graph, parent, done, failed, pending = _ragged_join_graph()
    policy = SimpleMergePolicy(settings={"resolved_value_channel_enabled": channel})

    policy.merge(graph, parent.node_id)

    merged = {m["node_id"]: m for m in parent.details[DetailKey.MERGED_RESULTS.value]}
    assert set(merged) == {done.node_id, failed.node_id, pending.node_id}
    assert merged[done.node_id]["waypoint"] == (_anchor_waypoint() if channel else None)
    # A branch that never produced one contributes None either way, not a missing key.
    assert merged[failed.node_id]["waypoint"] is None
    assert merged[pending.node_id]["waypoint"] is None


def test_a_ragged_join_is_still_not_ready_and_does_not_crash():
    """``are_children_ready_to_merge`` decides on STATUS alone and needs no change: a branch
    that has not finished blocks the join whether or not it would have carried a waypoint."""
    graph, parent, _, _, pending = _ragged_join_graph()
    policy = SimpleMergePolicy(settings={"resolved_value_channel_enabled": True})

    assert policy.are_children_ready_to_merge(graph, parent.node_id) is False
    pending.status = IdeaNodeStatus.SKIPPED
    assert policy.are_children_ready_to_merge(graph, parent.node_id) is True
    assert policy.merge(graph, parent.node_id) is not None


# ------------------------------------------------------------------------------------------
# layer 5 -- readiness must not learn about `slot`
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source_status", [IdeaNodeStatus.DONE, IdeaNodeStatus.PENDING])
def test_has_required_data_is_identical_with_and_without_a_slot(source_status):
    """Condition B must NOT widen. ``expansion.py``'s always-in-path writer only ever names
    sources that already ran, so a readiness rule that reacted to ``slot`` would serialize
    batches whose dependency is already satisfied -- explicitly prohibited by the design."""
    engine = _make_engine()
    without = _graph_with_source_and_dependent(
        source_status=source_status, source_result=_search_result(_SOURCE_URL), slot=None,
    )
    with_slot = _graph_with_source_and_dependent(
        source_status=source_status, source_result=_search_result(_SOURCE_URL), slot="url",
    )

    assert engine._has_required_data(without[0], without[3]) == engine._has_required_data(
        with_slot[0], with_slot[3]
    )
    assert engine._has_required_data(with_slot[0], with_slot[3]) is (
        source_status == IdeaNodeStatus.DONE
    )


def test_has_required_data_ignores_an_unresolvable_slot():
    """Even a slot the channel could never fill leaves readiness alone -- resolution failing
    is a logged no-op, never a gate."""
    engine = _make_engine()
    graph, _, _, dependent = _graph_with_source_and_dependent(
        source_result=_search_result(_SOURCE_URL), slot="link_idea",
    )
    assert engine._has_required_data(graph, dependent) is True
