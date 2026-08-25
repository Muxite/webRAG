"""A deficit noticed deep in a branch can only be answered at the root today.

``inject_coverage_visits`` mints corrective work for an enumerated candidate that no successful
visit covers -- but only where the coverage gate is consulted, and only as a child of the ROOT.
So a branch that finishes and leaves an entity unsupported has to wait for a root-level gate pass,
and the corrective pair lands outside the branch that raised the deficit.

``inject_ledger_deficit_followup`` generalises the same deterministic pattern to every node
completion, sourcing the unresolved set from the run's ``TaskLedger`` and attaching the
SEARCH+VISIT pair to the COMPLETING node.

Gated by ``run_policy.deficit_driven_injection`` AND ``ledger_mode == "observe"`` (both off by
default). The differential test below is the load-bearing one: with the flag off the graph must be
structurally identical to a run without the mechanism at all.

No network: graphs are hand-built and the engine is constructed without I/O.
"""
from __future__ import annotations

import asyncio
import logging

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.action_constants import NodeDetailsExtractor
from agent.app.idea_policies.base import DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.post_expansion_hooks import inject_ledger_deficit_followup
from agent.app.task_ledger import TaskLedger


LOG = logging.getLogger("test")
MANDATE = (
    "For each of the following, find the year of first ascent:\n"
    "1. Mount Everest\n2. Aconcagua\n"
)


def _graph() -> IdeaDag:
    return IdeaDag(root_title="root", root_details={DetailKey.ORIGINAL_GOAL.value: MANDATE})


def _branch(graph: IdeaDag, title: str = "Research branch"):
    """A non-root node that has itself completed -- the trigger point under test."""
    return graph.add_child(
        graph.root_id(), title,
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.QUERY.value: "unrelated background query",
            DetailKey.IS_LEAF.value: True,
            DetailKey.ACTION_RESULT.value: {"action": "search", "success": True, "results": []},
        },
        status=IdeaNodeStatus.DONE,
    )


def _shape(graph: IdeaDag):
    """Structural fingerprint: title, action, status, details and parent POSITION per node, in
    traversal order. Positional because node ids are per-graph UUIDs; details are compared
    verbatim, so a stray marker written into them fails the comparison."""
    order = {n.node_id: i for i, n in enumerate(graph.iter_depth_first())}
    return [
        (
            i, n.title,
            NodeDetailsExtractor.get_action(n.details),
            n.status,
            order.get(n.parent_id),
            {k: v for k, v in (n.details or {}).items()},
        )
        for i, n in enumerate(graph.iter_depth_first())
    ]


def _engine(**overrides) -> IdeaDagEngine:
    settings = dict(load_idea_dag_settings())
    settings.update(overrides)
    engine = IdeaDagEngine.__new__(IdeaDagEngine)
    engine._cfg = IdeaConfig.from_settings(settings)
    engine.settings = settings
    engine._logger = logging.getLogger("test-engine")
    engine.io = None
    engine._task_ledger = None
    return engine


# --- golden trajectory ---------------------------------------------------------------------

def test_a_corrective_pair_is_injected_under_the_completing_node():
    graph = _graph()
    branch = _branch(graph)

    injected = inject_ledger_deficit_followup(
        graph, branch.node_id, 0, LOG,
        unresolved_entities=["Mount Everest"], mandate=MANDATE,
    )

    assert injected == 1
    children = [graph.get_node(c) for c in branch.children]
    actions = [NodeDetailsExtractor.get_action(c.details) for c in children]
    assert actions == [IdeaActionType.SEARCH.value, IdeaActionType.VISIT.value]

    search_node, visit_node = children
    assert "Mount Everest" in search_node.details[DetailKey.QUERY.value]
    requires = visit_node.details.get(DetailKey.REQUIRES_DATA.value) or {}
    assert requires.get("source_node_id") == search_node.node_id
    # The generalisation under test: attached where the deficit was noticed, NOT at the root.
    assert visit_node.parent_id == branch.node_id
    assert visit_node.parent_id != graph.root_id()


def test_every_unresolved_entity_gets_its_own_pair():
    graph = _graph()
    branch = _branch(graph)

    injected = inject_ledger_deficit_followup(
        graph, branch.node_id, 0, LOG,
        unresolved_entities=["Mount Everest", "Aconcagua"], mandate=MANDATE,
    )

    assert injected == 2
    visits = [n for n in graph.iter_depth_first()
              if NodeDetailsExtractor.get_action(n.details) == IdeaActionType.VISIT.value]
    assert len(visits) == 2
    assert all(v.parent_id == branch.node_id for v in visits)


def test_a_completed_search_for_the_entity_is_reused_instead_of_duplicated():
    """Same economy as ``inject_coverage_visits``: pay for the visit, not a second search."""
    graph = _graph()
    existing = graph.add_child(
        graph.root_id(), "Search Aconcagua first ascent",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.QUERY.value: "aconcagua first ascent",
            DetailKey.ACTION_RESULT.value: {
                "action": "search", "success": True,
                "results": [{"url": "https://example.org/aconcagua"}],
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    branch = _branch(graph)

    assert inject_ledger_deficit_followup(
        graph, branch.node_id, 0, LOG, unresolved_entities=["aconcagua"], mandate=MANDATE,
    ) == 1
    children = [graph.get_node(c) for c in branch.children]
    assert [NodeDetailsExtractor.get_action(c.details) for c in children] == \
        [IdeaActionType.VISIT.value]
    assert children[0].details[DetailKey.REQUIRES_DATA.value]["source_node_id"] == existing.node_id


def test_no_unresolved_entities_is_a_no_op():
    graph = _graph()
    branch = _branch(graph)
    before = _shape(graph)
    assert inject_ledger_deficit_followup(graph, branch.node_id, 0, LOG,
                                          unresolved_entities=[]) == 0
    assert _shape(graph) == before


def test_a_missing_node_is_handled():
    graph = _graph()
    assert inject_ledger_deficit_followup(graph, "no-such-node", 0, LOG,
                                          unresolved_entities=["Mount Everest"]) == 0


# --- dedup ---------------------------------------------------------------------------------

def test_an_entity_a_pending_visit_already_targets_is_not_re_injected():
    """In-flight elsewhere in the graph: the ledger calls it unresolved until that visit
    SUCCEEDS, and re-minting in between is how a remediation loop starts."""
    graph = _graph()
    graph.add_child(
        graph.root_id(), "Visit a page about Mount Everest",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.IS_LEAF.value: True,
        },
        status=IdeaNodeStatus.PENDING,
    )
    branch = _branch(graph)
    count_before = graph.node_count()

    assert inject_ledger_deficit_followup(
        graph, branch.node_id, 0, LOG, unresolved_entities=["Mount Everest"], mandate=MANDATE,
    ) == 0
    assert graph.node_count() == count_before
    assert branch.children == []


def test_an_entity_a_finished_visit_already_targets_is_not_re_injected():
    graph = _graph()
    graph.add_child(
        graph.root_id(), "Visit a page about Aconcagua",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {"action": "visit", "success": False},
        },
        status=IdeaNodeStatus.DONE,
    )
    branch = _branch(graph)
    count_before = graph.node_count()

    assert inject_ledger_deficit_followup(
        graph, branch.node_id, 0, LOG, unresolved_entities=["Aconcagua"], mandate=MANDATE,
    ) == 0
    assert graph.node_count() == count_before


def test_a_second_completion_of_the_same_node_injects_nothing():
    graph = _graph()
    branch = _branch(graph)
    assert inject_ledger_deficit_followup(
        graph, branch.node_id, 0, LOG, unresolved_entities=["Mount Everest"], mandate=MANDATE,
    ) == 1
    after_first = _shape(graph)
    assert inject_ledger_deficit_followup(
        graph, branch.node_id, 1, LOG, unresolved_entities=["Mount Everest"], mandate=MANDATE,
    ) == 0
    assert _shape(graph) == after_first


def test_a_later_node_does_not_duplicate_an_earlier_nodes_injection():
    """Dedup is graph-wide, not node-local: the first branch's pair covers the entity."""
    graph = _graph()
    first = _branch(graph, "Branch one")
    second = _branch(graph, "Branch two")

    assert inject_ledger_deficit_followup(
        graph, first.node_id, 0, LOG, unresolved_entities=["Mount Everest"], mandate=MANDATE,
    ) == 1
    assert inject_ledger_deficit_followup(
        graph, second.node_id, 1, LOG, unresolved_entities=["Mount Everest"], mandate=MANDATE,
    ) == 0
    assert second.children == []


# --- cap -----------------------------------------------------------------------------------

def test_injections_stop_at_the_run_budget():
    """The trigger is per node completion, so the budget has to be run-scoped."""
    graph = _graph()
    entities = [f"Institution {i}" for i in range(6)]
    injected = 0
    for i, entity in enumerate(entities):
        branch = _branch(graph, f"Branch {i}")
        injected += inject_ledger_deficit_followup(
            graph, branch.node_id, i, LOG, unresolved_entities=[entity], max_injections=2,
        )
    assert injected == 2
    visits = [n for n in graph.iter_depth_first()
              if NodeDetailsExtractor.get_action(n.details) == IdeaActionType.VISIT.value]
    assert len(visits) == 2


def test_the_budget_bounds_one_completion_too():
    graph = _graph()
    branch = _branch(graph)
    assert inject_ledger_deficit_followup(
        graph, branch.node_id, 0, LOG,
        unresolved_entities=["Mount Everest", "Aconcagua", "Denali"], max_injections=2,
    ) == 2


def test_a_zero_budget_injects_nothing():
    graph = _graph()
    branch = _branch(graph)
    before = _shape(graph)
    assert inject_ledger_deficit_followup(
        graph, branch.node_id, 0, LOG, unresolved_entities=["Mount Everest"], max_injections=0,
    ) == 0
    assert _shape(graph) == before


# --- differential: flag off must be byte-identical ------------------------------------------

def _scenario():
    graph = _graph()
    branch = _branch(graph)
    return graph, branch


def test_flag_off_is_structurally_identical_to_not_having_the_mechanism():
    off_graph, off_branch = _scenario()
    control_graph, _ = _scenario()  # never touched by remediation at all

    engine = _engine()
    assert engine._cfg.run_policy.deficit_driven_injection is False
    engine._task_ledger = TaskLedger.compile(MANDATE, None, off_graph)
    assert engine._maybe_inject_ledger_deficit_followup(off_graph, off_branch.node_id, 0) == 0

    assert off_graph.node_count() == control_graph.node_count()
    # Same nodes, same order, same actions, same statuses, same details, same parents.
    assert _shape(off_graph) == _shape(control_graph)
    # Spelled out for the two nodes the mechanism would touch: no marker on the completing node,
    # no budget counter on the root.
    control_branch = list(control_graph.iter_depth_first())[1]
    assert off_branch.details == control_branch.details
    assert off_graph.get_node(off_graph.root_id()).details == \
        control_graph.get_node(control_graph.root_id()).details


def test_flag_on_injects_where_flag_off_did_not():
    on_graph, on_branch = _scenario()
    off_graph, off_branch = _scenario()

    off_engine = _engine()
    off_engine._task_ledger = TaskLedger.compile(MANDATE, None, off_graph)
    off_engine._maybe_inject_ledger_deficit_followup(off_graph, off_branch.node_id, 0)

    on_engine = _engine(run_policy_deficit_driven_injection=True,
                        run_policy_ledger_mode="observe")
    on_engine._task_ledger = TaskLedger.compile(MANDATE, None, on_graph)
    injected = on_engine._maybe_inject_ledger_deficit_followup(on_graph, on_branch.node_id, 0)

    assert injected == 2  # both mandate entities are unresolved
    assert on_graph.node_count() == off_graph.node_count() + 4
    assert all(on_graph.get_node(c).parent_id == on_branch.node_id for c in on_branch.children)


# --- dependency: the ledger has to be observing ----------------------------------------------

def test_flag_on_with_the_ledger_off_is_a_graceful_no_op():
    engine = _engine(run_policy_deficit_driven_injection=True)
    assert engine._cfg.run_policy.ledger_mode == "off"
    graph, branch = _scenario()
    before = _shape(graph)
    assert engine._maybe_inject_ledger_deficit_followup(graph, branch.node_id, 0) == 0
    assert _shape(graph) == before


def test_flag_on_with_no_compiled_ledger_is_a_graceful_no_op():
    engine = _engine(run_policy_deficit_driven_injection=True,
                     run_policy_ledger_mode="observe")
    engine._task_ledger = None
    graph, branch = _scenario()
    assert engine._maybe_inject_ledger_deficit_followup(graph, branch.node_id, 0) == 0


def test_remediation_never_crashes_the_run():
    """A broken graph is a warning, not a failed run -- same contract as the other injectors."""

    class _Exploding:
        def get_node(self, _node_id):
            raise RuntimeError("boom")

    class _Ledger:
        mandate = MANDATE

        def refresh(self, _graph):
            raise RuntimeError("boom")

    engine = _engine(run_policy_deficit_driven_injection=True,
                     run_policy_ledger_mode="observe")
    engine._task_ledger = _Ledger()
    assert engine._maybe_inject_ledger_deficit_followup(_Exploding(), "x", 0) == 0


def test_the_completion_point_reaches_the_injector():
    """Wired into `_apply_action_result`, not just callable in isolation."""
    graph, branch = _scenario()
    engine = _engine(run_policy_deficit_driven_injection=True,
                     run_policy_ledger_mode="observe")
    engine._task_ledger = TaskLedger.compile(MANDATE, None, graph)
    engine._handle_action_result = lambda g, n, s: "done"
    engine._record_race_completion = lambda n, s: None
    engine._maybe_promote_alternative_branch = lambda g, n: False

    async def _noop(*args, **kwargs):
        return []

    engine._maybe_plan_library_reexpand = _noop
    engine._maybe_reexpand_fallback_parent = _noop
    engine._maybe_judge_step_confidence = _noop
    engine._maybe_contract_reexpand_batch = _noop
    engine._maybe_confidence_reexpand_batch = _noop
    engine._maybe_reexpand_leaf = _noop

    asyncio.run(engine._apply_action_result(graph, branch.node_id, 0))
    # root + branch + a search/visit pair per unresolved mandate entity.
    assert graph.node_count() == 6
