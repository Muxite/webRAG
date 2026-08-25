"""A search that hands the run nothing to open is a dead end nothing notices.

``inject_coverage_visits`` remediates one half of the visit gap: an enumerated candidate that no
successful visit covers. The other half has no owner at all -- a SEARCH completes, its result
list is empty (or every URL in it is already visited elsewhere), and the branch simply ends. The
visit path logs "No URLs extracted from search results" and the run finalizes on whatever text
the search snippets happened to contain.

``inject_empty_search_followup`` closes that with the same deterministic pattern: detect the gap
from the node's own action result, then inject a broadened search plus a visit that ``REQUIRES_
DATA``-depends on it, as siblings of the dead search so the pooled sibling URL resolution can see
the new results.

Gated by ``run_policy.search_must_yield_visit`` (default off). The differential test below is the
load-bearing one: with the flag off the graph must be structurally identical to a run without the
mechanism at all.

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
from agent.app.idea_policies.post_expansion_hooks import (
    inject_empty_search_followup,
    search_yielded_no_visit,
)


LOG = logging.getLogger("test")
MANDATE = (
    "For each of the following, find the year of first ascent:\n"
    "1. Mount Everest\n2. Aconcagua\n"
)


def _graph() -> IdeaDag:
    return IdeaDag(root_title="root", root_details={DetailKey.ORIGINAL_GOAL.value: MANDATE})


def _search(graph: IdeaDag, query: str, results, *, parent_id=None):
    node = graph.add_child(
        parent_id or graph.root_id(), f"Search {query}",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.QUERY.value: query,
            DetailKey.IS_LEAF.value: True,
        },
        status=IdeaNodeStatus.DONE,
    )
    node.details[DetailKey.ACTION_RESULT.value] = {
        "action": "search", "success": True, "query": query, "results": results,
    }
    return node


def _shape(graph: IdeaDag):
    """Structural fingerprint: title, action, status, details and parent POSITION per node,
    in traversal order. Positional because node ids are per-graph UUIDs; everything else is
    compared verbatim, so a stray marker written into details fails the comparison."""
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
    return engine


# --- gap detection -------------------------------------------------------------------------

def test_an_empty_result_list_is_a_gap():
    graph = _graph()
    node = _search(graph, "Mount Everest first ascent official year", [])
    assert search_yielded_no_visit(graph, node) is True


def test_results_with_urls_are_not_a_gap():
    graph = _graph()
    node = _search(graph, "Mount Everest first ascent",
                   [{"url": "https://example.org/everest", "title": "Everest"}])
    assert search_yielded_no_visit(graph, node) is False


def test_results_already_visited_elsewhere_are_a_gap():
    """Non-empty results, but the pool is exhausted: every URL is already opened."""
    graph = _graph()
    graph.add_child(
        graph.root_id(), "Visit Everest",
        details={
            DetailKey.ACTION.value: IdeaActionType.VISIT.value,
            DetailKey.ACTION_RESULT.value: {
                "action": "visit", "success": True,
                "url": "https://example.org/everest", "page_title": "Everest", "content": "x",
            },
        },
        status=IdeaNodeStatus.DONE,
    )
    node = _search(graph, "Mount Everest first ascent recorded year",
                   [{"url": "https://example.org/everest", "title": "Everest"}])
    assert search_yielded_no_visit(graph, node) is True


def test_results_without_usable_urls_are_a_gap():
    graph = _graph()
    node = _search(graph, "Mount Everest first ascent official year",
                   [{"title": "snippet only"}, {"url": "not-a-url"}])
    assert search_yielded_no_visit(graph, node) is True


# --- golden trajectory ---------------------------------------------------------------------

def test_a_corrective_search_and_dependent_visit_are_injected():
    graph = _graph()
    dead = _search(graph, "Mount Everest first ascent official recorded year", [])

    assert inject_empty_search_followup(graph, dead.node_id, 0, LOG) == 1

    new_nodes = [n for n in graph.iter_depth_first()
                 if n.node_id not in (graph.root_id(), dead.node_id)]
    actions = [NodeDetailsExtractor.get_action(n.details) for n in new_nodes]
    assert actions == [IdeaActionType.SEARCH.value, IdeaActionType.VISIT.value]

    search_node, visit_node = new_nodes
    # Broadened, not a re-issue of the query that already failed.
    assert search_node.details[DetailKey.QUERY.value] != dead.details[DetailKey.QUERY.value]
    assert search_node.details[DetailKey.QUERY.value] == "Mount Everest first ascent"
    # The visit is wired to the corrective search: a search must yield a visit.
    requires = visit_node.details.get(DetailKey.REQUIRES_DATA.value) or {}
    assert requires.get("source_node_id") == search_node.node_id
    # Injected as siblings of the dead search so sibling URL pooling can see them.
    assert search_node.parent_id == dead.parent_id
    assert visit_node.parent_id == dead.parent_id


def test_the_ledgers_unresolved_entity_picks_the_corrective_query():
    graph = _graph()
    dead = _search(graph, "Aconcagua first ascent year definitive source", [])

    assert inject_empty_search_followup(
        graph, dead.node_id, 0, LOG, unresolved_entities=["Aconcagua"]
    ) == 1
    injected = [n for n in graph.iter_depth_first()
                if NodeDetailsExtractor.get_action(n.details) == IdeaActionType.SEARCH.value
                and n.node_id != dead.node_id][0]
    assert injected.details[DetailKey.QUERY.value] == "Aconcagua"


def test_it_works_with_no_ledger_at_all():
    """``ledger_mode`` off: the dead search node alone is signal enough."""
    graph = _graph()
    dead = _search(graph, "Aconcagua first ascent year definitive source", [])
    assert inject_empty_search_followup(graph, dead.node_id, 0, LOG, unresolved_entities=None) == 1


def test_a_search_that_did_yield_urls_is_left_alone():
    graph = _graph()
    node = _search(graph, "Mount Everest first ascent",
                   [{"url": "https://example.org/everest"}])
    before = _shape(graph)
    assert inject_empty_search_followup(graph, node.node_id, 0, LOG) == 0
    assert _shape(graph) == before


def test_an_unexecuted_search_is_left_alone():
    graph = _graph()
    node = graph.add_child(
        graph.root_id(), "Search later",
        details={
            DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
            DetailKey.QUERY.value: "Mount Everest first ascent official year",
            DetailKey.IS_LEAF.value: True,
        },
    )
    assert inject_empty_search_followup(graph, node.node_id, 0, LOG) == 0


def test_a_non_search_node_is_left_alone():
    graph = _graph()
    node = graph.add_child(
        graph.root_id(), "Visit something",
        details={DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                 DetailKey.ACTION_RESULT.value: {"action": "visit", "success": False}},
        status=IdeaNodeStatus.DONE,
    )
    assert inject_empty_search_followup(graph, node.node_id, 0, LOG) == 0


def test_a_missing_node_is_handled():
    graph = _graph()
    assert inject_empty_search_followup(graph, "no-such-node", 0, LOG) == 0


def test_an_unbroadenable_query_injects_nothing():
    """Nothing to drop, so a retry would be the identical search -- decline instead of loop."""
    graph = _graph()
    dead = _search(graph, "Everest", [])
    before = _shape(graph)
    assert inject_empty_search_followup(graph, dead.node_id, 0, LOG) == 0
    assert _shape(graph) == before


def test_a_query_the_run_already_ran_is_not_re_issued():
    graph = _graph()
    _search(graph, "Mount Everest first ascent", [{"url": "https://example.org/e"}])
    dead = _search(graph, "Mount Everest first ascent official recorded year", [])
    assert inject_empty_search_followup(graph, dead.node_id, 0, LOG) == 0


def test_the_same_dead_search_is_remediated_once():
    graph = _graph()
    dead = _search(graph, "Mount Everest first ascent official recorded year", [])
    assert inject_empty_search_followup(graph, dead.node_id, 0, LOG) == 1
    after_first = _shape(graph)
    assert inject_empty_search_followup(graph, dead.node_id, 1, LOG) == 0
    assert _shape(graph) == after_first


# --- cap -----------------------------------------------------------------------------------

def test_repeated_dead_searches_stop_at_the_run_budget():
    """The trigger is per completed search, so the budget has to be run-scoped."""
    graph = _graph()
    injected = 0
    for i in range(8):
        dead = _search(graph, f"Institution {i} founding year official public record", [])
        injected += inject_empty_search_followup(graph, dead.node_id, i, LOG, max_injections=2)
    assert injected == 2
    corrective = [n for n in graph.iter_depth_first()
                  if NodeDetailsExtractor.get_action(n.details) == IdeaActionType.VISIT.value]
    assert len(corrective) == 2


def test_a_zero_budget_injects_nothing():
    graph = _graph()
    dead = _search(graph, "Mount Everest first ascent official recorded year", [])
    before = _shape(graph)
    assert inject_empty_search_followup(graph, dead.node_id, 0, LOG, max_injections=0) == 0
    assert _shape(graph) == before


# --- differential: flag off must be byte-identical ------------------------------------------

def _scenario():
    """A graph whose only completed search yielded nothing visitable."""
    graph = _graph()
    dead = _search(graph, "Mount Everest first ascent official recorded year", [])
    return graph, dead


def test_flag_off_is_structurally_identical_to_not_having_the_mechanism():
    off_graph, off_dead = _scenario()
    control_graph, _ = _scenario()  # never touched by remediation at all

    engine = _engine()
    assert engine._cfg.run_policy.search_must_yield_visit is False
    assert engine._maybe_inject_empty_search_followup(off_graph, off_dead.node_id, 0) == 0

    assert off_graph.node_count() == control_graph.node_count()
    # Same nodes, same order, same actions, same statuses, same details, same parents.
    assert _shape(off_graph) == _shape(control_graph)
    # Spelled out for the two nodes the mechanism would touch: no marker on the dead search,
    # no budget counter on the root.
    control_dead = list(control_graph.iter_depth_first())[1]
    assert off_dead.details == control_dead.details
    assert off_graph.get_node(off_graph.root_id()).details == \
        control_graph.get_node(control_graph.root_id()).details


def test_flag_on_injects_where_flag_off_did_not():
    on_graph, on_dead = _scenario()
    off_graph, off_dead = _scenario()

    _engine()._maybe_inject_empty_search_followup(off_graph, off_dead.node_id, 0)
    injected = _engine(run_policy_search_must_yield_visit=True) \
        ._maybe_inject_empty_search_followup(on_graph, on_dead.node_id, 0)

    assert injected == 1
    assert on_graph.node_count() == off_graph.node_count() + 2
    actions = [NodeDetailsExtractor.get_action(n.details) for n in on_graph.iter_depth_first()]
    assert actions.count(IdeaActionType.VISIT.value) == 1


def test_the_engine_falls_back_to_local_signal_when_the_ledger_is_off():
    engine = _engine(run_policy_search_must_yield_visit=True)
    assert engine._cfg.run_policy.ledger_mode == "off"
    graph, dead = _scenario()
    assert engine._maybe_inject_empty_search_followup(graph, dead.node_id, 0) == 1


def test_the_engine_uses_the_ledger_when_it_is_observing():
    from agent.app.task_ledger import TaskLedger

    engine = _engine(run_policy_search_must_yield_visit=True,
                     run_policy_ledger_mode="observe")
    graph = _graph()
    dead = _search(graph, "Aconcagua first ascent year definitive source", [])
    engine._task_ledger = TaskLedger.compile(MANDATE, None, graph)

    assert engine._maybe_inject_empty_search_followup(graph, dead.node_id, 0) == 1
    injected = [n for n in graph.iter_depth_first()
                if NodeDetailsExtractor.get_action(n.details) == IdeaActionType.SEARCH.value
                and n.node_id != dead.node_id][0]
    assert injected.details[DetailKey.QUERY.value] == "Aconcagua"


def test_remediation_never_crashes_the_run():
    """A broken graph is a warning, not a failed run -- same contract as the coverage injector."""

    class _Exploding:
        def get_node(self, _node_id):
            raise RuntimeError("boom")

    engine = _engine(run_policy_search_must_yield_visit=True)
    assert engine._maybe_inject_empty_search_followup(_Exploding(), "x", 0) == 0


def test_the_completion_point_reaches_the_injector():
    """Wired into `_apply_action_result`, not just callable in isolation."""
    graph, dead = _scenario()
    engine = _engine(run_policy_search_must_yield_visit=True)
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

    asyncio.run(engine._apply_action_result(graph, dead.node_id, 0))
    assert graph.node_count() == 4  # root + dead search + corrective search + visit
