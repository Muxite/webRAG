"""``max_branching`` capped the graph below the size of the question.

On the four-way baseline the native DAG's mean node count was **4.6** against a
``max_branching`` of 5 -- the cap was not shaping the fan-out, it *was* the graph. A mandate
enumerating seven candidates was answered by a structure with fewer nodes than the question had
parts, and since each candidate generally needs two nodes (search, then visit), the reachable
ceiling was roughly two or three of the seven.

Raising the constant unconditionally is the wrong fix: ``max_branching`` is a global budget
shared by every task shape, so a blanket raise risks regressing chain and narrow tasks that
have no use for a wide root. The cap is instead made *demand-driven*: when the mandate
enumerates N candidates -- the same enumeration ``candidate_coverage`` already parses, and
which fails open below two names -- the ROOT may fan out to N, bounded by an explicit ceiling.
Every non-root node, and every non-enumerated mandate, keeps the flat default.

Opt-in (``breadth_aware_branching_enabled``), so it can be A/B'd against the current baseline.

No network: caps are computed from hand-built graphs.
"""
from __future__ import annotations

import logging

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.config import IdeaConfig


SEVEN = (
    "For each of the following seven summits, find the year of its first ascent:\n"
    "1. Mount Everest\n2. Aconcagua\n3. Denali\n4. Kilimanjaro\n"
    "5. Mount Elbrus\n6. Vinson Massif\n7. Puncak Jaya\n"
    "Then report which was climbed most recently."
)
CHAIN = (
    "Find the engineer who designed the Pontcysyllte Aqueduct, then find the year "
    "that engineer was born."
)


def _engine(**overrides) -> IdeaDagEngine:
    settings = dict(load_idea_dag_settings())
    settings.update(overrides)
    engine = IdeaDagEngine.__new__(IdeaDagEngine)
    engine._cfg = IdeaConfig.from_settings(settings)
    engine.settings = settings
    engine._logger = logging.getLogger("test-engine")
    return engine


def _graph(mandate: str) -> IdeaDag:
    from agent.app.idea_policies.base import DetailKey
    return IdeaDag(root_title="root", root_details={DetailKey.ORIGINAL_GOAL.value: mandate})


def test_flag_is_off_by_default():
    assert _engine()._cfg.engine.breadth_aware_branching_enabled is False


def test_disabled_keeps_the_flat_cap_on_an_enumerated_mandate():
    """The control: with the flag off, a seven-way question still gets a five-wide budget."""
    engine = _engine(breadth_aware_branching_enabled=False, max_branching=5)
    graph = _graph(SEVEN)
    assert engine._effective_branching(graph, graph.root_id(), 5) == 5


def test_enabled_widens_the_root_to_the_enumerated_count():
    engine = _engine(breadth_aware_branching_enabled=True, max_branching=5)
    graph = _graph(SEVEN)
    assert engine._effective_branching(graph, graph.root_id(), 5) == 7


def test_a_non_enumerated_mandate_keeps_the_flat_cap():
    """Chain tasks must not inherit a wide budget they have no use for."""
    engine = _engine(breadth_aware_branching_enabled=True, max_branching=5)
    graph = _graph(CHAIN)
    assert engine._effective_branching(graph, graph.root_id(), 5) == 5


def test_non_root_nodes_keep_the_flat_cap():
    """The widening is about the top-level fan-out, not about every node in the tree."""
    engine = _engine(breadth_aware_branching_enabled=True, max_branching=5)
    graph = _graph(SEVEN)
    child = graph.add_child(graph.root_id(), "sub-goal")
    assert engine._effective_branching(graph, child.node_id, 5) == 5


def test_the_ceiling_bounds_a_pathological_enumeration():
    engine = _engine(
        breadth_aware_branching_enabled=True, max_branching=5, breadth_branching_max=8,
    )
    mandate = "Find the founding year of each:\n" + "\n".join(
        f"{i}. Institution {i}" for i in range(1, 30)
    )
    graph = _graph(mandate)
    assert engine._effective_branching(graph, graph.root_id(), 5) == 8


def test_it_never_narrows_below_the_incoming_width():
    """A dynamic beam that already chose a wider width must not be cut by this."""
    engine = _engine(breadth_aware_branching_enabled=True, max_branching=5)
    graph = _graph(SEVEN)
    assert engine._effective_branching(graph, graph.root_id(), 9) == 9


def test_a_short_enumeration_does_not_narrow_the_default():
    """``extract_named_candidates`` fails open below two names; a 3-way list must not cut 5."""
    engine = _engine(breadth_aware_branching_enabled=True, max_branching=5)
    mandate = "Find the height of each:\n1. Ben Nevis\n2. Snowdon\n3. Scafell Pike\n"
    graph = _graph(mandate)
    assert engine._effective_branching(graph, graph.root_id(), 5) == 5


def test_a_missing_root_goal_falls_back_to_the_flat_cap():
    engine = _engine(breadth_aware_branching_enabled=True, max_branching=5)
    graph = IdeaDag(root_title="root")
    assert engine._effective_branching(graph, graph.root_id(), 5) == 5


@pytest.mark.parametrize("bad_node", ["missing-id", None])
def test_an_unknown_node_falls_back_to_the_flat_cap(bad_node):
    engine = _engine(breadth_aware_branching_enabled=True, max_branching=5)
    graph = _graph(SEVEN)
    assert engine._effective_branching(graph, bad_node, 5) == 5


def test_the_prompt_child_count_follows_the_widened_width():
    """The prompt's "2-N" instruction and the beam slice must agree.

    If the slice widens to 7 while the prompt still asks for "2-5", the model emits at most
    five candidates and the extra width is never used -- the widening would look like it did
    nothing. If the prompt asks for more than the slice keeps, the surplus is truncated
    silently. Either way the two numbers have to come from the same computation.
    """
    from agent.app.idea_policies.expansion import LlmExpansionPolicy

    settings = dict(load_idea_dag_settings())
    settings["max_branching"] = 5
    policy = LlmExpansionPolicy(io=None, settings=settings)
    graph = _graph(SEVEN)
    node = graph.get_node(graph.root_id())

    widened = policy._build_messages(graph, node, max_children=7)
    assert any("2-7" in m.get("content", "") for m in widened)

    default = policy._build_messages(graph, node)
    assert any("2-5" in m.get("content", "") for m in default)
