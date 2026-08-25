"""The root expanded exactly once, so no remediation path could ever widen the graph.

``step()`` gated expansion on ``if not node.children``. Once the root had children it could
never be expanded again -- and EVERY remediation path in the engine ends by setting
``root.status = ACTIVE``:

* ``_candidate_coverage_extension`` (its own comment says "re-activate the root so the extended
  budget re-expands and re-checks the missing candidates" -- which the code could not do),
* ``_grounding_replan``,
* the two budget-exhaustion branches in ``run()``.

Re-activating a root that already has children falls through to ``_handle_intermediate_node``,
which only picks among nodes that ALREADY EXIST. So the engine's entire "notice we are
incomplete, go get more" machinery re-entered a structure it was architecturally forbidden from
widening.

This is the mechanism behind the coverage A/B's null result (n=24, delta +0.019, p=0.70): with
the gate ON, wide-breadth cells spent 46 searches and made 1 visit. The gate detected the four
missing candidates correctly; the extension granted +10 steps; and those steps were spent
re-walking a graph frozen at its first LLM call.

Re-expansion is additive (``IdeaDag.expand`` appends), bounded by ``root_reexpansion_max``, and
carries the already-covered candidates into the prompt as exclusions -- without which the model
simply re-emits the children it produced the first time.

Opt-in via ``root_reexpansion_enabled``. No network: expansion is stubbed.
"""
from __future__ import annotations

import logging

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies.base import DetailKey
from agent.app.idea_policies.config import IdeaConfig


MANDATE = (
    "For each of the following, find the year of first ascent:\n"
    "1. Mount Everest\n2. Aconcagua\n3. Denali\n4. Kilimanjaro\n"
)


def _engine(**overrides) -> IdeaDagEngine:
    settings = dict(load_idea_dag_settings())
    settings.update(overrides)
    engine = IdeaDagEngine.__new__(IdeaDagEngine)
    engine._cfg = IdeaConfig.from_settings(settings)
    engine.settings = settings
    engine._logger = logging.getLogger("test-engine")
    return engine


def _graph_with_children() -> IdeaDag:
    graph = IdeaDag(root_title="root",
                    root_details={DetailKey.ORIGINAL_GOAL.value: MANDATE})
    graph.add_child(graph.root_id(), "Search Mount Everest first ascent")
    graph.add_child(graph.root_id(), "Visit Mount Everest page")
    return graph


def test_flag_is_off_by_default():
    assert _engine()._cfg.engine.root_reexpansion_enabled is False


def test_disabled_refuses_the_widen_request():
    """The control: current behaviour, where a remediation path cannot widen anything."""
    engine = _engine(root_reexpansion_enabled=False)
    graph = _graph_with_children()
    assert engine._request_root_widen(graph, "coverage") is False
    assert engine._consume_widen_request(graph.get_node(graph.root_id())) is False


def test_enabled_grants_and_then_consumes_the_request():
    engine = _engine(root_reexpansion_enabled=True)
    graph = _graph_with_children()
    root = graph.get_node(graph.root_id())

    assert engine._request_root_widen(graph, "coverage") is True
    assert engine._consume_widen_request(root) is True
    # Single-use: the flag must not re-fire on the next step.
    assert engine._consume_widen_request(root) is False


def test_the_request_is_bounded():
    """A widen that keeps re-granting itself is an infinite expansion loop."""
    engine = _engine(root_reexpansion_enabled=True, root_reexpansion_max=2)
    graph = _graph_with_children()
    root = graph.get_node(graph.root_id())

    granted = 0
    for _ in range(6):
        if engine._request_root_widen(graph, "coverage"):
            granted += 1
            engine._consume_widen_request(root)
    assert granted == 2


def test_re_expansion_is_additive_not_replacing():
    """New children must join the existing ones; the earlier work is not thrown away."""
    graph = _graph_with_children()
    before = list(graph.get_node(graph.root_id()).children)
    graph.expand(graph.root_id(), [{"title": "Search Aconcagua first ascent"}])
    after = graph.get_node(graph.root_id()).children
    assert after[:len(before)] == before
    assert len(after) == len(before) + 1


def test_exclusions_name_the_existing_children():
    """Without exclusions the model re-emits the children it produced the first time."""
    engine = _engine(root_reexpansion_enabled=True)
    graph = _graph_with_children()
    exclusions = engine._reexpansion_exclusions(graph, graph.root_id())
    assert "Search Mount Everest first ascent" in exclusions
    assert "Visit Mount Everest page" in exclusions


def test_exclusions_are_empty_for_a_childless_node():
    engine = _engine(root_reexpansion_enabled=True)
    graph = IdeaDag(root_title="root")
    assert engine._reexpansion_exclusions(graph, graph.root_id()) == []


def test_the_prompt_carries_the_exclusions():
    from agent.app.idea_policies.expansion import LlmExpansionPolicy

    policy = LlmExpansionPolicy(io=None, settings=dict(load_idea_dag_settings()))
    graph = _graph_with_children()
    node = graph.get_node(graph.root_id())

    messages = policy._build_messages(
        graph, node, exclude=["Search Mount Everest first ascent"],
    )
    blob = "\n".join(m.get("content", "") for m in messages)
    assert "Search Mount Everest first ascent" in blob

    plain = policy._build_messages(graph, node)
    plain_blob = "\n".join(m.get("content", "") for m in plain)
    assert "already been covered" not in plain_blob


def test_a_missing_root_is_handled():
    engine = _engine(root_reexpansion_enabled=True)
    graph = IdeaDag(root_title="root")
    graph._nodes.clear()
    assert engine._request_root_widen(graph, "coverage") is False


def test_consume_tolerates_a_none_node():
    engine = _engine(root_reexpansion_enabled=True)
    assert engine._consume_widen_request(None) is False
