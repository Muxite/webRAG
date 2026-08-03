"""Tests for capability-tiered ``got.reexpand_max_iterations`` (Phase 5 of the
capability-continuum plan) — the native-engine port of badmodel-lab's per-page leaf-extraction
retry lever, applied to the native engine's re-expansion budget instead (no per-leaf extraction
call exists to attach a retry to in the native engine; re-expansion is the closest analog).

``_effective_reexpand_max_iterations()`` is the SINGLE method all three budget-check sites
(``_reexpand_check``, ``_apply_reexpand``, ``_reexpand_guards_ok``) must read through — a prior,
real bug (documented in ``_apply_reexpand``) had this exact knob silently inert at one call site
for any value > 1, so this suite explicitly proves the tiered override binds at more than one site,
not just that the dispatch function itself returns the right number.
"""
from __future__ import annotations

import pytest

from agent.app import model_tiers
from agent.app.idea_dag import IdeaDag
from agent.app.idea_engine import IdeaDagEngine
from agent.app.idea_policies import BestScoreSelectionPolicy, SimpleMergePolicy
from agent.app.idea_policies.base import DetailKey, IdeaActionType

from agent.tests.reexpand_leaf_test import (
    DummyIO, FakeExpansion, FakeEvaluation, FakeDecomposition, FakeRegistry,
    _attach_got, _completed_leaf,
)


def _patch_price(monkeypatch, price):
    monkeypatch.setattr(model_tiers.model_costs, "_lookup_pricing",
                        lambda name: {"output_per_million": price})


def _make_engine(model_name, extra=None):
    settings = {
        "allow_unscored_selection": True,
        "min_score_threshold": 0.0,
        "best_first_global": False,
        "got_reexpand_enabled": True,
        "got_reexpand_max_iterations": 1,  # the static default the tiering flag overrides
        "got_dedup_enabled": False,
        "got_embed_on_create": False,
        "auto_parallel_siblings": False,
    }
    if extra:
        settings.update(extra)
    expansion = FakeExpansion(settings)
    engine = IdeaDagEngine(
        io=DummyIO(),
        settings=settings,
        model_name=model_name,
        expansion=expansion,
        evaluation=FakeEvaluation(settings),
        selection=BestScoreSelectionPolicy(settings=settings),
        decomposition=FakeDecomposition(settings),
        merge=SimpleMergePolicy(settings=settings),
        actions=FakeRegistry(settings),
    )
    return engine, expansion


# ---------------------------------------------------------------------------
# _effective_reexpand_max_iterations dispatch
# ---------------------------------------------------------------------------
def test_flag_off_returns_the_static_value_regardless_of_tier(monkeypatch):
    _patch_price(monkeypatch, 0.0)  # weak
    engine, _ = _make_engine("some/unpriced-model", {"got_reexpand_max_iterations": 1})
    assert engine._effective_reexpand_max_iterations() == 1


def test_weak_model_gets_the_weak_band(monkeypatch):
    _patch_price(monkeypatch, 0.0)  # unpriced -> weak
    engine, _ = _make_engine("some/unpriced-model", {
        "got_reexpand_max_iterations_tiered_enabled": True,
        "got_reexpand_max_iterations_weak": 2,
        "got_reexpand_max_iterations_standard": 1,
        "got_reexpand_max_iterations_strong": 1,
    })
    assert engine._effective_reexpand_max_iterations() == 2


def test_strong_model_tapers_to_the_strong_band(monkeypatch):
    _patch_price(monkeypatch, 10.0)  # premium -> strong
    engine, _ = _make_engine("some/premium-model", {
        "got_reexpand_max_iterations_tiered_enabled": True,
        "got_reexpand_max_iterations_weak": 2,
        "got_reexpand_max_iterations_standard": 1,
        "got_reexpand_max_iterations_strong": 1,
    })
    assert engine._effective_reexpand_max_iterations() == 1


# ---------------------------------------------------------------------------
# Binds at _reexpand_guards_ok (the shared-guard site, no LLM call — direct unit test)
# ---------------------------------------------------------------------------
def _leaf_with_seeded_count(engine, count: int):
    graph = IdeaDag(root_title="root")
    root = graph.root_id()
    leaf = graph.add_child(root, "leaf", details={
        DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
        DetailKey.ACTION_RESULT.value: {"success": True},
        "_got_reexpand_count": count,
    })
    return graph, leaf


def test_reexpand_guards_ok_binds_the_tiered_budget_not_just_the_static_one(monkeypatch):
    _patch_price(monkeypatch, 0.0)  # weak -> budget 2
    engine, _ = _make_engine("some/unpriced-model", {
        "got_reexpand_max_iterations_tiered_enabled": True,
        "got_reexpand_max_iterations_weak": 2,
        "got_reexpand_max_iterations_standard": 1,
        "got_reexpand_max_iterations_strong": 1,
    })
    graph, leaf = _leaf_with_seeded_count(engine, count=1)
    # Under the STATIC default (1) this would already be at the cap and blocked; under the
    # tiered weak budget (2) it must still be allowed — proving the guard reads the tiered value.
    assert engine._reexpand_guards_ok(graph, leaf.node_id) is True

    graph2, leaf2 = _leaf_with_seeded_count(engine, count=2)
    # At the tiered weak cap (2) -> blocked.
    assert engine._reexpand_guards_ok(graph2, leaf2.node_id) is False


def test_reexpand_guards_ok_strong_model_hits_the_lower_cap_sooner(monkeypatch):
    _patch_price(monkeypatch, 10.0)  # strong -> budget 1
    engine, _ = _make_engine("some/premium-model", {
        "got_reexpand_max_iterations_tiered_enabled": True,
        "got_reexpand_max_iterations_weak": 2,
        "got_reexpand_max_iterations_standard": 1,
        "got_reexpand_max_iterations_strong": 1,
    })
    graph, leaf = _leaf_with_seeded_count(engine, count=1)
    # Strong tier's cap is 1 -> already blocked at count=1, unlike the weak-tier case above.
    assert engine._reexpand_guards_ok(graph, leaf.node_id) is False


# ---------------------------------------------------------------------------
# Binds at _reexpand_check / _apply_reexpand (the follow-up-detector path, end to end)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_followup_path_honors_the_tiered_budget_across_repeated_reexpansion(monkeypatch):
    _patch_price(monkeypatch, 0.0)  # weak -> budget 2 (static default is 1)
    engine, expansion = _make_engine("some/unpriced-model", {
        "got_reexpand_max_iterations_tiered_enabled": True,
        "got_reexpand_max_iterations_weak": 2,
        "got_reexpand_max_iterations_standard": 1,
        "got_reexpand_max_iterations_strong": 1,
    })
    checker = _attach_got(engine, {"needs_followup": True, "reason": "still unresolved"})
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    # First pass: re-expands (count 0 -> 1), well under either budget.
    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)
    assert len(leaf.children) == 1
    assert leaf.children[0] and leaf.details.get("_got_reexpand_count") == 1

    # Second pass on the freshly spawned child (seeded with count=1 by the propagation fix this
    # helper method also participates in): under the STATIC default of 1 this would already be
    # blocked; under the tiered weak budget of 2 it must still be allowed to re-expand once more.
    # The child is a fresh FakeExpansion leaf with no ACTION_RESULT yet (it was never executed) —
    # give it one directly so `_reexpand_check` reaches its budget comparison rather than bailing
    # out early on the "not a completed leaf" guard.
    child = graph.get_node(leaf.children[0])
    assert child.details.get("_got_reexpand_count") == 1
    child.details[DetailKey.ACTION_RESULT.value] = {"success": True}
    verdict = await engine._reexpand_check(graph, child.node_id, 1)
    assert verdict is not None, "weak-tier budget (2) must still permit a second re-expansion"


@pytest.mark.asyncio
async def test_master_switch_off_is_a_noop_even_with_tiering_on(monkeypatch):
    # Mirrors native_vote_k_tiering_test.py's symmetric test: got_reexpand_enabled=False (the
    # master switch) must fully suppress re-expansion regardless of the tiered flag/weak budget —
    # _effective_reexpand_max_iterations() must never even be consulted.
    _patch_price(monkeypatch, 0.0)  # weak -> would pick budget 2 if the master switch didn't gate it
    engine, expansion = _make_engine("some/unpriced-model", {
        "got_reexpand_enabled": False,
        "got_reexpand_max_iterations_tiered_enabled": True,
        "got_reexpand_max_iterations_weak": 2,
        "got_reexpand_max_iterations_standard": 1,
        "got_reexpand_max_iterations_strong": 1,
    })
    checker = _attach_got(engine, {"needs_followup": True, "reason": "should never be read"})
    graph = IdeaDag(root_title="root")
    parent, leaf = _completed_leaf(graph)

    result = await engine._handle_leaf_node(graph, leaf.node_id, 0, None)

    assert leaf.children == [], "no children should be created when the master switch is off"
    assert checker.calls == 0, "follow-up check must not be invoked when the master switch is off"
    assert expansion.calls == 0
