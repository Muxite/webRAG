"""Tests for capability-tiered ``native_vote_k`` (Phase 4 of the capability-continuum plan):
``native_vote_k_tiered_enabled`` overrides the static ``native_vote_k`` with a band picked via
``model_tiers.capability_tier(model_name)`` — weak models get more redundant finalize votes,
strong models taper toward k=1 (fully off downstream).

Reuses ``finalize_reconcile_test.py``'s fixtures (``_FakeIO``, ``_graph``, ``_resp``,
``_ANSWER_MANDATE``, ``_settings``) rather than duplicating them — see that module for the
non-tiered k-vote wiring this builds on (``test_variation_supersedes_kvote_when_both_enabled`` etc).
"""
from __future__ import annotations

import asyncio

from agent.app import model_tiers
from agent.app.idea_finalize import build_final_payload
from agent.tests.finalize_reconcile_test import _FakeIO, _graph, _resp, _ANSWER_MANDATE, _settings, _kinds


def _run(responder, model_name, **overrides):
    io = _FakeIO(responder)
    payload = asyncio.run(
        build_final_payload(io, _settings(**overrides), _graph(), _ANSWER_MANDATE, model_name)
    )
    return io, payload


def _patch_price(monkeypatch, price):
    monkeypatch.setattr(model_tiers.model_costs, "_lookup_pricing",
                        lambda name: {"output_per_million": price})


def _finalize_only_responder(value):
    def responder(kind, n):
        assert kind == "finalize"
        return _resp(f"Maximum depth is {value} m")
    return responder


def test_tiered_flag_off_leaves_static_native_vote_k_untouched(monkeypatch):
    _patch_price(monkeypatch, 0.0)  # weak by price -> would matter if tiering were on
    io, _ = _run(
        _finalize_only_responder("151"), "any-model",
        native_vote_k_enabled=True, native_vote_k=1,  # k=1 -> single finalize call either way
        native_vote_k_tiered_enabled=False,
    )
    assert _kinds(io).count("finalize") == 1


def test_weak_model_gets_the_weak_band_vote_count(monkeypatch):
    _patch_price(monkeypatch, 0.0)  # unpriced -> "unknown" price_tier -> "weak" capability_tier
    io, _ = _run(
        _finalize_only_responder("151"), "some/unpriced-local-model",
        native_vote_k_enabled=True,
        native_vote_k_tiered_enabled=True,
        native_vote_k_weak=3, native_vote_k_standard=2, native_vote_k_strong=1,
    )
    assert _kinds(io).count("finalize") == 3


def test_strong_model_tapers_to_k1_which_is_fully_off_downstream(monkeypatch):
    _patch_price(monkeypatch, 10.0)  # premium -> "strong"
    io, _ = _run(
        _finalize_only_responder("151"), "some/premium-model",
        native_vote_k_enabled=True,
        native_vote_k_tiered_enabled=True,
        native_vote_k_weak=3, native_vote_k_standard=2, native_vote_k_strong=1,
    )
    # k=1 fails the >= 2 gate in idea_finalize -> exactly the single passthrough finalize call.
    assert _kinds(io).count("finalize") == 1


def test_standard_model_gets_the_middle_band(monkeypatch):
    _patch_price(monkeypatch, 2.0)  # mid -> "standard"
    io, _ = _run(
        _finalize_only_responder("151"), "some/mid-model",
        native_vote_k_enabled=True,
        native_vote_k_tiered_enabled=True,
        native_vote_k_weak=3, native_vote_k_standard=2, native_vote_k_strong=1,
    )
    assert _kinds(io).count("finalize") == 2


def test_tiering_is_a_noop_without_the_master_switch(monkeypatch):
    # native_vote_k_tiered_enabled=True but native_vote_k_enabled=False -> master switch still
    # gates everything, exactly like the untiered flag does.
    _patch_price(monkeypatch, 0.0)  # weak -> would pick k=3 if the master switch didn't gate it
    io, _ = _run(
        _finalize_only_responder("151"), "some/unpriced-local-model",
        native_vote_k_enabled=False,
        native_vote_k_tiered_enabled=True,
        native_vote_k_weak=3, native_vote_k_standard=2, native_vote_k_strong=1,
    )
    assert _kinds(io).count("finalize") == 1


def test_both_flags_default_off_is_byte_identical():
    io, _ = _run(_finalize_only_responder("151"), "any-model")
    assert _kinds(io) == ["finalize"]
