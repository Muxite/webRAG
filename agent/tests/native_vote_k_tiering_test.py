"""Tests for capability-tiered ``native_vote_k`` (Phase 4 of the capability-continuum plan):
``native_vote_k_tiered_enabled`` overrides the static ``native_vote_k`` with a band picked via
``model_tiers.capability_tier(model_name)`` — weak models get more redundant finalize votes,
strong models taper toward k=1 (fully off downstream) — plus its size-band refinement
(``native_vote_k_size_band_enabled``), which splits the flat weak band by local model size.

Reuses ``finalize_reconcile_test.py``'s fixtures (``_FakeIO``, ``_graph``, ``_resp``,
``_ANSWER_MANDATE``, ``_settings``) rather than duplicating them — see that module for the
non-tiered k-vote wiring this builds on (``test_variation_supersedes_kvote_when_both_enabled`` etc).
"""
from __future__ import annotations

import asyncio

import pytest

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


# --- size-band refinement within the weak band (local/unpriced models only) -------------------
#
# Layered on the tiering above: when ``native_vote_k_size_band_enabled`` is on, a local model whose
# tag encodes a parameter count uses its band's k instead of the flat weak k.

_BAND_BANDS = dict(
    native_vote_k_local_tiny=4,
    native_vote_k_local_small=3,
    native_vote_k_local_medium=2,
    native_vote_k_local_large=1,
)
_TIER_BANDS = dict(native_vote_k_weak=3, native_vote_k_standard=2, native_vote_k_strong=1)


def _run_banded(monkeypatch, model_name, price=0.0, **overrides):
    _patch_price(monkeypatch, price)
    io, _ = _run(
        _finalize_only_responder("151"), model_name,
        native_vote_k_enabled=True,
        native_vote_k_tiered_enabled=True,
        native_vote_k_size_band_enabled=True,
        **_TIER_BANDS, **_BAND_BANDS, **overrides,
    )
    return _kinds(io).count("finalize")


def test_tiny_local_model_gets_more_votes_than_the_flat_weak_band(monkeypatch):
    assert _run_banded(monkeypatch, "qwen2.5:0.5b") == 4


def test_small_local_model_keeps_the_flat_weak_band(monkeypatch):
    assert _run_banded(monkeypatch, "llama3.2:3b") == 3


def test_medium_local_model_tapers_to_the_standard_dose(monkeypatch):
    # qwen2.5:7b ties gpt-4.1-nano on 5 of 7 reachable tasks -> less blanket redundancy.
    assert _run_banded(monkeypatch, "qwen2.5:7b") == 2


def test_large_local_model_tapers_off_this_blanket_lever(monkeypatch):
    # qwen2.5:14b sits at the paid-API ceiling; k=1 fails the >= 2 gate -> single finalize call.
    assert _run_banded(monkeypatch, "qwen2.5:14b") == 1


@pytest.mark.parametrize("model_name", ["tinyllama", "phi3:mini", "some/unpriced-local-model"])
def test_unparseable_tag_is_identical_to_todays_flat_weak_behavior(monkeypatch, model_name):
    """THE COMPATIBILITY PIN: no parseable size -> band None -> exactly the flat weak k (3), the
    same value the size-band flag being off would produce."""
    assert _run_banded(monkeypatch, model_name) == 3
    _patch_price(monkeypatch, 0.0)
    io, _ = _run(
        _finalize_only_responder("151"), model_name,
        native_vote_k_enabled=True,
        native_vote_k_tiered_enabled=True,
        native_vote_k_size_band_enabled=False,
        **_TIER_BANDS, **_BAND_BANDS,
    )
    assert _kinds(io).count("finalize") == 3


def test_priced_model_with_a_size_shaped_tag_is_not_refined(monkeypatch):
    # Priced cheap -> "weak" tier, but price is the stronger signal: no band, so the flat weak k
    # stands even though the tag would parse as "medium".
    assert _run_banded(monkeypatch, "qwen2.5:7b", price=0.5) == 3


def test_size_band_flag_alone_is_a_noop_without_the_tiered_flag(monkeypatch):
    _patch_price(monkeypatch, 0.0)
    io, _ = _run(
        _finalize_only_responder("151"), "qwen2.5:0.5b",
        native_vote_k_enabled=True, native_vote_k=1,
        native_vote_k_tiered_enabled=False,
        native_vote_k_size_band_enabled=True,
        **_TIER_BANDS, **_BAND_BANDS,
    )
    assert _kinds(io).count("finalize") == 1


def test_size_band_default_off_leaves_existing_tiering_untouched(monkeypatch):
    # Same call as test_weak_model_gets_the_weak_band_vote_count, but with a size-encoding tag:
    # without the new flag the tiny model still gets the flat weak 3, not 4.
    _patch_price(monkeypatch, 0.0)
    io, _ = _run(
        _finalize_only_responder("151"), "qwen2.5:0.5b",
        native_vote_k_enabled=True,
        native_vote_k_tiered_enabled=True,
        **_TIER_BANDS,
    )
    assert _kinds(io).count("finalize") == 3
