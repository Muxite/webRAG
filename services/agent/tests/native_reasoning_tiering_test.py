"""Tests for native executor param tiering:
  * A3b — reasoning-effort/token discipline (``native_reasoning_effort_discipline_enabled``):
    a reasoning-model perception micro-prompt uses reasoning_effort=minimal + a token floor.
  * A5  — price-tier token budget scaling (``price_tier_param_tiering_enabled``).

Both are opt-in; the default (both flags off) path is byte-identical — asserted here at the
``LeafAction`` helper level (the single place every native micro-prompt resolves its params).
"""
from __future__ import annotations

import pytest

from agent.app import model_tiers
from agent.app.idea_policies.actions import SearchLeafAction


# ---------------------------------------------------------------------------
# Shared price_tier / multiplier helper
# ---------------------------------------------------------------------------
def _patch_price(monkeypatch, price):
    monkeypatch.setattr(model_tiers.model_costs, "_lookup_pricing",
                        lambda name: {"output_per_million": price})


def test_price_tier_buckets(monkeypatch):
    for price, tier in [(0.0, "unknown"), (0.5, "cheap"), (1.0, "cheap"),
                        (2.0, "mid"), (5.0, "mid"), (10.0, "premium")]:
        _patch_price(monkeypatch, price)
        assert model_tiers.price_tier("m") == tier, (price, tier)


def test_price_tier_unknown_on_lookup_error(monkeypatch):
    def _boom(name):
        raise RuntimeError("no pricing")
    monkeypatch.setattr(model_tiers.model_costs, "_lookup_pricing", _boom)
    assert model_tiers.price_tier("m") == "unknown"


def test_tier_token_multiplier(monkeypatch):
    for price, mult in [(0.5, 1.0), (2.0, 2.0), (10.0, 4.0), (0.0, 2.0)]:
        _patch_price(monkeypatch, price)
        assert model_tiers.tier_token_multiplier("m") == mult, (price, mult)


# ---------------------------------------------------------------------------
# LeafAction helpers — default OFF is byte-identical
# ---------------------------------------------------------------------------
def _action(**settings):
    return SearchLeafAction(settings=settings)


def test_helpers_default_off_are_byte_identical():
    a = _action()  # both flags default false
    # Token budget unchanged for reasoning AND non-reasoning models.
    assert a._executor_max_tokens("gpt-5-mini", 500) == 500
    assert a._executor_max_tokens("gemini-3.1-pro-preview", 500) == 500
    assert a._executor_max_tokens("gpt-5-mini", None) is None
    # No reasoning-effort override.
    assert a._micro_prompt_reasoning_effort("gpt-5-mini") is None
    assert a._micro_prompt_reasoning_effort("gpt-5-mini", default="high") == "high"


# ---------------------------------------------------------------------------
# A3b — reasoning discipline when ON
# ---------------------------------------------------------------------------
def test_a3b_minimal_effort_for_reasoning_model_only():
    a = _action(native_reasoning_effort_discipline_enabled=True)
    # gpt-5-mini is a reasoning model (accepts_reasoning_effort) -> minimal.
    assert a._micro_prompt_reasoning_effort("gpt-5-mini") == "minimal"
    assert a._micro_prompt_reasoning_effort("openai/gpt-4.1-nano") == "minimal"
    # A non-reasoning model is untouched (falls back to the caller default).
    assert a._micro_prompt_reasoning_effort("gemini-3.1-pro-preview", default="high") == "high"
    assert a._micro_prompt_reasoning_effort("anthropic/claude-opus-4.7") is None


def test_a3b_floors_reasoning_model_tokens():
    a = _action(native_reasoning_effort_discipline_enabled=True, native_reasoning_min_tokens_floor=2048)
    # A small budget is floored up for a reasoning model...
    assert a._executor_max_tokens("gpt-5-mini", 500) == 2048
    # ...but a budget already above the floor is unchanged.
    assert a._executor_max_tokens("gpt-5-mini", 8192) == 8192
    # A non-reasoning model is never floored.
    assert a._executor_max_tokens("gemini-3.1-pro-preview", 500) == 500


# ---------------------------------------------------------------------------
# A5 — price-tier token scaling when ON
# ---------------------------------------------------------------------------
def test_a5_scales_tokens_by_price_tier(monkeypatch):
    a = _action(price_tier_param_tiering_enabled=True)
    _patch_price(monkeypatch, 0.5)   # cheap -> 1.0x
    assert a._executor_max_tokens("m", 500) == 500
    _patch_price(monkeypatch, 2.0)   # mid -> 2.0x
    assert a._executor_max_tokens("m", 500) == 1000
    _patch_price(monkeypatch, 10.0)  # premium -> 4.0x
    assert a._executor_max_tokens("m", 500) == 2000


def test_a5_price_tier_skipped_when_flag_argument_false(monkeypatch):
    # ``price_tier=False`` (merge/verify sites) skips the A5 multiplier even when the flag is on.
    a = _action(price_tier_param_tiering_enabled=True)
    _patch_price(monkeypatch, 10.0)  # premium
    assert a._executor_max_tokens("m", 500, price_tier=False) == 500


def test_a3b_and_a5_compose(monkeypatch):
    # Both flags on: A5 scales, then A3b floors — the floor wins when the scaled value is lower.
    a = _action(
        native_reasoning_effort_discipline_enabled=True,
        native_reasoning_min_tokens_floor=2048,
        price_tier_param_tiering_enabled=True,
    )
    # gpt-5-mini is a reasoning model; give it mid pricing (2x). 500 * 2 = 1000 -> floored to 2048.
    _patch_price(monkeypatch, 2.0)
    assert a._executor_max_tokens("gpt-5-mini", 500) == 2048
    # A larger base where the scaled value already clears the floor: 2000 * 2 = 4000 (> 2048).
    assert a._executor_max_tokens("gpt-5-mini", 2000) == 4000
