"""Tests for ``model_tiers.capability_tier`` / ``tier_value`` — the 3-band mitigation-strength
classifier that tiered levers (native vote-k, native re-expansion budget) dispatch on.

Mirrors ``native_reasoning_tiering_test.py``'s ``_patch_price`` pattern.
"""
from __future__ import annotations

from agent.app import model_tiers


def _patch_price(monkeypatch, price):
    monkeypatch.setattr(model_tiers.model_costs, "_lookup_pricing",
                        lambda name: {"output_per_million": price})


def test_capability_tier_buckets(monkeypatch):
    for price, tier in [
        (0.0, "weak"),      # unpriced -> unknown price_tier -> weak (deliberate hedge)
        (0.5, "weak"),      # cheap
        (1.0, "weak"),      # cheap boundary
        (2.0, "standard"),  # mid
        (5.0, "standard"),  # mid boundary
        (10.0, "strong"),   # premium
    ]:
        _patch_price(monkeypatch, price)
        assert model_tiers.capability_tier("m") == tier, (price, tier)


def test_capability_tier_unknown_on_lookup_error(monkeypatch):
    def _boom(name):
        raise RuntimeError("no pricing")
    monkeypatch.setattr(model_tiers.model_costs, "_lookup_pricing", _boom)
    assert model_tiers.capability_tier("m") == "weak"


def test_tier_value_dispatch():
    assert model_tiers.tier_value("weak", weak=1, standard=2, strong=3) == 1
    assert model_tiers.tier_value("standard", weak=1, standard=2, strong=3) == 2
    assert model_tiers.tier_value("strong", weak=1, standard=2, strong=3) == 3


def test_tier_value_unrecognized_falls_back_to_standard():
    assert model_tiers.tier_value("bogus", weak=1, standard=2, strong=3) == 2
    assert model_tiers.tier_value(None, weak=1, standard=2, strong=3) == 2
