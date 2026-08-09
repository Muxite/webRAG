"""Tests for ``model_tiers.local_model_params_b`` / ``local_model_size_band`` / ``size_band_value``
— the size refinement WITHIN ``capability_tier()``'s flat ``weak`` bucket for local models.

The tag list is the real ``badmodel-lab/roster.yaml`` roster (every subject + anchor), because that
is the population the bands were calibrated against; the edge cases pin the two "return None rather
than guess" contracts (named sizes, priced slugs).

``local_model_params_b`` is a pure parser and needs no pricing patch; ``local_model_size_band``
gates on ``price_tier`` and reuses ``capability_tier_test``'s ``_patch_price`` pattern.
"""
from __future__ import annotations

import pytest

from agent.app import model_tiers


def _patch_price(monkeypatch, price):
    monkeypatch.setattr(model_tiers.model_costs, "_lookup_pricing",
                        lambda name: {"output_per_million": price})


# Every tag in badmodel-lab/roster.yaml (subjects then anchors), with the parameter count its tag
# encodes -- None where the tag encodes none, even when the model's real size is known (tinyllama is
# 1.1B and phi3:mini is 3.8B, but neither TAG says so, and this parser never guesses).
ROSTER_TAGS = [
    ("tinyllama", None),
    ("qwen2.5:0.5b", 0.5),
    ("llama3.2:1b", 1.0),
    ("qwen2.5:1.5b", 1.5),
    ("gemma2:2b", 2.0),
    ("llama3.2:3b", 3.0),
    ("phi3:mini", None),
    ("qwen2.5:14b", 14.0),
    ("qwen2.5:7b", 7.0),
    ("openai/gpt-4.1-nano", None),
]


@pytest.mark.parametrize("tag,expected", ROSTER_TAGS)
def test_params_parsed_for_every_roster_tag(tag, expected):
    assert model_tiers.local_model_params_b(tag) == expected


@pytest.mark.parametrize("tag,expected", [
    ("QWEN2.5:7B", 7.0),                  # case-insensitive on both family and suffix
    ("qwen2.5:14b-instruct-q4_K_M", 14.0),  # quantization/variant suffixes
    ("llama3.1:8b", 8.0),
    ("llama3.1:70b", 70.0),
    ("tinyllama:1.1b", 1.1),              # the same family DOES parse once the tag pins a size
    ("phi3:3.8b", 3.8),                   # ditto -- phi3:mini is unparseable, phi3:3.8b is not
    ("deepseek-r1:1.5b", 1.5),
])
def test_params_parsed_for_common_tag_variants(tag, expected):
    assert model_tiers.local_model_params_b(tag) == expected


@pytest.mark.parametrize("tag", [
    None, "", ":", "qwen2.5:", "mistral:latest", "llama3:instruct",
    "phi3:medium",                 # named sizes are deliberately not guessed at
    "mixtral:8x7b",                # MoE total != a comparable dense count -> unparsed on purpose
    "qwen2.5:b", "qwen2.5:0b", "qwen2.5:0.0b",   # malformed / nonsense sizes
    "qwen2.5:7bx", "some:12billion",
    "openai/gpt-4.1-nano", "google/gemini-2.5-flash-lite",  # priced API names, no tag at all
    "qwen/qwen3-8b:free",          # size is in the FAMILY segment, not the tag -> not parsed
])
def test_params_none_for_unparseable_tags(tag):
    assert model_tiers.local_model_params_b(tag) is None


@pytest.mark.parametrize("tag,band", [
    ("qwen2.5:0.5b", "tiny"),
    ("llama3.2:1b", "tiny"),
    ("qwen2.5:1.5b", "tiny"),
    ("gemma2:2b", "small"),       # 2B is the tiny/small boundary (exclusive upper bound)
    ("llama3.2:3b", "small"),
    ("phi3:3.8b", "small"),
    ("qwen2.5:7b", "medium"),     # ties gpt-4.1-nano on 5 of 7 reachable tasks
    ("llama3.1:8b", "medium"),
    ("qwen2.5:14b", "large"),     # at the paid-API ceiling (reachable 0.97 / hard 0.95)
    ("llama3.1:70b", "large"),
])
def test_size_bands_for_unpriced_models(monkeypatch, tag, band):
    _patch_price(monkeypatch, 0.0)  # unpriced -> "unknown" price_tier -> "weak" capability_tier
    assert model_tiers.capability_tier(tag) == "weak"
    assert model_tiers.local_model_size_band(tag) == band


@pytest.mark.parametrize("tag", ["tinyllama", "phi3:mini", "mixtral:8x7b", "mistral:latest"])
def test_unparseable_local_tags_have_no_band(monkeypatch, tag):
    _patch_price(monkeypatch, 0.0)
    assert model_tiers.capability_tier(tag) == "weak"
    assert model_tiers.local_model_size_band(tag) is None


@pytest.mark.parametrize("price", [0.5, 1.0, 2.0, 10.0])
def test_priced_models_never_get_a_band(monkeypatch, price):
    """Cheap-but-priced is ``weak`` too, yet must NOT be refined: price is the stronger signal
    there, and an OpenRouter slug's embedded size (``meta-llama/llama-3.1-8b-instruct``) is not
    this axis's business."""
    _patch_price(monkeypatch, price)
    for tag in ("openai/gpt-4.1-nano", "qwen2.5:7b", "meta-llama/llama-3.1-8b-instruct:free"):
        assert model_tiers.local_model_size_band(tag) is None


def test_band_none_on_pricing_lookup_error(monkeypatch):
    def _boom(name):
        raise RuntimeError("no pricing")
    monkeypatch.setattr(model_tiers.model_costs, "_lookup_pricing", _boom)
    # price_tier swallows the error into "unknown", so the band still resolves -- the same hedge
    # capability_tier makes (a lookup failure means "assume local/weak", not "assume capable").
    assert model_tiers.local_model_size_band("qwen2.5:7b") == "medium"


def test_size_band_value_dispatch():
    kwargs = dict(tiny=4, small=3, medium=2, large=1, unknown=99)
    assert model_tiers.size_band_value("tiny", **kwargs) == 4
    assert model_tiers.size_band_value("small", **kwargs) == 3
    assert model_tiers.size_band_value("medium", **kwargs) == 2
    assert model_tiers.size_band_value("large", **kwargs) == 1


def test_size_band_value_unknown_falls_back_to_the_explicit_unknown_arg():
    # NOT a middle band: "unknown" means "keep what you would have done without size refinement".
    kwargs = dict(tiny=4, small=3, medium=2, large=1, unknown=99)
    assert model_tiers.size_band_value(None, **kwargs) == 99
    assert model_tiers.size_band_value("bogus", **kwargs) == 99


def test_capability_tier_contract_is_unchanged(monkeypatch):
    """The refinement is additive: capability_tier still returns exactly three strings, and a big
    local model is still ``weak`` (it needs the mitigation stack, just not the blanket dose)."""
    _patch_price(monkeypatch, 0.0)
    for tag, _ in ROSTER_TAGS:
        assert model_tiers.capability_tier(tag) == "weak"
