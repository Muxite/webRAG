"""Tests for ``model_costs.is_local_row`` and ``idea_test_runner._result_origin``.

Local (self-hosted) model runs must be recognizable for cost-reporting purposes without
relying on missing-price inference alone, since that inference is ambiguous (an unpriced
model string could also just be a pricing-table gap for a real API model). The ``origin``
field, stamped at result-write time from the resolved connector provider, is the primary
signal; result files predating that field fall back to the old implicit "unpriced" signal.
"""
from __future__ import annotations

from types import SimpleNamespace

from agent.app.model_costs import is_local_row
from agent.app.idea_test_runner import _result_origin


def test_is_local_row_prefers_explicit_origin_local():
    assert is_local_row({"origin": "local", "model": "gpt-5-mini", "usd": None}) is True


def test_is_local_row_prefers_explicit_origin_api():
    # Even if the model happens to be unpriced, an explicit "api" origin wins.
    assert is_local_row({"origin": "api", "model": "some-unpriced-api-model"}) is False


def test_is_local_row_falls_back_to_unpriced_model_when_origin_absent():
    # Pre-existing result files (written before the `origin` field existed) have no
    # `origin` key at all — fall back to the old implicit "unpriced -> local" signal.
    assert is_local_row({"model": "qwen2.5:14b"}) is True


def test_is_local_row_falls_back_to_priced_model_when_origin_absent():
    assert is_local_row({"model": "gpt-5-mini"}) is False


def test_is_local_row_no_model_no_origin_is_not_local():
    assert is_local_row({}) is False


def test_is_local_row_accepts_flattened_row_shape():
    # scripts/bench_common.py::load_row's flattened output uses the same key names.
    flattened_row = {"path": "/x.json", "model": "qwen2.5:14b", "origin": None, "variant": "graph"}
    assert is_local_row(flattened_row) is True


def _connector_llm(provider):
    return SimpleNamespace(config=SimpleNamespace(llm_provider=provider))


def test_result_origin_ollama_is_local():
    assert _result_origin(_connector_llm("ollama")) == "local"


def test_result_origin_local_is_local():
    assert _result_origin(_connector_llm("local")) == "local"


def test_result_origin_openrouter_is_api():
    assert _result_origin(_connector_llm("openrouter")) == "api"


def test_result_origin_anthropic_is_api():
    assert _result_origin(_connector_llm("anthropic")) == "api"


def test_result_origin_case_insensitive():
    assert _result_origin(_connector_llm("Ollama")) == "local"


def test_result_origin_missing_config_defaults_api():
    assert _result_origin(SimpleNamespace(config=None)) == "api"
