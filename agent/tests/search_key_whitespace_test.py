"""An unstripped search key fails silently, and looks like a model failure.

``keys.env`` is CRLF-terminated. ``SERPER_KEY`` was read with a bare ``os.environ.get`` and
handed to the provider with its trailing ``\\r`` attached, so Serper answered 403 Unauthorized.
``ConnectorSearch`` turns a failed lookup into "no results" rather than an error, so the run
COMPLETES: the agent searches, gets nothing back, answers badly, and the artifact records a
low score with ``infra.failed = False``. Nothing anywhere says the credential was rejected.

Observed live during this cycle: a 42-character read of a 40-character key, producing exactly
that shape -- ``search.count = 0`` on every cell, no infra failure recorded, and a score that
looked like the model could not find anything.

The LLM key path already stripped (``_resolve_llm_api_key``), which is why the model worked
and only search was dead -- the most confusing possible presentation.

No network: only the config layer is exercised.
"""
from __future__ import annotations

import pytest

from shared.connector_config import ConnectorConfig, _clean_secret


KEY = "a" * 40


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("SERPER_KEY", "SEARCH_API_KEY", "SEARCH_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("raw", [
    KEY,
    KEY + "\r",          # the live case: CRLF line ending in keys.env
    KEY + "\n",
    KEY + "\r\n",
    "  " + KEY + "  ",
    f'"{KEY}"',
    f"'{KEY}'",
    f' "{KEY}"\r\n',
])
def test_the_key_survives_its_file_formatting(monkeypatch, raw):
    monkeypatch.setenv("SERPER_KEY", raw)
    assert ConnectorConfig().search_api_key == KEY


def test_an_empty_key_is_none_not_an_empty_string(monkeypatch):
    """``if not self.search_api_key`` must still warn, rather than pass a blank through."""
    monkeypatch.setenv("SERPER_KEY", "   \r\n")
    assert ConnectorConfig().search_api_key is None


def test_a_missing_key_is_still_none(monkeypatch):
    assert ConnectorConfig().search_api_key is None


def test_the_fallback_env_var_is_cleaned_too(monkeypatch):
    monkeypatch.setenv("SEARCH_API_KEY", KEY + "\r")
    assert ConnectorConfig().search_api_key == KEY


def test_serper_key_still_wins_over_the_fallback(monkeypatch):
    monkeypatch.setenv("SERPER_KEY", KEY + "\r")
    monkeypatch.setenv("SEARCH_API_KEY", "b" * 31)
    assert ConnectorConfig().search_api_key == KEY


def test_a_non_serper_provider_reads_only_the_generic_var(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("SERPER_KEY", KEY)
    monkeypatch.setenv("SEARCH_API_KEY", "brave-key\r")
    assert ConnectorConfig().search_api_key == "brave-key"


@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("", None),
    ("   ", None),
    ('""', None),
    ("x", "x"),
])
def test_clean_secret_edge_cases(value, expected):
    assert _clean_secret(value) == expected
