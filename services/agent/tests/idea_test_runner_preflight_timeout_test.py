"""Regression test for the configurable pre-flight-probe timeout (2026-08-07 barrage smoke finding):
a local (Ollama) model evicted from Ollama's single-loaded-model slot by a sibling local model needs
longer than the 20s-per-payload-candidate default to respond to preflight_check_llm's cold-load
probe, silently killing the whole cell with an empty-string asyncio.TimeoutError before any real
work starts. IDEA_TEST_PREFLIGHT_TIMEOUT_SECONDS lets a caller raise it without touching the
unmodified cloud-API default. Narrow unit test of the pure helper only — the live wait_for behavior
is exercised by a live run, not here.
"""
import agent.app.idea_test_runner as runner


def test_preflight_timeout_defaults_to_20_seconds(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_PREFLIGHT_TIMEOUT_SECONDS", raising=False)
    assert runner._preflight_call_timeout_seconds() == 20.0


def test_preflight_timeout_reads_env_override(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_PREFLIGHT_TIMEOUT_SECONDS", "120")
    assert runner._preflight_call_timeout_seconds() == 120.0


def test_preflight_timeout_falls_back_to_default_on_garbage_value(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_PREFLIGHT_TIMEOUT_SECONDS", "not-a-number")
    assert runner._preflight_call_timeout_seconds() == 20.0
