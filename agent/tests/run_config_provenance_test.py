"""A stored cell must record the configuration that produced it.

Found by post-run analysis: an analyst could not determine which mechanisms were active in the
`phi3_both` run, because env-gated module flags appear NOWHERE in the stored artifact. The cfg
hash in the filename covers only `variant_specific_settings`, and every module this project added
this phase is switched by an env var — so two cells produced by materially different systems are
indistinguishable after the fact, and a comparison between them cannot be validated by anyone.

This is the same argument the cell already makes for `model_metadata`: the model tag alone cannot
tell two arms apart when one was served a different quantization. Telemetry only; nothing reads it
back.
"""
from __future__ import annotations

from agent.app.testing.runner import capture_run_config


def test_module_flags_that_change_behaviour_are_recorded(monkeypatch):
    monkeypatch.setenv("LEDGER_HOST_MODULES", "derive")
    monkeypatch.setenv("LEDGER_ZERO_VISIT_GATE", "1")
    monkeypatch.setenv("LLM_SEED", "12345")

    captured = capture_run_config()

    assert captured["LEDGER_HOST_MODULES"] == "derive"
    assert captured["LEDGER_ZERO_VISIT_GATE"] == "1"
    assert captured["LLM_SEED"] == "12345"


def test_an_unset_flag_is_absent_rather_than_recorded_as_a_default(monkeypatch):
    """Absent is never zero. Recording an unset flag as "0" would assert that the run took the
    off-path, when in truth the code's own default decided and that default may since have
    changed."""
    monkeypatch.delenv("LEDGER_CONTEXT_FIT", raising=False)

    assert "LEDGER_CONTEXT_FIT" not in capture_run_config()


def test_a_secret_is_never_captured_even_under_a_watched_prefix(monkeypatch):
    """A provenance record ships inside a result file that gets shared, diffed and pasted into
    handoffs. Anything key-shaped must never enter it, however it is named."""
    monkeypatch.setenv("LEDGER_API_KEY", "sk-do-not-store-me")
    monkeypatch.setenv("IDEA_TEST_AUTH_TOKEN", "t-do-not-store-me")
    monkeypatch.setenv("LEDGER_SEARCH_SECRET", "s-do-not-store-me")

    captured = capture_run_config()

    assert not any("do-not-store-me" in str(v) for v in captured.values())
    for name in ("LEDGER_API_KEY", "IDEA_TEST_AUTH_TOKEN", "LEDGER_SEARCH_SECRET"):
        assert name not in captured


# 2026-09-03: capture_run_config only watched LEDGER_/IDEA_TEST_ (plus a handful of individually
# named extras), so AGENT_*, BROWSER_* and IDEA_CHECKPOINT_* flags -- real, behaviour-changing env
# vars read by agent/app/agent.py, agent/app/interface_agent.py and agent/app/idea_checkpointer.py
# -- were invisible to provenance forever. This only improves cells produced from here on; nothing
# is retrofitted onto the ~10,400 already stored (only 144 of which carry a run_config block at all).


def test_agent_prefixed_flags_are_recorded(monkeypatch):
    monkeypatch.setenv("AGENT_USE_IDEA_DAG", "1")
    monkeypatch.setenv("AGENT_BLOCKED_LIMIT", "5")

    captured = capture_run_config()

    assert captured["AGENT_USE_IDEA_DAG"] == "1"
    assert captured["AGENT_BLOCKED_LIMIT"] == "5"


def test_browser_prefixed_flags_are_recorded(monkeypatch):
    monkeypatch.setenv("BROWSER_HEADLESS", "0")

    captured = capture_run_config()

    assert captured["BROWSER_HEADLESS"] == "0"


def test_idea_checkpoint_prefixed_flags_are_recorded(monkeypatch):
    monkeypatch.setenv("IDEA_CHECKPOINT_ENABLED", "1")
    monkeypatch.setenv("IDEA_CHECKPOINT_BACKEND", "redis")

    captured = capture_run_config()

    assert captured["IDEA_CHECKPOINT_ENABLED"] == "1"
    assert captured["IDEA_CHECKPOINT_BACKEND"] == "redis"


def test_an_unset_new_prefix_flag_is_still_absent_rather_than_defaulted(monkeypatch):
    monkeypatch.delenv("AGENT_BLOCKED_LIMIT", raising=False)
    monkeypatch.delenv("BROWSER_HEADLESS", raising=False)
    monkeypatch.delenv("IDEA_CHECKPOINT_ENABLED", raising=False)

    captured = capture_run_config()

    assert "AGENT_BLOCKED_LIMIT" not in captured
    assert "BROWSER_HEADLESS" not in captured
    assert "IDEA_CHECKPOINT_ENABLED" not in captured


def test_a_secret_under_a_new_prefix_is_never_captured(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "sk-do-not-store-me")
    monkeypatch.setenv("BROWSER_AUTH_TOKEN", "t-do-not-store-me")
    monkeypatch.setenv("IDEA_CHECKPOINT_SECRET", "s-do-not-store-me")

    captured = capture_run_config()

    assert not any("do-not-store-me" in str(v) for v in captured.values())
    for name in ("AGENT_API_KEY", "BROWSER_AUTH_TOKEN", "IDEA_CHECKPOINT_SECRET"):
        assert name not in captured
