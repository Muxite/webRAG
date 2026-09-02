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
