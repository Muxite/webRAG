"""Tests for ``idea_test_runner._settings_fingerprint``, the result-filename disambiguator.

Regression coverage for a real defect found during a live A/B run (2026-08-20): the result
filename encoded run_id/test_id/model/variant/tier/repeat but NOT which arm or IDEA_TEST_*
override combination produced it, so running two arms for the same model under one run_id
silently overwrote the first arm's JSON with the second's.
"""
from __future__ import annotations

from agent.app.idea_test_runner import _settings_fingerprint


def test_fingerprint_is_deterministic():
    settings = {"got_reexpand_enabled": True, "max_steps": 50}
    assert _settings_fingerprint(settings) == _settings_fingerprint(settings)


def test_fingerprint_is_independent_of_key_order():
    a = {"got_reexpand_enabled": True, "max_steps": 50}
    b = {"max_steps": 50, "got_reexpand_enabled": True}
    assert _settings_fingerprint(a) == _settings_fingerprint(b)


def test_fingerprint_differs_for_different_arms():
    baseline = {"merge_goal_evaluation_first_enabled": False, "verify_reason_first_enabled": False}
    reason_first = {"merge_goal_evaluation_first_enabled": True, "verify_reason_first_enabled": True}
    assert _settings_fingerprint(baseline) != _settings_fingerprint(reason_first)


def test_fingerprint_differs_for_a_single_flag_flip():
    a = {"verify_reason_first_enabled": False}
    b = {"verify_reason_first_enabled": True}
    assert _settings_fingerprint(a) != _settings_fingerprint(b)


def test_fingerprint_is_short_hex():
    fp = _settings_fingerprint({"a": 1})
    assert len(fp) == 8
    assert all(c in "0123456789abcdef" for c in fp)
