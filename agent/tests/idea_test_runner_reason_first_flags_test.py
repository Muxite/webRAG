"""Tests for the ``IDEA_TEST_*`` overrides and the ``reason_first`` arm profile
that make the three reason-first field-ordering flags (merge_goal_evaluation_first_enabled,
verify_reason_first_enabled, got_reexpand_followup_reason_first_enabled) reachable from a
benchmark run (``idea_test_runner._apply_got_experiment_overrides``).

Prompt/schema substitution correctness for these flags is covered separately in
``reason_first_ordering_test.py``; this file only pins the env-var/arm-profile wiring.
"""
from __future__ import annotations

from agent.app.idea_test_runner import (
    _GOT_ARM_PROFILES,
    _apply_got_experiment_overrides,
)


def _base():
    return {
        "merge_goal_evaluation_first_enabled": False,
        "verify_reason_first_enabled": False,
        "got_reexpand_followup_reason_first_enabled": False,
    }


def test_absent_flags_preserve_defaults():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={})
    assert settings == _base()


def test_merge_goal_eval_first_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_MERGE_GOAL_EVAL_FIRST": "1"}
    )
    assert settings["merge_goal_evaluation_first_enabled"] is True
    assert settings["verify_reason_first_enabled"] is False
    assert settings["got_reexpand_followup_reason_first_enabled"] is False


def test_verify_reason_first_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_VERIFY_REASON_FIRST": "true"}
    )
    assert settings["verify_reason_first_enabled"] is True
    assert settings["merge_goal_evaluation_first_enabled"] is False
    assert settings["got_reexpand_followup_reason_first_enabled"] is False


def test_followup_reason_first_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_GOT_FOLLOWUP_REASON_FIRST": "on"}
    )
    assert settings["got_reexpand_followup_reason_first_enabled"] is True
    assert settings["merge_goal_evaluation_first_enabled"] is False
    assert settings["verify_reason_first_enabled"] is False


def test_explicit_falsey_forces_off():
    settings = _base()
    settings["merge_goal_evaluation_first_enabled"] = True
    settings["verify_reason_first_enabled"] = True
    settings["got_reexpand_followup_reason_first_enabled"] = True
    _apply_got_experiment_overrides(
        settings,
        environ={
            "IDEA_TEST_MERGE_GOAL_EVAL_FIRST": "0",
            "IDEA_TEST_VERIFY_REASON_FIRST": "no",
            "IDEA_TEST_GOT_FOLLOWUP_REASON_FIRST": "off",
        },
    )
    assert settings["merge_goal_evaluation_first_enabled"] is False
    assert settings["verify_reason_first_enabled"] is False
    assert settings["got_reexpand_followup_reason_first_enabled"] is False


def test_blank_env_values_are_noop():
    settings = _base()
    _apply_got_experiment_overrides(
        settings,
        environ={
            "IDEA_TEST_MERGE_GOAL_EVAL_FIRST": "   ",
            "IDEA_TEST_VERIFY_REASON_FIRST": "",
            "IDEA_TEST_GOT_FOLLOWUP_REASON_FIRST": "   ",
        },
    )
    assert settings == _base()


def test_arm_reason_first_expands_all_three_flags():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "reason_first"})
    assert settings["merge_goal_evaluation_first_enabled"] is True
    assert settings["verify_reason_first_enabled"] is True
    assert settings["got_reexpand_followup_reason_first_enabled"] is True


def test_arm_reason_first_is_registered():
    assert "reason_first" in _GOT_ARM_PROFILES
    profile = _GOT_ARM_PROFILES["reason_first"]
    assert profile == {
        "merge_goal_evaluation_first_enabled": True,
        "verify_reason_first_enabled": True,
        "got_reexpand_followup_reason_first_enabled": True,
    }


def test_arm_profile_then_individual_override_composes_with_override_winning():
    settings = _base()
    _apply_got_experiment_overrides(
        settings,
        environ={
            "IDEA_TEST_ARM": "reason_first",
            "IDEA_TEST_VERIFY_REASON_FIRST": "0",
        },
    )
    # Profile flags not explicitly overridden still apply.
    assert settings["merge_goal_evaluation_first_enabled"] is True
    assert settings["got_reexpand_followup_reason_first_enabled"] is True
    # Explicit override wins over the profile's True.
    assert settings["verify_reason_first_enabled"] is False
