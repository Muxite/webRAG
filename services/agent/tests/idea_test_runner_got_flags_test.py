"""Tests for the per-run ``IDEA_TEST_GOT_*`` experiment toggles
(``idea_test_runner._apply_got_experiment_overrides``).

These flags let a research run flip a dormant adaptive mechanism on/off without
editing the checked-in ``idea_dag_settings.json`` (whose defaults stay OFF). The
contract: truthy enables, explicit falsey forces off, ABSENT leaves the loaded
default untouched. Pinned here for the confidence->action loop toggle
``IDEA_TEST_GOT_CONFIDENCE_REEXPAND`` (and its siblings for coverage).
"""
from __future__ import annotations

import logging

from agent.app.idea_test_runner import (
    _GOT_ARM_PROFILES,
    _apply_got_experiment_overrides,
)


def _base():
    # Mirrors the shipped defaults: every adaptive mechanism OFF.
    return {
        "got_reexpand_enabled": False,
        "got_reexpand_max_iterations": 1,
        "got_step_confidence_judge_enabled": False,
        "got_step_confidence_judge_sample_every": 1,
        "got_step_confidence_reexpand_enabled": False,
        "got_step_confidence_reexpand_threshold": 0.5,
        "got_reexpand_corrective_context_enabled": False,
        "got_backtrack_enabled": False,
        "connector_retry_on_failure_enabled": False,
        "tool_failure_recovery_enabled": False,
        "native_vote_k_enabled": False,
        "native_vote_k": 1,
        "expansion_expect_contract_enabled": False,
        "native_reasoning_effort_discipline_enabled": False,
        "price_tier_param_tiering_enabled": False,
    }


def test_absent_flags_preserve_defaults():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={})
    assert settings == _base(), "no env flags -> loaded defaults untouched"


def test_confidence_reexpand_truthy_enables():
    for truthy in ("1", "true", "yes", "on", "TRUE"):
        settings = _base()
        _apply_got_experiment_overrides(
            settings, environ={"IDEA_TEST_GOT_CONFIDENCE_REEXPAND": truthy}
        )
        assert settings["got_step_confidence_reexpand_enabled"] is True, truthy


def test_confidence_reexpand_explicit_falsey_forces_off():
    # Even if a settings file shipped it on, an explicit falsey env forces off.
    settings = _base()
    settings["got_step_confidence_reexpand_enabled"] = True
    for falsey in ("0", "false", "no", "off"):
        s = dict(settings)
        _apply_got_experiment_overrides(
            s, environ={"IDEA_TEST_GOT_CONFIDENCE_REEXPAND": falsey}
        )
        assert s["got_step_confidence_reexpand_enabled"] is False, falsey


def test_confidence_reexpand_blank_is_noop():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_GOT_CONFIDENCE_REEXPAND": "   "}
    )
    assert settings["got_step_confidence_reexpand_enabled"] is False


def test_sibling_flags_still_parse():
    # Regression guard on the extracted helper: the pre-existing toggles behave too.
    settings = _base()
    _apply_got_experiment_overrides(
        settings,
        environ={
            "IDEA_TEST_GOT_REEXPAND": "on",
            "IDEA_TEST_GOT_STEP_CONFIDENCE_JUDGE": "1",
            "IDEA_TEST_GOT_STEP_CONFIDENCE_SAMPLE_EVERY": "3",
        },
    )
    assert settings["got_reexpand_enabled"] is True
    assert settings["got_step_confidence_judge_enabled"] is True
    assert settings["got_step_confidence_judge_sample_every"] == 3


# --- R1 extension: remaining opt-in flag overrides ---------------------------------


def test_reexpand_max_iter_int_override():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_GOT_REEXPAND_MAX_ITER": "4"}
    )
    assert settings["got_reexpand_max_iterations"] == 4


def test_corrective_context_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_GOT_CORRECTIVE_CONTEXT": "true"}
    )
    assert settings["got_reexpand_corrective_context_enabled"] is True

    settings2 = _base()
    settings2["got_reexpand_corrective_context_enabled"] = True
    _apply_got_experiment_overrides(
        settings2, environ={"IDEA_TEST_GOT_CORRECTIVE_CONTEXT": "off"}
    )
    assert settings2["got_reexpand_corrective_context_enabled"] is False


def test_confidence_threshold_float_override():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_GOT_CONFIDENCE_THRESHOLD": "0.75"}
    )
    assert settings["got_step_confidence_reexpand_threshold"] == 0.75


def test_backtrack_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_GOT_BACKTRACK": "1"})
    assert settings["got_backtrack_enabled"] is True


def test_connector_retry_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_CONNECTOR_RETRY": "yes"})
    assert settings["connector_retry_on_failure_enabled"] is True


def test_tool_failure_recovery_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_TOOL_FAILURE_RECOVERY": "on"}
    )
    assert settings["tool_failure_recovery_enabled"] is True


def test_native_vote_k_int_override_enables_when_ge_2():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_NATIVE_VOTE_K": "3"})
    assert settings["native_vote_k"] == 3
    assert settings["native_vote_k_enabled"] is True


def test_native_vote_k_of_1_leaves_enabled_flag_alone():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_NATIVE_VOTE_K": "1"})
    assert settings["native_vote_k"] == 1
    assert settings["native_vote_k_enabled"] is False

    settings2 = _base()
    settings2["native_vote_k_enabled"] = True
    _apply_got_experiment_overrides(settings2, environ={"IDEA_TEST_NATIVE_VOTE_K": "1"})
    assert settings2["native_vote_k"] == 1
    assert settings2["native_vote_k_enabled"] is True


def test_expect_contract_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_EXPECT_CONTRACT": "1"})
    assert settings["expansion_expect_contract_enabled"] is True


def test_native_reasoning_discipline_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_NATIVE_REASONING_DISCIPLINE": "1"}
    )
    assert settings["native_reasoning_effort_discipline_enabled"] is True


def test_price_tier_tiering_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_PRICE_TIER_TIERING": "1"}
    )
    assert settings["price_tier_param_tiering_enabled"] is True


def test_all_new_flags_blank_are_noop():
    settings = _base()
    _apply_got_experiment_overrides(
        settings,
        environ={
            "IDEA_TEST_GOT_REEXPAND_MAX_ITER": "",
            "IDEA_TEST_GOT_CORRECTIVE_CONTEXT": "  ",
            "IDEA_TEST_GOT_CONFIDENCE_THRESHOLD": "",
            "IDEA_TEST_GOT_BACKTRACK": "",
            "IDEA_TEST_CONNECTOR_RETRY": "",
            "IDEA_TEST_TOOL_FAILURE_RECOVERY": "",
            "IDEA_TEST_NATIVE_VOTE_K": "",
            "IDEA_TEST_EXPECT_CONTRACT": "",
            "IDEA_TEST_NATIVE_REASONING_DISCIPLINE": "",
            "IDEA_TEST_PRICE_TIER_TIERING": "",
        },
    )
    assert settings == _base()


# --- R1 extension: IDEA_TEST_ARM named profiles -------------------------------------


def test_arm_absent_is_byte_identical_to_no_overrides():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={})
    assert settings == _base()


def test_arm_baseline_matches_shipped_defaults():
    settings = _base()
    # Flip a few flags on so we can prove "baseline" arm actually forces them off.
    settings["got_reexpand_enabled"] = True
    settings["native_vote_k_enabled"] = True
    settings["native_vote_k"] = 5
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "baseline"})
    assert settings == _base()


def test_arm_good_adaptive_expands_expected_flags():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "good_adaptive"})
    assert settings["got_reexpand_enabled"] is True
    assert settings["got_reexpand_max_iterations"] == 2
    assert settings["got_step_confidence_judge_enabled"] is True
    assert settings["got_step_confidence_reexpand_enabled"] is True
    assert settings["got_reexpand_corrective_context_enabled"] is True
    assert settings["tool_failure_recovery_enabled"] is True
    # Not part of this arm.
    assert settings["native_vote_k_enabled"] is False
    assert settings["got_backtrack_enabled"] is False


def test_arm_reexpand_only_expands_expected_flags():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "reexpand_only"})
    assert settings["got_reexpand_enabled"] is True
    assert settings["got_reexpand_max_iterations"] == 2
    assert settings["got_step_confidence_judge_enabled"] is False
    assert settings["got_step_confidence_reexpand_enabled"] is False


def test_arm_confidence_only_expands_expected_flags():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "confidence_only"})
    assert settings["got_step_confidence_judge_enabled"] is True
    assert settings["got_step_confidence_reexpand_enabled"] is True
    assert settings["got_reexpand_enabled"] is False


def test_arm_backtrack_only_expands_expected_flags():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "backtrack_only"})
    assert settings["got_backtrack_enabled"] is True
    assert settings["got_reexpand_enabled"] is False
    assert settings["native_vote_k_enabled"] is False


def test_arm_kvote_only_expands_expected_flags():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "kvote_only"})
    assert settings["native_vote_k_enabled"] is True
    assert settings["native_vote_k"] == 3
    assert settings["got_reexpand_enabled"] is False


def test_arm_full_expands_every_axis():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "full"})
    expected_true = {
        "got_reexpand_enabled",
        "got_step_confidence_judge_enabled",
        "got_step_confidence_reexpand_enabled",
        "got_reexpand_corrective_context_enabled",
        "tool_failure_recovery_enabled",
        "native_vote_k_enabled",
        "got_backtrack_enabled",
        "expansion_expect_contract_enabled",
        "native_reasoning_effort_discipline_enabled",
        "price_tier_param_tiering_enabled",
    }
    for key in expected_true:
        assert settings[key] is True, key
    assert settings["got_reexpand_max_iterations"] == 2
    assert settings["native_vote_k"] == 3
    # connector_retry is a separate ablation axis; not part of "full".
    assert settings["connector_retry_on_failure_enabled"] is False


def test_arm_profile_then_individual_override_composes_with_override_winning():
    settings = _base()
    _apply_got_experiment_overrides(
        settings,
        environ={
            "IDEA_TEST_ARM": "good_adaptive",
            # Override one flag from the profile explicitly.
            "IDEA_TEST_GOT_CONFIDENCE_REEXPAND": "0",
            # And enable an axis the profile doesn't touch.
            "IDEA_TEST_GOT_BACKTRACK": "1",
        },
    )
    # Profile flags not explicitly overridden still apply.
    assert settings["got_reexpand_enabled"] is True
    assert settings["tool_failure_recovery_enabled"] is True
    # Explicit override wins over the profile's True.
    assert settings["got_step_confidence_reexpand_enabled"] is False
    # Explicit override outside the profile applies too.
    assert settings["got_backtrack_enabled"] is True


def test_unknown_arm_warns_and_is_noop(caplog):
    settings = _base()
    with caplog.at_level(logging.WARNING):
        _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "nonexistent"})
    assert settings == _base()
    assert any("Unknown IDEA_TEST_ARM" in rec.message for rec in caplog.records)


def test_arm_profiles_registry_keys_are_valid_settings_keys():
    # Guard against typos: every jsonkey referenced by a profile must be one this
    # module knows how to represent (i.e. present in the "full" superset or the base
    # settings fixture above).
    known_keys = set(_base().keys())
    for arm_name, profile in _GOT_ARM_PROFILES.items():
        for key in profile:
            assert key in known_keys, f"{arm_name} references unknown key {key}"
