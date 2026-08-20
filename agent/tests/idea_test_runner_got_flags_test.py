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
        "got_contract_reexpand_enabled": False,
        "got_contract_veto_requires_datum_enabled": False,
        "got_reexpand_corrective_context_enabled": False,
        "got_backtrack_enabled": False,
        "native_confidence_early_exit_enabled": False,
        "native_confidence_early_exit_margin": 0.05,
        "connector_retry_on_failure_enabled": False,
        "tool_failure_recovery_enabled": False,
        "native_vote_k_enabled": False,
        "native_vote_k": 1,
        "expansion_expect_contract_enabled": False,
        "plan_library_enabled": False,
        "plan_library_auto_enabled": False,
        "plan_library_action_enabled": False,
        "native_reasoning_effort_discipline_enabled": False,
        "price_tier_param_tiering_enabled": False,
        # live-proven 2026-08-06, shipped default flipped ON 2026-08-14 — the one exception to
        # this fixture's "every adaptive mechanism OFF" mirror. `baseline` pins it back to
        # False explicitly (see test_arm_baseline_matches_shipped_defaults) so the "adaptive
        # OFF" ladder rung stays byte-identical regardless of this default.
        "expansion_input_output_framing_enabled": True,
        # finalize grounding gate (a validity gate: every ARM turns it on, but the shipped
        # JSON default — what this fixture mirrors — is OFF)
        "final_require_grounding": False,
        # post-synthesis reconcile chain (finalize; the `max_burn` arm turns these on)
        "final_recompute_enabled": False,
        "final_verify_enabled": False,
        "final_variations_enabled": False,
        "final_variations_k": 3,
        # structural knobs the `max_burn` arm tunes (real settings keys; defaults here)
        "max_steps": 50,
        "got_candidate_coverage_enabled": False,
        "visit_link_query_top_k": 15,
        "max_links_per_visit": 20,
        "got_beam_max": 5,
        "max_branching": 5,
        "grounding_max_replans": 2,
        # reason-first field ordering (reasoning before verdict), see
        # idea_test_runner_reason_first_flags_test.py for their dedicated coverage.
        "merge_goal_evaluation_first_enabled": False,
        "verify_reason_first_enabled": False,
        "got_reexpand_followup_reason_first_enabled": False,
    }


def test_absent_flags_preserve_defaults():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={})
    assert settings == _base(), "no env flags -> loaded defaults untouched"


def test_max_burn_arm_cranks_productive_levers_only():
    """The `max_burn` arm turns UP re-expansion depth + supporting knobs, and keeps the
    net-negative/inert mechanisms explicitly OFF (they must not sneak back as burn)."""
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "max_burn"})
    # productive burn: deeper re-expansion + the room/coverage it needs to execute
    assert settings["got_reexpand_enabled"] is True
    assert settings["got_reexpand_max_iterations"] == 4    # the core lever, up from 2
    assert settings["max_steps"] == 90                     # or depth-4 re-expansion starves
    assert settings["got_candidate_coverage_enabled"] is True
    assert settings["got_beam_max"] == 7 and settings["max_branching"] == 7
    assert settings["grounding_max_replans"] == 3
    # net-negative / inert / bugged mechanisms must stay OFF
    assert settings["native_vote_k_enabled"] is False and settings["native_vote_k"] == 1
    assert settings["got_backtrack_enabled"] is False
    assert settings["expansion_expect_contract_enabled"] is False
    assert settings["native_reasoning_effort_discipline_enabled"] is False
    assert settings["price_tier_param_tiering_enabled"] is False


def test_max_burn_carries_final_reconcile_chain_flags():
    """max_burn is where we spend to close the SYNTHESIS gap: the finalize reconcile chain
    (recompute + verify + variation ensemble) must be on, k-vote off (they overlap)."""
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "max_burn"})
    assert settings["final_recompute_enabled"] is True
    assert settings["final_verify_enabled"] is True
    assert settings["final_variations_enabled"] is True
    assert settings["final_variations_k"] == 3
    # variations supersede k-vote — the overlapping k-vote path stays off.
    assert settings["native_vote_k_enabled"] is False


def test_max_burn_is_strictly_more_reexpansion_than_good_adaptive():
    """max_burn must dominate good_adaptive on the productive lever, else it isn't a burn."""
    ga = _GOT_ARM_PROFILES["good_adaptive"]
    mb = _GOT_ARM_PROFILES["max_burn"]
    assert mb["got_reexpand_max_iterations"] > ga.get("got_reexpand_max_iterations", 2)
    # and it must NOT enable the mechanisms good_adaptive intentionally leaves off
    assert mb.get("native_vote_k_enabled") is False
    assert mb.get("got_backtrack_enabled") is False


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


def test_native_early_exit_bool_override():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_NATIVE_EARLY_EXIT": "1"})
    assert settings["native_confidence_early_exit_enabled"] is True


def test_native_early_exit_explicit_falsey_forces_off():
    settings = _base()
    settings["native_confidence_early_exit_enabled"] = True
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_NATIVE_EARLY_EXIT": "0"})
    assert settings["native_confidence_early_exit_enabled"] is False


def test_native_early_exit_margin_float_override():
    settings = _base()
    _apply_got_experiment_overrides(
        settings, environ={"IDEA_TEST_NATIVE_EARLY_EXIT_MARGIN": "0.15"}
    )
    assert settings["native_confidence_early_exit_margin"] == 0.15


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


def test_plan_library_bool_overrides_are_independent():
    """Three switches, not one: an operator can arm the subsystem without arming either
    consumer, and can run the automatic short-circuit and the on-demand action separately."""
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_PLAN_LIBRARY": "1"})
    assert settings["plan_library_enabled"] is True
    assert settings["plan_library_auto_enabled"] is False
    assert settings["plan_library_action_enabled"] is False

    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_PLAN_LIBRARY_AUTO": "1"})
    assert settings["plan_library_auto_enabled"] is True
    assert settings["plan_library_action_enabled"] is False

    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_PLAN_LIBRARY_ACTION": "1"})
    assert settings["plan_library_action_enabled"] is True

    _apply_got_experiment_overrides(
        settings,
        environ={
            "IDEA_TEST_PLAN_LIBRARY": "0",
            "IDEA_TEST_PLAN_LIBRARY_AUTO": "false",
            "IDEA_TEST_PLAN_LIBRARY_ACTION": "0",
        },
    )
    assert settings["plan_library_enabled"] is False
    assert settings["plan_library_auto_enabled"] is False
    assert settings["plan_library_action_enabled"] is False


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


def test_expansion_io_framing_and_echo_retry_are_independent_bool_overrides():
    """Two levers, not one: the prompt fix (framing) and its safety net (retry) must be armable
    separately so a live A/B can attribute the effect to the prompt alone."""
    framing_only = _base()
    _apply_got_experiment_overrides(
        framing_only, environ={"IDEA_TEST_EXPANSION_IO_FRAMING": "1"}
    )
    assert framing_only["expansion_input_output_framing_enabled"] is True
    assert "expansion_echo_retry_enabled" not in framing_only

    retry_only = _base()
    _apply_got_experiment_overrides(
        retry_only, environ={"IDEA_TEST_EXPANSION_ECHO_RETRY": "1"}
    )
    assert retry_only["expansion_echo_retry_enabled"] is True
    # Independence: arming retry alone must not touch framing's own value either way.
    assert retry_only["expansion_input_output_framing_enabled"] == _base()["expansion_input_output_framing_enabled"]

    both_off = _base()
    _apply_got_experiment_overrides(
        both_off,
        environ={"IDEA_TEST_EXPANSION_IO_FRAMING": "0", "IDEA_TEST_EXPANSION_ECHO_RETRY": "0"},
    )
    assert both_off["expansion_input_output_framing_enabled"] is False
    assert both_off["expansion_echo_retry_enabled"] is False


def test_all_new_flags_blank_are_noop():
    settings = _base()
    _apply_got_experiment_overrides(
        settings,
        environ={
            "IDEA_TEST_GOT_REEXPAND_MAX_ITER": "",
            "IDEA_TEST_GOT_CORRECTIVE_CONTEXT": "  ",
            "IDEA_TEST_GOT_CONFIDENCE_THRESHOLD": "",
            "IDEA_TEST_GOT_BACKTRACK": "",
            "IDEA_TEST_NATIVE_EARLY_EXIT": "",
            "IDEA_TEST_NATIVE_EARLY_EXIT_MARGIN": "  ",
            "IDEA_TEST_CONNECTOR_RETRY": "",
            "IDEA_TEST_TOOL_FAILURE_RECOVERY": "",
            "IDEA_TEST_NATIVE_VOTE_K": "",
            "IDEA_TEST_EXPECT_CONTRACT": "",
            "IDEA_TEST_EXPANSION_IO_FRAMING": "",
            "IDEA_TEST_EXPANSION_ECHO_RETRY": "  ",
            "IDEA_TEST_PLAN_LIBRARY": "",
            "IDEA_TEST_PLAN_LIBRARY_AUTO": "  ",
            "IDEA_TEST_PLAN_LIBRARY_ACTION": "",
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
    # Every adaptive LEVER is forced back to the shipped default, with two deliberate
    # exceptions: the finalize grounding gate (a validity gate every arm carries — an answer
    # produced with zero opened pages must not be presented as researched in the control arm
    # either, or the comparison credits one arm's hallucinations), and the expansion
    # input/output framing fix, which the shipped JSON now defaults ON but `baseline` pins
    # back to False so the "adaptive OFF" rung stays byte-identical to prior behavior
    # regardless of what the global default becomes.
    expected = _base()
    expected["final_require_grounding"] = True
    expected["expansion_input_output_framing_enabled"] = False
    assert settings == expected


def test_arm_good_adaptive_expands_expected_flags():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={"IDEA_TEST_ARM": "good_adaptive"})
    assert settings["got_reexpand_enabled"] is True
    assert settings["got_reexpand_max_iterations"] == 2
    assert settings["got_step_confidence_judge_enabled"] is True
    assert settings["got_step_confidence_reexpand_enabled"] is True
    assert settings["got_reexpand_corrective_context_enabled"] is True
    assert settings["tool_failure_recovery_enabled"] is True
    # F33: contract satisfaction replaces the anti-calibrated judge as the re-expansion
    # control signal; the grounding gate rides along as a validity gate.
    assert settings["got_contract_reexpand_enabled"] is True
    assert settings["final_require_grounding"] is True
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


def test_contract_veto_requires_datum_truthy_enables():
    """F35: the toggle that stops a subject-only contract vetoing the confidence loop."""
    for truthy in ("1", "true", "yes", "on", "TRUE"):
        settings = _base()
        _apply_got_experiment_overrides(
            settings, environ={"IDEA_TEST_GOT_CONTRACT_VETO_REQUIRES_DATUM": truthy}
        )
        assert settings["got_contract_veto_requires_datum_enabled"] is True, truthy


def test_contract_veto_requires_datum_absent_leaves_the_default():
    settings = _base()
    _apply_got_experiment_overrides(settings, environ={})
    assert settings["got_contract_veto_requires_datum_enabled"] is False


def test_contract_veto_requires_datum_explicit_falsey_forces_off():
    settings = _base()
    settings["got_contract_veto_requires_datum_enabled"] = True
    for falsey in ("0", "false", "no", "off"):
        s = dict(settings)
        _apply_got_experiment_overrides(
            s, environ={"IDEA_TEST_GOT_CONTRACT_VETO_REQUIRES_DATUM": falsey}
        )
        assert s["got_contract_veto_requires_datum_enabled"] is False, falsey
