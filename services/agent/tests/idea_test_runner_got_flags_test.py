"""Tests for the per-run ``IDEA_TEST_GOT_*`` experiment toggles
(``idea_test_runner._apply_got_experiment_overrides``).

These flags let a research run flip a dormant adaptive mechanism on/off without
editing the checked-in ``idea_dag_settings.json`` (whose defaults stay OFF). The
contract: truthy enables, explicit falsey forces off, ABSENT leaves the loaded
default untouched. Pinned here for the confidence->action loop toggle
``IDEA_TEST_GOT_CONFIDENCE_REEXPAND`` (and its siblings for coverage).
"""
from __future__ import annotations

from agent.app.idea_test_runner import _apply_got_experiment_overrides


def _base():
    # Mirrors the shipped defaults: every adaptive mechanism OFF.
    return {
        "got_reexpand_enabled": False,
        "got_step_confidence_judge_enabled": False,
        "got_step_confidence_judge_sample_every": 1,
        "got_step_confidence_reexpand_enabled": False,
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
