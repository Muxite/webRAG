"""``RunPolicy``: profile-level run semantics as a typed config group.

The seam for run-scoped flags that are decided ONCE per mandate (rather than per
expansion/merge/leaf call) — starting with ``ledger_mode``, which a later change
reads to decide whether the task ledger observes a run. Nothing consumes it yet,
so the load-bearing assertions here are that it resolves from an absent key
(default ``"off"``) and that the aggregate exposes it under the same snake_case
attribute convention every other group uses (``.got``, ``.expansion``, ...).
"""

from __future__ import annotations

import dataclasses

from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.config import IdeaConfig, RunPolicy


def test_ledger_mode_defaults_to_off_when_the_key_is_absent():
    assert RunPolicy.from_settings({}).ledger_mode == "off"


def test_ledger_mode_resolves_from_settings():
    assert RunPolicy.from_settings({"run_policy_ledger_mode": "observe"}).ledger_mode == "observe"


def test_shipped_settings_leave_the_ledger_off():
    """The key is deliberately absent from ``idea_dag_settings.json``: absent-safe."""
    settings = load_idea_dag_settings()
    assert "run_policy_ledger_mode" not in settings
    assert RunPolicy.from_settings(settings).ledger_mode == "off"


def test_search_must_yield_visit_defaults_to_false_when_the_key_is_absent():
    assert RunPolicy.from_settings({}).search_must_yield_visit is False


def test_search_must_yield_visit_resolves_from_settings():
    resolved = RunPolicy.from_settings({"run_policy_search_must_yield_visit": True})
    assert resolved.search_must_yield_visit is True


def test_shipped_settings_leave_the_empty_search_remediation_off():
    """Absent by design, like ``ledger_mode``: flag-off is the byte-identical baseline."""
    settings = load_idea_dag_settings()
    assert "run_policy_search_must_yield_visit" not in settings
    assert RunPolicy.from_settings(settings).search_must_yield_visit is False


def test_idea_config_exposes_the_group():
    cfg = IdeaConfig.from_settings({})
    assert isinstance(cfg.run_policy, RunPolicy)
    assert cfg.run_policy.ledger_mode == "off"

    assert cfg.run_policy.search_must_yield_visit is False

    observed = IdeaConfig.from_settings({"run_policy_ledger_mode": "observe"})
    assert observed.run_policy.ledger_mode == "observe"

    armed = IdeaConfig.from_settings({"run_policy_search_must_yield_visit": True})
    assert armed.run_policy.search_must_yield_visit is True


def test_group_follows_the_frozen_dataclass_convention():
    assert dataclasses.is_dataclass(RunPolicy)
    assert RunPolicy.__dataclass_params__.frozen
