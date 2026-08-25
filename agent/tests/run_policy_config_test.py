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


def test_sibling_context_delta_defaults_to_false_when_the_key_is_absent():
    assert RunPolicy.from_settings({}).sibling_context_delta is False


def test_sibling_context_delta_resolves_from_settings():
    resolved = RunPolicy.from_settings({"run_policy_sibling_context_delta": True})
    assert resolved.sibling_context_delta is True


def test_shipped_settings_leave_the_sibling_context_delta_off():
    """Absent by design; the expansion prompt is byte-identical without it."""
    settings = load_idea_dag_settings()
    assert "run_policy_sibling_context_delta" not in settings
    assert RunPolicy.from_settings(settings).sibling_context_delta is False


def test_evidence_store_mode_defaults_to_off_when_the_key_is_absent():
    assert RunPolicy.from_settings({}).evidence_store_mode == "off"


def test_evidence_store_mode_resolves_from_settings():
    resolved = RunPolicy.from_settings({"run_policy_evidence_store_mode": "observe"})
    assert resolved.evidence_store_mode == "observe"


def test_shipped_settings_leave_the_evidence_store_off():
    """Absent by design; "observe" costs one extra LLM call per successful visit."""
    settings = load_idea_dag_settings()
    assert "run_policy_evidence_store_mode" not in settings
    assert RunPolicy.from_settings(settings).evidence_store_mode == "off"


def test_deterministic_merge_view_defaults_to_false_when_the_key_is_absent():
    assert RunPolicy.from_settings({}).deterministic_merge_view is False


def test_deterministic_merge_view_resolves_from_settings():
    resolved = RunPolicy.from_settings({"run_policy_deterministic_merge_view": True})
    assert resolved.deterministic_merge_view is True


def test_shipped_settings_leave_the_deterministic_merge_view_off():
    """Absent by design; the view also needs ``evidence_store_mode == "observe"`` to do anything."""
    settings = load_idea_dag_settings()
    assert "run_policy_deterministic_merge_view" not in settings
    assert RunPolicy.from_settings(settings).deterministic_merge_view is False


def test_merge_uses_evidence_view_defaults_to_false_when_the_key_is_absent():
    assert RunPolicy.from_settings({}).merge_uses_evidence_view is False


def test_merge_uses_evidence_view_resolves_from_settings():
    resolved = RunPolicy.from_settings({"run_policy_merge_uses_evidence_view": True})
    assert resolved.merge_uses_evidence_view is True


def test_shipped_settings_leave_the_merge_evidence_view_off():
    """Absent by design; it also needs the merge view AND the store to render anything."""
    settings = load_idea_dag_settings()
    assert "run_policy_merge_uses_evidence_view" not in settings
    assert RunPolicy.from_settings(settings).merge_uses_evidence_view is False


def test_deficit_driven_injection_defaults_to_false_when_the_key_is_absent():
    assert RunPolicy.from_settings({}).deficit_driven_injection is False


def test_deficit_driven_injection_resolves_from_settings():
    resolved = RunPolicy.from_settings({"run_policy_deficit_driven_injection": True})
    assert resolved.deficit_driven_injection is True


def test_shipped_settings_leave_the_deficit_driven_injection_off():
    """Absent by design; it also needs ``ledger_mode == "observe"`` to inject anything."""
    settings = load_idea_dag_settings()
    assert "run_policy_deficit_driven_injection" not in settings
    assert RunPolicy.from_settings(settings).deficit_driven_injection is False


def test_idea_config_exposes_the_group():
    cfg = IdeaConfig.from_settings({})
    assert isinstance(cfg.run_policy, RunPolicy)
    assert cfg.run_policy.ledger_mode == "off"

    assert cfg.run_policy.search_must_yield_visit is False

    observed = IdeaConfig.from_settings({"run_policy_ledger_mode": "observe"})
    assert observed.run_policy.ledger_mode == "observe"

    armed = IdeaConfig.from_settings({"run_policy_search_must_yield_visit": True})
    assert armed.run_policy.search_must_yield_visit is True

    assert cfg.run_policy.sibling_context_delta is False
    delta = IdeaConfig.from_settings({"run_policy_sibling_context_delta": True})
    assert delta.run_policy.sibling_context_delta is True

    assert cfg.run_policy.evidence_store_mode == "off"
    store = IdeaConfig.from_settings({"run_policy_evidence_store_mode": "observe"})
    assert store.run_policy.evidence_store_mode == "observe"

    assert cfg.run_policy.deterministic_merge_view is False
    view = IdeaConfig.from_settings({"run_policy_deterministic_merge_view": True})
    assert view.run_policy.deterministic_merge_view is True

    assert cfg.run_policy.merge_uses_evidence_view is False
    consumed = IdeaConfig.from_settings({"run_policy_merge_uses_evidence_view": True})
    assert consumed.run_policy.merge_uses_evidence_view is True

    assert cfg.run_policy.deficit_driven_injection is False
    deficit = IdeaConfig.from_settings({"run_policy_deficit_driven_injection": True})
    assert deficit.run_policy.deficit_driven_injection is True


def test_group_follows_the_frozen_dataclass_convention():
    assert dataclasses.is_dataclass(RunPolicy)
    assert RunPolicy.__dataclass_params__.frozen
