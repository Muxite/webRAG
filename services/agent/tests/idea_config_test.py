"""Unit tests for the typed settings views in idea_policies/config.py.

These pin the migration contract: typed views must reproduce the exact defaults
and coercion that the old per-call-site ``settings.get(key, default)`` code did,
including for keys deliberately absent from idea_dag_settings.json.
"""

from __future__ import annotations

import pytest

from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_policies.config import (
    GoTConfig,
    IdeaConfig,
    TimeoutConfig,
    EvaluationConfig,
    validate_settings,
)


def test_defaults_match_legacy_fallbacks():
    # Empty settings -> every field falls back to its declared default, which
    # mirrors the hard-coded fallback previously used at each call site.
    cfg = GoTConfig.from_settings({})
    assert cfg.embed_on_create is True
    assert cfg.dedup_similarity_threshold == 0.85
    assert cfg.beam_min == 2
    assert cfg.beam_max == 5
    # Keys absent from idea_dag_settings.json must still resolve to their default.
    assert cfg.adaptive_policies is True
    assert cfg.dedup_threshold_min == 0.75
    assert cfg.beam_target_spread == 0.4
    assert cfg.prune_stddev_factor == 1.0
    # Defaults now mirror the shipped JSON (got_backtrack_enabled=False,
    # got_backtrack_dead_end_threshold=5).
    assert cfg.backtrack_enabled is False
    assert cfg.backtrack_dead_end_threshold == 5


def test_settings_override_and_coercion():
    cfg = GoTConfig.from_settings({
        "got_backtrack_enabled": False,
        "got_backtrack_dead_end_threshold": "5",   # str -> int coercion
        "got_beam_score_high": 1,                    # int -> float coercion
        "got_dedup_enabled": 0,                      # falsy -> bool
    })
    assert cfg.backtrack_enabled is False
    assert cfg.backtrack_dead_end_threshold == 5
    assert isinstance(cfg.backtrack_dead_end_threshold, int)
    assert cfg.beam_score_high == 1.0
    assert isinstance(cfg.beam_score_high, float)
    assert cfg.dedup_enabled is False


def test_optional_string_models_pass_through():
    # Absent -> None; empty-string (as shipped in JSON) stays falsy.
    assert GoTConfig.from_settings({}).telemetry_routing_score_model is None
    assert GoTConfig.from_settings(
        {"got_telemetry_routing_score_model": ""}
    ).telemetry_routing_score_model == ""
    assert GoTConfig.from_settings(
        {"got_telemetry_routing_score_model": "gpt-5-mini"}
    ).telemetry_routing_score_model == "gpt-5-mini"


def test_step_confidence_judge_fields_default_off():
    # Opt-in instrumentation: default OFF, sample every step, no model override.
    cfg = GoTConfig.from_settings({})
    assert cfg.step_confidence_judge_enabled is False
    assert cfg.step_confidence_judge_temperature == 0.0
    assert cfg.step_confidence_judge_sample_every == 1
    assert cfg.step_confidence_judge_model is None


def test_step_confidence_judge_overrides_and_coercion():
    cfg = GoTConfig.from_settings({
        "got_step_confidence_judge_enabled": 1,           # truthy -> bool
        "got_step_confidence_judge_temperature": "0.3",   # str -> float
        "got_step_confidence_judge_sample_every": "2",    # str -> int
        "got_step_confidence_judge_model": "gpt-4.1-nano",
    })
    assert cfg.step_confidence_judge_enabled is True
    assert cfg.step_confidence_judge_temperature == 0.3
    assert isinstance(cfg.step_confidence_judge_sample_every, int)
    assert cfg.step_confidence_judge_sample_every == 2
    assert cfg.step_confidence_judge_model == "gpt-4.1-nano"


def test_step_confidence_reexpand_fields_default_off():
    # Confidence->action loop: default OFF with a 0.5 threshold. A settings load that
    # omits both keys must preserve the current (off) behavior byte-for-byte.
    cfg = GoTConfig.from_settings({})
    assert cfg.step_confidence_reexpand_enabled is False
    assert cfg.step_confidence_reexpand_threshold == 0.5
    # The shipped production JSON ships both keys OFF too.
    prod = GoTConfig.from_settings(load_idea_dag_settings())
    assert prod.step_confidence_reexpand_enabled is False
    assert prod.step_confidence_reexpand_threshold == 0.5


def test_step_confidence_reexpand_overrides_and_coercion():
    cfg = GoTConfig.from_settings({
        "got_step_confidence_reexpand_enabled": 1,        # truthy -> bool
        "got_step_confidence_reexpand_threshold": "0.4",  # str -> float
    })
    assert cfg.step_confidence_reexpand_enabled is True
    assert cfg.step_confidence_reexpand_threshold == 0.4
    assert isinstance(cfg.step_confidence_reexpand_threshold, float)


def test_frozen_view_is_immutable():
    import dataclasses
    cfg = GoTConfig.from_settings({})
    try:
        cfg.beam_max = 9  # type: ignore[misc]
        assert False, "GoTConfig should be frozen/immutable"
    except dataclasses.FrozenInstanceError:
        pass


def test_expansion_expect_contract_defaults_off():
    # Opt-in decomposition contract: default OFF, both from an empty load and from the
    # shipped production JSON (so the default expansion prompt/schema stay byte-identical).
    from agent.app.idea_policies.config import ExpansionConfig
    assert ExpansionConfig.from_settings({}).expect_contract_enabled is False
    assert ExpansionConfig.from_settings(
        load_idea_dag_settings()
    ).expect_contract_enabled is False
    # Truthy override coerces to bool.
    assert ExpansionConfig.from_settings(
        {"expansion_expect_contract_enabled": 1}
    ).expect_contract_enabled is True


def test_expansion_input_output_framing_flags_default_off():
    # Prompt-hygiene lever + its retry safety net: both OFF from an empty load and from the
    # shipped JSON, so the expansion prompt bytes are unchanged until an operator arms them.
    # They are SEPARATE flags on purpose (independent ablation of prompt fix vs safety net).
    from agent.app.idea_policies.config import ExpansionConfig
    empty = ExpansionConfig.from_settings({})
    assert empty.input_output_framing_enabled is False
    assert empty.echo_retry_enabled is False
    prod = ExpansionConfig.from_settings(load_idea_dag_settings())
    assert prod.input_output_framing_enabled is False
    assert prod.echo_retry_enabled is False
    # Truthy overrides coerce to bool, independently.
    framing_only = ExpansionConfig.from_settings({"expansion_input_output_framing_enabled": 1})
    assert framing_only.input_output_framing_enabled is True
    assert framing_only.echo_retry_enabled is False
    retry_only = ExpansionConfig.from_settings({"expansion_echo_retry_enabled": "true"})
    assert retry_only.echo_retry_enabled is True
    assert retry_only.input_output_framing_enabled is False


def test_native_tiering_flags_default_off():
    # A3b + A5 executor-tiering flags: default OFF from an empty load and the shipped JSON,
    # so the native micro-prompt token/effort path stays byte-identical.
    from agent.app.idea_policies.config import ActionConfig
    empty = ActionConfig.from_settings({})
    assert empty.native_reasoning_effort_discipline_enabled is False
    assert empty.native_reasoning_min_tokens_floor == 2048
    assert empty.price_tier_param_tiering_enabled is False
    prod = ActionConfig.from_settings(load_idea_dag_settings())
    assert prod.native_reasoning_effort_discipline_enabled is False
    assert prod.price_tier_param_tiering_enabled is False
    # Overrides coerce.
    on = ActionConfig.from_settings({
        "native_reasoning_effort_discipline_enabled": 1,
        "native_reasoning_min_tokens_floor": "4096",
        "price_tier_param_tiering_enabled": "true",
    })
    assert on.native_reasoning_effort_discipline_enabled is True
    assert on.native_reasoning_min_tokens_floor == 4096
    assert on.price_tier_param_tiering_enabled is True


def test_tool_failure_recovery_flags_default_off():
    # C1a tool-failure recovery flags: default OFF from an empty load and the shipped JSON.
    from agent.app.idea_policies.config import ActionConfig
    empty = ActionConfig.from_settings({})
    assert empty.connector_retry_on_failure_enabled is False
    assert empty.connector_retry_max_attempts == 2
    assert empty.connector_retry_backoff_seconds == 0.5
    assert empty.tool_failure_recovery_enabled is False
    prod = ActionConfig.from_settings(load_idea_dag_settings())
    assert prod.connector_retry_on_failure_enabled is False
    assert prod.tool_failure_recovery_enabled is False
    # Overrides coerce.
    on = ActionConfig.from_settings({
        "connector_retry_on_failure_enabled": 1,
        "connector_retry_max_attempts": "3",
        "connector_retry_backoff_seconds": "1.5",
        "tool_failure_recovery_enabled": "true",
    })
    assert on.connector_retry_on_failure_enabled is True
    assert on.connector_retry_max_attempts == 3
    assert on.connector_retry_backoff_seconds == 1.5
    assert on.tool_failure_recovery_enabled is True


def test_native_vote_k_flags_default_off():
    # C1b terminal-answer vote flags: default OFF / k=1 (single extraction = current behavior).
    from agent.app.idea_policies.config import FinalConfig
    empty = FinalConfig.from_settings({})
    assert empty.native_vote_k_enabled is False
    assert empty.native_vote_k == 1
    prod = FinalConfig.from_settings(load_idea_dag_settings())
    assert prod.native_vote_k_enabled is False
    assert prod.native_vote_k == 1
    on = FinalConfig.from_settings({"native_vote_k_enabled": 1, "native_vote_k": "3"})
    assert on.native_vote_k_enabled is True
    assert on.native_vote_k == 3
    # Local size-band refinement of the weak band: default OFF, tapering k with model size.
    assert empty.native_vote_k_size_band_enabled is False
    assert prod.native_vote_k_size_band_enabled is False
    assert (empty.native_vote_k_local_tiny, empty.native_vote_k_local_small,
            empty.native_vote_k_local_medium, empty.native_vote_k_local_large) == (4, 3, 2, 1)
    banded = FinalConfig.from_settings({"native_vote_k_size_band_enabled": 1,
                                        "native_vote_k_local_medium": "5"})
    assert banded.native_vote_k_size_band_enabled is True
    assert banded.native_vote_k_local_medium == 5


def test_final_reconcile_chain_flags_default_off():
    # Post-synthesis reconcile chain: default OFF from an empty load and the shipped JSON, so the
    # finalize path stays byte-identical. The shape gate defaults to the answer-shaped label set.
    from agent.app.idea_policies.config import FinalConfig
    empty = FinalConfig.from_settings({})
    assert empty.final_recompute_enabled is False
    assert empty.final_verify_enabled is False
    assert empty.final_variations_enabled is False
    assert empty.final_variations_k == 3
    assert set(empty.final_recompute_shapes) == {
        "computation", "count", "argmax", "disambiguation", "single_value"
    }
    prod = FinalConfig.from_settings(load_idea_dag_settings())
    assert prod.final_recompute_enabled is False
    assert prod.final_verify_enabled is False
    assert prod.final_variations_enabled is False
    assert "single_value" in prod.final_recompute_shapes
    # Overrides coerce; the shape list is read straight from JSON.
    on = FinalConfig.from_settings({
        "final_recompute_enabled": 1,
        "final_verify_enabled": "true",
        "final_variations_enabled": 1,
        "final_variations_k": "5",
        "final_recompute_shapes": ["count", "argmax"],
    })
    assert on.final_recompute_enabled is True
    assert on.final_verify_enabled is True
    assert on.final_variations_enabled is True
    assert on.final_variations_k == 5
    assert list(on.final_recompute_shapes) == ["count", "argmax"]


def test_plan_library_flags_default_off_and_carry_no_threshold():
    # Retrieval-augmented planning: both switches OFF from an empty load and from the shipped
    # JSON, so the expansion path stays byte-identical until an operator arms them.
    from agent.app.idea_policies.config import PlanLibraryConfig
    import dataclasses

    empty = PlanLibraryConfig.from_settings({})
    assert (empty.enabled, empty.auto_enabled, empty.action_enabled) == (False, False, False)
    prod = PlanLibraryConfig.from_settings(load_idea_dag_settings())
    assert (prod.enabled, prod.auto_enabled, prod.action_enabled) == (False, False, False)
    on = PlanLibraryConfig.from_settings(
        {
            "plan_library_enabled": 1,
            "plan_library_auto_enabled": "true",
            "plan_library_action_enabled": 1,
        }
    )
    assert (on.enabled, on.auto_enabled, on.action_enabled) == (True, True, True)
    # The two sub-flags are independent: auto-only and on-demand-only are both valid runs.
    auto_only = PlanLibraryConfig.from_settings(
        {"plan_library_enabled": True, "plan_library_auto_enabled": True}
    )
    assert auto_only.auto_enabled is True and auto_only.action_enabled is False
    # The calibrated decision constants live in plan_library/retrieval.py and are trusted as
    # the verdict; a second engine-level similarity gate would silently disagree with them.
    assert not any(
        "threshold" in f.name or "similarity" in f.name
        for f in dataclasses.fields(PlanLibraryConfig)
    )


def test_aggregate_builds_from_production_settings():
    # The shipped JSON must build cleanly and reflect its production values.
    cfg = IdeaConfig.from_settings(load_idea_dag_settings())
    assert cfg.engine.best_first_global is True      # JSON value, not the legacy False
    assert cfg.action.max_retries == 2               # JSON value, not the legacy 0
    assert cfg.memory.document_chunk_size == 4000     # JSON value, not the legacy 2000
    assert cfg.final.chroma_results == 10
    assert cfg.timeouts.action == 20


def test_timeout_for_action_falls_back_to_generic():
    t = TimeoutConfig.from_settings({})
    assert t.for_action("search") == 15
    assert t.for_action("visit") == 20
    # Actions without a dedicated timeout (save/think/merge/verify) fall back.
    assert t.for_action("save") == t.action
    assert t.for_action("verify") == t.action


def test_evaluation_weight_for_falls_back_to_default():
    e = EvaluationConfig.from_settings({"evaluation_weight_visit": 2.0})
    assert e.weight_for("visit") == 2.0
    assert e.weight_for("search") == 1.0
    assert e.weight_for("nonexistent") == e.weight_default


def test_evaluation_config_folds_legacy_weights():
    # Production settings (JSON defaults are always merged at runtime) must
    # resolve the penalty knobs to the shipped values, and the per-action
    # multipliers default to 1.0. This pins the fold of the old
    # ``EvaluationWeights`` class into the ``EvaluationConfig`` typed view.
    e = EvaluationConfig.from_settings(load_idea_dag_settings())
    assert e.no_action_result_base_score == 0.4
    assert e.no_action_result_score_cap == 0.5
    for action in ("search", "visit", "think", "save", "verify"):
        assert e.weight_for(action) == 1.0


def test_evaluation_weight_for_is_case_insensitive_and_none_safe():
    # Faithful drop-in for the old ``apply_action_weight``: action names are
    # lowercased before lookup and a falsy action falls back to the default.
    e = EvaluationConfig.from_settings({"evaluation_weight_visit": 2.0})
    assert e.weight_for("VISIT") == 2.0
    assert e.weight_for("Visit") == 2.0
    assert e.weight_for(None) == e.weight_default
    assert e.weight_for("") == e.weight_default


def test_finalize_merge_reservations_are_capped(tmp_path, monkeypatch):
    """The finalize/merge completion RESERVATION must stay <= the provider cap.

    OpenRouter holds ``max_completion_tokens x output_price`` against the remaining daily credit
    before running a call, so the old 120000/100000 budgets 402'd (and silently degraded the stage)
    once a cap drained. The cap is enforced at LOAD time so neither an alternate settings file nor
    an ``IDEA_DAG_*_MAX_TOKENS`` env override can reintroduce it.
    """
    import json

    from agent.app.idea_dag_settings import _MAX_TOKENS_RESERVATION_CAP
    from agent.app.idea_policies.config import FinalConfig, MergeConfig

    cap = _MAX_TOKENS_RESERVATION_CAP
    shipped = load_idea_dag_settings()
    assert shipped["final_max_tokens"] <= cap
    assert shipped["merge_max_tokens"] <= cap
    # Typed defaults agree with the shipped JSON (config_drift_test guards this globally).
    assert FinalConfig.from_settings({}).max_tokens <= cap
    assert MergeConfig.from_settings({}).max_tokens <= cap

    # An alternate settings file carrying the old values is clamped on load...
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"final_max_tokens": 120000, "merge_max_tokens": 100000}), encoding="utf-8")
    clamped = load_idea_dag_settings(path)
    assert clamped["final_max_tokens"] == cap
    assert clamped["merge_max_tokens"] == cap

    # ...and so is an env override, while a SMALLER value is left untouched.
    monkeypatch.setenv("IDEA_DAG_FINAL_MAX_TOKENS", "120000")
    monkeypatch.setenv("IDEA_DAG_MERGE_MAX_TOKENS", "4096")
    env_loaded = load_idea_dag_settings(path)
    assert env_loaded["final_max_tokens"] == cap
    assert env_loaded["merge_max_tokens"] == 4096


def test_validate_settings_rejects_bad_type():
    # A non-numeric value for an int knob must fail loudly at validation time.
    with pytest.raises(ValueError):
        validate_settings({"max_total_nodes": "not-a-number"})


def test_validate_settings_accepts_production():
    cfg = validate_settings(load_idea_dag_settings())
    assert isinstance(cfg, IdeaConfig)
