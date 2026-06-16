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
    assert cfg.improve_enabled is True
    assert cfg.dedup_similarity_threshold == 0.85
    assert cfg.beam_min == 2
    assert cfg.beam_max == 5
    # Keys absent from idea_dag_settings.json must still resolve to their default.
    assert cfg.adaptive_policies is True
    assert cfg.dedup_threshold_min == 0.75
    assert cfg.beam_target_spread == 0.4
    assert cfg.prune_stddev_factor == 1.0
    # backtrack default is enabled=True / threshold=3 at the code level even
    # though the shipped JSON disables it; the typed view keeps the code default.
    assert cfg.backtrack_enabled is True
    assert cfg.backtrack_dead_end_threshold == 3


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


def test_frozen_view_is_immutable():
    import dataclasses
    cfg = GoTConfig.from_settings({})
    try:
        cfg.beam_max = 9  # type: ignore[misc]
        assert False, "GoTConfig should be frozen/immutable"
    except dataclasses.FrozenInstanceError:
        pass


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


def test_validate_settings_rejects_bad_type():
    # A non-numeric value for an int knob must fail loudly at validation time.
    with pytest.raises(ValueError):
        validate_settings({"max_total_nodes": "not-a-number"})


def test_validate_settings_accepts_production():
    cfg = validate_settings(load_idea_dag_settings())
    assert isinstance(cfg, IdeaConfig)
