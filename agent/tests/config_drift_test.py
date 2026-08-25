"""Config-drift guard: every ``*Config`` dataclass default must match the value
actually shipped in ``idea_dag_settings.json``.

This bug class -- a dataclass default silently disagreeing with the shipped JSON,
so behaviour flips the moment a JSON key is ever dropped -- has bitten this
project three times already (``improve_enabled``, ``backtrack_enabled``,
``backtrack_dead_end_threshold``). This is a regression guard, not a live bug
hunt: it should pass today.

For every ``*Config`` group in ``idea_policies/config.py`` it resolves each
dataclass field to its JSON key using the *same* mapping the group's
``from_settings`` uses (``_KEYS`` override, or the ``got_`` prefix that
``GoTConfig.from_settings`` hard-codes), then asserts default == shipped value
for every JSON-supplied field. Fields the module documents as intentionally
absent from the JSON keep their default and are pinned by an explicit allow-list
so a *new* undocumented absence also trips this test.
"""

from __future__ import annotations

import dataclasses
import pathlib

from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app import idea_dag_settings as settings_mod
from agent.app.idea_policies import config as config_mod


# JSON keys the module's own docstrings/comments document as intentionally absent
# from idea_dag_settings.json (they rely solely on the dataclass default):
#   * module docstring: semantic_dedup_*, sequential_prune_siblings,
#     final_max_prompt_chars, require_score
#   * GoTConfig docstring: adaptive_policies, dedup_threshold_min/max,
#     beam_target_spread, prune_stddev_factor
INTENTIONALLY_JSON_ABSENT = {
    # GoTConfig
    "got_adaptive_policies",
    "got_dedup_threshold_min",
    "got_dedup_threshold_max",
    "got_beam_target_spread",
    "got_prune_stddev_factor",
    "got_backtrack_dead_end_path_fraction",
    # FinalConfig
    "final_max_prompt_chars",
    # EngineConfig
    "sequential_prune_siblings",
    "semantic_dedup_visits_enabled",
    "semantic_dedup_require_hook_source",
    "waypoint_enabled",
    "parallel_requires_evidence",
    "defer_unresolved_slots",
    "resolved_value_channel_enabled",
    "beam_after_evaluation",
    "evaluate_parallel_siblings",
    # PolicyConfig
    "require_score",
    # RunPolicy: nothing reads the ledger mode yet, so the dataclass default is its
    # only source of truth (documented in the group's own docstring).
    "run_policy_ledger_mode",
    # Same reasoning for the empty-search remediation switch: absent means the shipped
    # settings describe exactly today's behaviour.
    "run_policy_search_must_yield_visit",
    # And for the sibling-context ledger delta, which is inert without the ledger anyway.
    "run_policy_sibling_context_delta",
    # And for the evidence store, whose "observe" mode buys one extra LLM call per visit:
    # absent means the shipped settings cost exactly what they cost today.
    "run_policy_evidence_store_mode",
    # And for the deterministic merge view, which is inert without that store anyway.
    "run_policy_deterministic_merge_view",
    # And for the flag that renders that view into the merge prompt, which is the first
    # of these that can change an answer: absent means the shipped merge prompt is
    # byte-identical to today's.
    "run_policy_merge_uses_evidence_view",
    # And for the deficit-driven injection, which is inert without the ledger and, when
    # armed, is the only one of these that adds NODES: absent means the shipped graph
    # shape is byte-identical to today's.
    "run_policy_deficit_driven_injection",
    # And for the visit URL identity guard, which is the only one of these that can turn a
    # (wrongly) successful visit into a failure: absent means the shipped fallback cascade
    # behaves exactly as it does today.
    "run_policy_visit_url_identity_guard",
    # And for the sequencing identity guard, visit_url_identity_guard's execution-order
    # twin in idea_sequencing.reorder_for_sequential: absent means candidate reordering
    # behaves exactly as it does today (highest-scored/first candidate, no identity check).
    "run_policy_sequencing_identity_guard",
    # And for the coverage entity-conflict check, which is strictly observe-only: absent
    # means no coverage_entity_conflicts field is ever attached to the final payload, and
    # coverage_ratio/finalization_status are untouched either way.
    "run_policy_coverage_entity_conflict_check",
    # And for constrained decoding, which only upgrades a malformed-JSON REPAIR call to real
    # schema-constrained decoding on a confirmed local-Ollama backend: absent means every
    # repair re-ask behaves exactly as it does today (plain json_object, no schema).
    "run_policy_constrained_decoding_enabled",
    # And for the sibling evidence digest, which only ADDS a bounded block to the expansion
    # system prompt: absent means that prompt is byte-identical to today's.
    "run_policy_sibling_evidence_digest_enabled",
    # ToolsConfig: the core menu's shipped source of truth is still the legacy
    # `allowed_actions` key (which callers override per run); shipping the typed
    # override too would let a JSON default clobber those overrides. Documented
    # in ToolsConfig's own docstring.
    "tools_core_actions",
}


def _config_groups():
    """Every ``*Config`` dataclass with a ``from_settings`` classmethod, except
    the ``IdeaConfig`` aggregate (whose fields are sub-configs, not JSON keys)."""
    for obj in vars(config_mod).values():
        if (
            isinstance(obj, type)
            and dataclasses.is_dataclass(obj)
            and hasattr(obj, "from_settings")
            and obj is not config_mod.IdeaConfig
        ):
            yield obj


def _json_key_for(cls, field_name: str) -> str:
    """Resolve a field to its JSON key using the group's own mapping logic.

    Mirrors the two code paths in config.py: ``_build`` reads ``cls._KEYS``
    (falling back to the bare field name), while ``GoTConfig`` -- the only group
    without a ``_KEYS`` ClassVar -- resolves through its own ``json_key`` (the
    ``got_`` prefix, or a ``_NATIVE_KEYS`` override for the ``native_``-prefixed
    A-series flags that live in the GoT group).
    """
    key_map = getattr(cls, "_KEYS", None)
    if key_map is None:
        return cls.json_key(field_name)
    return key_map.get(field_name, field_name)


def _equivalent(default, shipped) -> bool:
    if default == shipped:
        return True
    # Sequence sentinel: JSON has no tuple type, so a tuple dataclass default (used for
    # an immutable, hashable frozen-view field like final_recompute_shapes) ships as a
    # JSON array. Compare element-wise; this is not a meaningful disagreement.
    if isinstance(default, tuple) and isinstance(shipped, list):
        return list(default) == shipped
    # Optional-model sentinel: the JSON ships "" where the dataclass uses None to
    # mean "no model override"; both are the falsy "unset" value (and _coerce
    # passes either straight through), so this is not a meaningful disagreement.
    return {default, shipped} <= {None, ""}


def test_no_config_drift():
    settings = load_idea_dag_settings()

    observed_absent: set[str] = set()
    for cls in _config_groups():
        defaults = cls()
        for field in dataclasses.fields(cls):
            json_key = _json_key_for(cls, field.name)
            default = getattr(defaults, field.name)
            if json_key not in settings:
                observed_absent.add(json_key)
                continue
            shipped = settings[json_key]
            assert _equivalent(default, shipped), (
                f"{cls.__name__}.{field.name}: dataclass default {default!r} "
                f"disagrees with idea_dag_settings.json['{json_key}'] = {shipped!r}"
            )

    # A field silently dropping out of the JSON without being documented as
    # intentionally-absent is itself a drift risk; pin the set both ways.
    assert observed_absent == INTENTIONALLY_JSON_ABSENT, (
        "JSON-absent config fields drifted from the documented allow-list. "
        f"unexpected absent: {observed_absent - INTENTIONALLY_JSON_ABSENT}; "
        f"documented-but-now-present: {INTENTIONALLY_JSON_ABSENT - observed_absent}"
    )


def test_no_per_arm_settings_snapshots():
    """``idea_dag_settings.json`` is the ONLY settings file; arms are overlays in code.

    ``idea_dag_settings.baseline.json`` and ``idea_dag_settings.good_adaptive.json`` existed
    from the first GoT commit, before ``IDEA_TEST_ARM``/``_GOT_ARM_PROFILES``, as hand-copied
    full snapshots of the whole settings dict with a handful of arm flags flipped. Nothing
    loaded them, no gate synced them, and by 2026-08-20 they were 48-51 keys behind the shipped
    JSON and disagreed with the arm profiles they were named after (both claimed k-vote on,
    which `good_adaptive` deliberately excludes as net-negative) -- stale enough to have already
    seeded a false claim in ADAPTIVE_ENGINE.md. Deleted; an arm is
    ``idea_dag_settings.json`` + ``_GOT_ARM_PROFILES[arm]``, resolved by the test runner.

    A snapshot per arm cannot be kept honest by hand, so re-adding one is the bug: this pins
    the deletion rather than trying to diff-sync files that would drift again on the next flag.
    """
    settings_dir = pathlib.Path(settings_mod.__file__).resolve().parent
    snapshots = sorted(
        p.name
        for p in settings_dir.glob("idea_dag_settings.*.json")
        if p.name != "idea_dag_settings.json"
    )
    assert not snapshots, (
        "per-arm settings snapshots are back and will silently drift from "
        f"idea_dag_settings.json: {snapshots}. Express the arm in "
        "idea_test_runner._GOT_ARM_PROFILES instead."
    )
