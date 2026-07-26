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

from agent.app.idea_dag_settings import load_idea_dag_settings
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
    # FinalConfig
    "final_max_prompt_chars",
    # EngineConfig
    "sequential_prune_siblings",
    "semantic_dedup_visits_enabled",
    "semantic_dedup_require_hook_source",
    # PolicyConfig
    "require_score",
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
    (falling back to the bare field name), while ``GoTConfig.from_settings`` --
    the only group without a ``_KEYS`` ClassVar -- hard-codes the ``got_`` prefix.
    """
    key_map = getattr(cls, "_KEYS", None)
    if key_map is None:
        return f"got_{field_name}"
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
