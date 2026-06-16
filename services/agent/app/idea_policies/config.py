"""Typed, grouped views over the raw idea-DAG settings dict.

The on-disk source of truth is ``idea_dag_settings.json`` (loaded as a plain
dict by :func:`agent.app.idea_dag_settings.load_idea_dag_settings`). Reading that
dict directly via ``settings.get(key, default)`` scattered the default value of
every knob across ~140 call sites, so renaming or retyping a key was error-prone
and there was no single place to see what a subsystem is actually tunable by.

These frozen dataclasses fix that: each group declares its fields and their
defaults *once*, and ``from_settings`` is the single place that maps JSON keys to
typed attributes. ``IdeaConfig.from_settings(settings)`` builds every group.

Defaults here mirror the production values shipped in ``idea_dag_settings.json``
(not always the historical per-call-site fallback, which sometimes disagreed with
the JSON — e.g. ``action_max_retries`` was ``0`` at one call site but ``2`` in the
JSON that always overrode it). Because the JSON always supplies these keys, the
runtime value is unchanged; the dataclass default only governs the rare case of a
key being absent. The handful of keys genuinely absent from the JSON
(``semantic_dedup_*``, ``sequential_prune_siblings``, ``final_max_prompt_chars``,
``require_score``) keep their original call-site default.

Content keys — prompts (``*_system_prompt`` / ``*_user_prompt`` /
``*_planning_addendum``) and JSON schemas (``*_json_schema``) — are intentionally
*not* modelled here; they remain dict/document content read directly from settings.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Mapping, Optional


def _coerce(raw: Any, default: Any) -> Any:
    """Coerce ``raw`` to the type encoded by ``default``.

    Mirrors the per-call-site ``bool()/int()/float()`` casts the old code did.
    ``None`` passes through so optional fields keep their absent/null state
    (preserving idioms like ``settings.get(k) if ... is not None else None``).
    """
    if raw is None:
        return None
    if isinstance(default, bool):
        return bool(raw)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw  # str / Optional[str] / anything else passes through


def _build(cls, settings: Mapping[str, Any]):
    """Construct a dataclass view, reading each field from its mapped JSON key.

    The field's own default is the single source of truth for the fallback;
    present values are coerced to the field's declared type.
    """
    defaults = cls()
    key_map: Mapping[str, str] = getattr(cls, "_KEYS", {})
    values: dict[str, Any] = {}
    for f in fields(cls):
        json_key = key_map.get(f.name, f.name)
        current = getattr(defaults, f.name)
        values[f.name] = _coerce(settings.get(json_key, current), current)
    return cls(**values)


@dataclass(frozen=True)
class GoTConfig:
    """Graph-of-Thought optimisation knobs (the ``got_*`` settings keys).

    Several of these keys (``adaptive_policies``, ``dedup_threshold_min/max``,
    ``beam_target_spread``, ``prune_stddev_factor``) are intentionally absent
    from ``idea_dag_settings.json`` and rely solely on these defaults.
    """

    embed_on_create: bool = True
    improve_enabled: bool = True
    improve_score_threshold: float = 0.3
    improve_max_iterations: int = 2
    improve_temperature: float = 0.3
    adaptive_policies: bool = True
    dedup_enabled: bool = True
    dedup_similarity_threshold: float = 0.85
    dedup_threshold_min: float = 0.75
    dedup_threshold_max: float = 0.92
    dedup_max_query: int = 5
    dynamic_beam_enabled: bool = True
    beam_min: int = 2
    beam_max: int = 5
    beam_target_spread: float = 0.4
    beam_score_high: float = 0.7
    beam_score_low: float = 0.3
    prune_enabled: bool = True
    prune_min_nodes_before_prune: int = 6
    prune_stddev_factor: float = 1.0
    prune_score_threshold: float = 0.15
    backtrack_enabled: bool = True
    backtrack_dead_end_threshold: int = 3
    backtrack_low_score_threshold: float = 0.3
    telemetry_routing_enabled: bool = False
    telemetry_routing_score_model: Optional[str] = None
    telemetry_routing_generate_model: Optional[str] = None

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "GoTConfig":
        # Every key is the field name prefixed with ``got_``.
        defaults = cls()
        values: dict[str, Any] = {}
        for f in fields(cls):
            current = getattr(defaults, f.name)
            values[f.name] = _coerce(settings.get(f"got_{f.name}", current), current)
        return cls(**values)


@dataclass(frozen=True)
class GenerationConfig:
    """LLM generation knobs shared across every stage."""

    fallback_model: Optional[str] = None
    reasoning_effort: str = "high"
    text_verbosity: str = "medium"

    _KEYS: ClassVar[dict] = {}

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "GenerationConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class TimeoutConfig:
    """All ``*_timeout_seconds`` knobs."""

    action: int = 20
    search: int = 15
    visit: int = 20
    fetch: int = 20
    chroma: int = 15
    llm: int = 60
    final: int = 180
    expansion: int = 180

    _KEYS: ClassVar[dict] = {
        "action": "action_timeout_seconds",
        "search": "search_timeout_seconds",
        "visit": "visit_timeout_seconds",
        "fetch": "fetch_timeout_seconds",
        "chroma": "chroma_timeout_seconds",
        "llm": "llm_timeout_seconds",
        "final": "final_timeout_seconds",
        "expansion": "expansion_timeout_seconds",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "TimeoutConfig":
        return _build(cls, settings)

    def for_action(self, action_name: str, default: Optional[int] = None) -> Optional[int]:
        """Resolve ``<action_name>_timeout_seconds`` for dynamic per-action lookups."""
        return getattr(self, action_name, default if default is not None else self.action)


@dataclass(frozen=True)
class ExpansionConfig:
    model: Optional[str] = None
    temperature: float = 0.4
    max_tokens: Optional[int] = 8192
    max_context_nodes: int = 5
    max_detail_chars: int = 5000

    _KEYS: ClassVar[dict] = {
        "model": "expansion_model",
        "temperature": "expansion_temperature",
        "max_tokens": "expansion_max_tokens",
        "max_context_nodes": "expansion_max_context_nodes",
        "max_detail_chars": "expansion_max_detail_chars",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "ExpansionConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class EvaluationConfig:
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = 16384
    max_context_nodes: int = 5
    max_detail_chars: int = 5000
    batch_max_candidates: int = 5
    no_action_result_base_score: float = 0.4
    no_action_result_score_cap: float = 0.5
    weight_search: float = 1.0
    weight_visit: float = 1.0
    weight_think: float = 1.0
    weight_save: float = 1.0
    weight_verify: float = 1.0
    weight_default: float = 1.0

    _KEYS: ClassVar[dict] = {
        "model": "evaluation_model",
        "temperature": "evaluation_temperature",
        "max_tokens": "evaluation_max_tokens",
        "max_context_nodes": "evaluation_max_context_nodes",
        "max_detail_chars": "evaluation_max_detail_chars",
        "batch_max_candidates": "evaluation_batch_max_candidates",
        "no_action_result_base_score": "evaluation_no_action_result_base_score",
        "no_action_result_score_cap": "evaluation_no_action_result_score_cap",
        "weight_search": "evaluation_weight_search",
        "weight_visit": "evaluation_weight_visit",
        "weight_think": "evaluation_weight_think",
        "weight_save": "evaluation_weight_save",
        "weight_verify": "evaluation_weight_verify",
        "weight_default": "evaluation_weight_default",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "EvaluationConfig":
        return _build(cls, settings)

    def weight_for(self, action: str) -> float:
        return getattr(self, f"weight_{action}", self.weight_default)


@dataclass(frozen=True)
class FinalConfig:
    model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: Optional[int] = 120000
    chroma_results: int = 10
    max_prompt_chars: int = 200000  # absent from JSON; original call-site default
    allow_partial_success: bool = True

    _KEYS: ClassVar[dict] = {
        "model": "final_model",
        "temperature": "final_temperature",
        "max_tokens": "final_max_tokens",
        "chroma_results": "final_chroma_results",
        "max_prompt_chars": "final_max_prompt_chars",
        "allow_partial_success": "final_allow_partial_success",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "FinalConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class MergeConfig:
    model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: Optional[int] = 100000

    _KEYS: ClassVar[dict] = {
        "model": "merge_model",
        "temperature": "merge_temperature",
        "max_tokens": "merge_max_tokens",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "MergeConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class VerifyConfig:
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = 1024

    _KEYS: ClassVar[dict] = {
        "model": "verify_model",
        "temperature": "verify_temperature",
        "max_tokens": "verify_max_tokens",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "VerifyConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class ActionConfig:
    max_retries: int = 2
    retry_backoff_steps: int = 1
    max_observation_chars: int = 100000
    default_search_count: int = 10
    max_links_per_visit: int = 20
    visit_max_sites_per_action: int = 20
    visit_link_query_top_k: int = 15
    visit_page_concurrency: int = 5
    visit_link_selection_model: Optional[str] = None
    visit_empty_content_retryable: bool = True

    _KEYS: ClassVar[dict] = {
        "max_retries": "action_max_retries",
        "retry_backoff_steps": "action_retry_backoff_steps",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "ActionConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class MemoryConfig:
    document_chunk_threshold: int = 200000
    document_chunk_size: int = 4000
    document_chunk_overlap: int = 400
    expansion_chroma_internal: int = 5
    expansion_chroma_observations: int = 5
    leaf_chroma_results: int = 3
    default_semantic_results: int = 3
    max_available_links_for_expansion: int = 50
    grep_context_window: int = 80

    _KEYS: ClassVar[dict] = {}

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "MemoryConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class EngineConfig:
    """Execution / graph-shape / parallelism / logging knobs read by the engine."""

    max_branching: int = 5
    max_total_nodes: int = 500
    grounding_max_replans: int = 2
    best_first_global: bool = True
    allow_execute_all_children: bool = True
    min_score_threshold: float = 0.0
    allow_unscored_selection: bool = True
    auto_parallel_siblings: bool = True
    parallel_action_limit: int = 4
    sequential_sibling_recovery_enabled: bool = True
    sequential_prune_siblings: bool = False  # absent from JSON
    semantic_dedup_visits_enabled: bool = True  # absent from JSON
    semantic_dedup_require_hook_source: bool = True  # absent from JSON
    got_prune_interval_steps: int = 5
    log_dag_ascii: bool = True
    log_dag_step_interval: int = 1

    _KEYS: ClassVar[dict] = {}

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "EngineConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class PolicyConfig:
    """Single-knob policies (selection / decomposition / recursive-merge)."""

    require_score: bool = True  # absent from JSON
    decomposition_threshold: float = 0.5
    enable_recursive_merge: bool = True

    _KEYS: ClassVar[dict] = {}

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "PolicyConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class IdeaConfig:
    """Aggregate of every typed group; built once from the raw settings dict."""

    got: GoTConfig
    generation: GenerationConfig
    timeouts: TimeoutConfig
    expansion: ExpansionConfig
    evaluation: EvaluationConfig
    final: FinalConfig
    merge: MergeConfig
    verify: VerifyConfig
    action: ActionConfig
    memory: MemoryConfig
    engine: EngineConfig
    policy: PolicyConfig

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "IdeaConfig":
        return cls(
            got=GoTConfig.from_settings(settings),
            generation=GenerationConfig.from_settings(settings),
            timeouts=TimeoutConfig.from_settings(settings),
            expansion=ExpansionConfig.from_settings(settings),
            evaluation=EvaluationConfig.from_settings(settings),
            final=FinalConfig.from_settings(settings),
            merge=MergeConfig.from_settings(settings),
            verify=VerifyConfig.from_settings(settings),
            action=ActionConfig.from_settings(settings),
            memory=MemoryConfig.from_settings(settings),
            engine=EngineConfig.from_settings(settings),
            policy=PolicyConfig.from_settings(settings),
        )


def validate_settings(settings: Mapping[str, Any]) -> "IdeaConfig":
    """Build every typed view, raising if any known knob has a non-coercible type.

    Call this at startup to fail loudly on a malformed settings dict instead of
    surfacing a ``ValueError`` deep inside a run. (Component constructors already
    build :class:`IdeaConfig`, so type errors also surface at construction; this
    is the explicit, eager entry point.) Returns the built config for reuse.
    """
    try:
        return IdeaConfig.from_settings(settings)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid idea-DAG settings: {exc}") from exc
