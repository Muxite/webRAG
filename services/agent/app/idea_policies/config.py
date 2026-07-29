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
from typing import Any, ClassVar, Mapping, Optional, Tuple


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
    reexpand_enabled: bool = False
    reexpand_max_iterations: int = 1
    reexpand_temperature: float = 0.2
    step_confidence_judge_enabled: bool = False
    step_confidence_judge_temperature: float = 0.0
    step_confidence_judge_sample_every: int = 1
    step_confidence_judge_model: Optional[str] = None
    step_confidence_reexpand_enabled: bool = False
    step_confidence_reexpand_threshold: float = 0.5
    # F33 — re-base the re-expansion trigger on CONTRACT SATISFACTION instead of the
    # anti-calibrated step-confidence judge (opt-in, default OFF -> byte-identical). When on,
    # a completed retrieval leaf whose deterministic contract check reports a missing payload /
    # datum / subject re-expands (no judge LLM call needed), and a leaf whose contract is
    # demonstrably SATISFIED is protected from the judge's low-score trigger. Where the check
    # has no verdict (``applicable=False``) the confidence trigger still applies as before.
    contract_reexpand_enabled: bool = False
    reexpand_corrective_context_enabled: bool = False
    candidate_coverage_enabled: bool = False
    candidate_coverage_budget_extension: int = 10
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
    backtrack_enabled: bool = False
    backtrack_dead_end_threshold: int = 5
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
    expect_contract_enabled: bool = False

    _KEYS: ClassVar[dict] = {
        "model": "expansion_model",
        "temperature": "expansion_temperature",
        "max_tokens": "expansion_max_tokens",
        "max_context_nodes": "expansion_max_context_nodes",
        "max_detail_chars": "expansion_max_detail_chars",
        "expect_contract_enabled": "expansion_expect_contract_enabled",
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

    def weight_for(self, action: Optional[str]) -> float:
        if not action:
            return self.weight_default
        return getattr(self, f"weight_{str(action).lower()}", self.weight_default)


@dataclass(frozen=True)
class FinalConfig:
    model: Optional[str] = None
    temperature: float = 0.3
    # Capped at the provider RESERVATION ceiling (see
    # ``idea_dag_settings._MAX_TOKENS_RESERVATION_CAP``): the old 120000 reserved ~30x the largest
    # deliverable ever observed and 402'd once a daily credit cap drained.
    max_tokens: Optional[int] = 32768
    chroma_results: int = 10
    max_prompt_chars: int = 200000  # absent from JSON; original call-site default
    allow_partial_success: bool = True
    # F31 — hard grounding gate before finalize (opt-in, default OFF -> byte-identical). When on
    # and the run opened ZERO pages on a grounded-research mandate, the answer is not presented as
    # a researched result: it is banner-flagged as ungrounded, its unverifiable URLs are stripped,
    # and success/goal_achieved are forced False rather than laundering parametric memory.
    require_grounding: bool = False
    # C1b — approximator-stripped k-sample vote for the terminal answer (opt-in). When
    # ``native_vote_k_enabled`` and ``native_vote_k`` >= 2, the finalize answer is extracted k
    # times (anchor temp-0 + diverse temps), normalized via the approximator-stripped vote key,
    # and the majority wins (tie-break toward the anchor). k=1 (or the flag off) == exactly one
    # extraction, the current behavior -> byte-identical default.
    native_vote_k_enabled: bool = False
    native_vote_k: int = 1
    # Post-synthesis reconcile chain (opt-in, default OFF -> byte-identical). Each pass runs ONLY
    # for answer-shaped tasks (see ``final_recompute_shapes``) and fails open (keeps the prior
    # draft on empty/error/timeout). ``final_recompute_enabled``: re-list the exact source values
    # (verbatim quote + URL) and re-derive the answer, correcting an arithmetic/extraction slip.
    # ``final_verify_enabled``: demand a verbatim passage that supports the draft, else replace it
    # with what the evidence actually says. ``final_variations_enabled`` (with ``final_variations_k``
    # framings): the DECORRELATED alternative to k-vote — answer K differently-framed versions of the
    # question independently, then reconcile — surfacing the correct value where one framing misreads.
    # Order when several are on: variations->collate, then recompute, then verify.
    final_recompute_enabled: bool = False
    final_verify_enabled: bool = False
    final_variations_enabled: bool = False
    final_variations_k: int = 3
    # The answer-shape labels (see ``shape_classifier.classify_answer_shape``) that gate the chain.
    final_recompute_shapes: Tuple[str, ...] = (
        "computation", "count", "argmax", "disambiguation", "single_value",
    )

    _KEYS: ClassVar[dict] = {
        "model": "final_model",
        "temperature": "final_temperature",
        "max_tokens": "final_max_tokens",
        "chroma_results": "final_chroma_results",
        "max_prompt_chars": "final_max_prompt_chars",
        "allow_partial_success": "final_allow_partial_success",
        "require_grounding": "final_require_grounding",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "FinalConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class MergeConfig:
    model: Optional[str] = None
    temperature: float = 0.3
    # Same reservation cap as ``FinalConfig.max_tokens`` (merge output is bounded by the same
    # deliverable); see ``idea_dag_settings._MAX_TOKENS_RESERVATION_CAP``.
    max_tokens: Optional[int] = 32768

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
class PlanLibraryConfig:
    """Retrieval-augmented planning — the ``plan_library_*`` keys.

    A new subsystem rather than a GoT-optimisation knob, so it gets its own group and does
    NOT use ``GoTConfig``'s ``got_`` auto-prefix. ``enabled`` is the master switch;
    ``auto_enabled`` governs the automatic pre-expansion short-circuit (a confident template
    match replaces the LLM's invented decomposition) and ``action_enabled`` the on-demand
    ``plan_library_search`` leaf action (the model asks for a strategy itself, and an adopted
    one grows children through ``_maybe_plan_library_reexpand``). The two sub-flags are
    independent — auto-only, on-demand-only or both — mirroring the
    ``got_step_confidence_judge_enabled``/``got_step_confidence_reexpand_enabled``
    relationship. All default OFF -> byte-identical.

    Deliberately carries NO similarity threshold: the decision constants live in
    ``plan_library/retrieval.py`` (``AUTO_APPLY_THRESHOLD``/``SUGGEST_THRESHOLD``/
    ``MIN_MARGIN``), calibrated against a labelled eval set, and the engine trusts
    ``RetrievalResult.decision`` as that calibrated verdict. A second, uncalibrated gate here
    would either be redundant or silently reject matches the calibration proved correct.
    """

    enabled: bool = False
    auto_enabled: bool = False
    action_enabled: bool = False

    _KEYS: ClassVar[dict] = {
        "enabled": "plan_library_enabled",
        "auto_enabled": "plan_library_auto_enabled",
        "action_enabled": "plan_library_action_enabled",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "PlanLibraryConfig":
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
    # A3b — reasoning-effort/token discipline for native leaf micro-prompts (opt-in). When on,
    # a reasoning-model executor's perception/selection micro-prompt uses reasoning_effort=minimal
    # and its token budget is floored so hidden reasoning can't starve the completion (the
    # content=None bug fixed on the compiled path). Default OFF -> byte-identical.
    native_reasoning_effort_discipline_enabled: bool = False
    native_reasoning_min_tokens_floor: int = 2048
    # A5 — price-tier parameter tiering for native executor micro-prompts (opt-in). When on, a
    # micro-prompt's token budget scales by the executor model's price tier (cheap stays tight,
    # mid/premium get headroom). Default OFF -> byte-identical.
    price_tier_param_tiering_enabled: bool = False
    # C1a — tool-failure recovery (opt-in). ``connector_retry_on_failure_enabled``: when a leaf
    # action returns a TOOL failure (empty/timeout/HTTP-error fetch, no search results), retry the
    # SAME action in place with bounded backoff before deciding the node's fate — so a TRANSIENT
    # failure recovers at the source instead of the re-expansion loop spawning a subtree that
    # repeats the failing fetch. ``tool_failure_recovery_enabled``: route the low-confidence
    # re-expansion trigger AWAY from re-expanding a leaf whose low score was caused by a tool
    # failure (a fresh subtree would just repeat it). Both default OFF -> byte-identical.
    connector_retry_on_failure_enabled: bool = False
    connector_retry_max_attempts: int = 2
    connector_retry_backoff_seconds: float = 0.5
    tool_failure_recovery_enabled: bool = False

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
    plan_library: PlanLibraryConfig
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
            plan_library=PlanLibraryConfig.from_settings(settings),
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
