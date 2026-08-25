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
the JSON (e.g. ``action_max_retries`` was ``0`` at one call site but ``2``) in the
JSON that always overrode it). Because the JSON always supplies these keys, the
runtime value is unchanged; the dataclass default only governs the rare case of a
key being absent. The handful of keys genuinely absent from the JSON
(``semantic_dedup_*``, ``sequential_prune_siblings``, ``final_max_prompt_chars``,
``require_score``) keep their original call-site default.

Content keys (prompts: ``*_system_prompt`` / ``*_user_prompt`` /
``*_planning_addendum``, and JSON schemas: ``*_json_schema``) are intentionally
*not* modelled here; they remain dict/document content read directly from settings.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
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
    ``beam_target_spread``, ``prune_stddev_factor``, ``backtrack_dead_end_path_fraction``)
    are intentionally absent
    from ``idea_dag_settings.json`` and rely solely on these defaults.
    """

    embed_on_create: bool = True
    reexpand_enabled: bool = False
    reexpand_max_iterations: int = 1
    # Capability-tiered re-expansion budget (opt-in, layered on top of reexpand_max_iterations
    # above; this flag alone is a no-op). The _effective_reexpand_max_iterations in idea_engine.py
    # is the single place all THREE re-expansion budget check sites read through. This is required
    # because the exact knob was once silently inert at one call site for any value > 1 (see
    # comment there). When on, effective budget is picked via model_tiers.capability_tier: weak
    # models get more bounded re-expansion attempts (native-engine analog of badmodel-lab's
    # per-page extraction retry lever, see capability-continuum plan), strong models taper
    # toward the current default. Placeholders pending live calibration.
    reexpand_max_iterations_tiered_enabled: bool = False
    reexpand_max_iterations_weak: int = 2
    reexpand_max_iterations_standard: int = 1
    reexpand_max_iterations_strong: int = 1
    reexpand_temperature: float = 0.2
    # Emit ``reason`` BEFORE the ``needs_followup`` boolean it justifies, in
    # ``check_needs_followup``. Opt-in, default OFF; JSON key is derived as
    # ``got_reexpand_followup_reason_first_enabled``. Doubly dormant by default -- the
    # whole path is already gated by ``reexpand_enabled`` above.
    #
    # promptbench v2 (2026-08-19): the largest measured effect in the run, pooled
    # A2-A1 = +0.196, CI [+0.080, +0.312], permutation p = 0.0064. End-to-end transfer
    # is unmeasured.
    reexpand_followup_reason_first_enabled: bool = False
    step_confidence_judge_enabled: bool = False
    step_confidence_judge_temperature: float = 0.0
    step_confidence_judge_sample_every: int = 1
    step_confidence_judge_model: Optional[str] = None
    step_confidence_reexpand_enabled: bool = False
    step_confidence_reexpand_threshold: float = 0.5
    # F33: re-base re-expansion trigger on CONTRACT SATISFACTION instead of the
    # anti-calibrated step-confidence judge (opt-in, default OFF for byte-identity).
    # A completed retrieval leaf whose contract check reports missing payload/datum/subject
    # re-expands without an LLM judge call. A leaf whose contract is demonstrably SATISFIED
    # is protected from the judge's low-score trigger. When the check has no verdict
    # (applicable=False) the confidence trigger still applies as before.
    contract_reexpand_enabled: bool = False
    # F35: narrow F33's VETO half. A satisfied contract silences the judge, but most leaf
    # contracts are derived from the leaf's own goal and carry no measurable datum, so
    # "satisfied" degrades to "the page I opened mentions the words in my own goal" -- true
    # for every hop of a chain that is nonetheless only one hop in. Corpus replay (168 judged
    # runs, 251 low-confidence visit leaves): F33 vetoes 171 of them and 134 of those vetoes
    # (78%) rest on a subject-only contract. With this on, only a contract that verified a
    # DATUM may veto; a subject-only one leaves the decision to the judge, as before F33.
    # Opt-in, default OFF for byte-identity.
    contract_veto_requires_datum_enabled: bool = False
    # Sequential A->B fallback triggers. Two flags, not one, so each trigger's contribution
    # is independently measurable; both are no-ops unless the expansion actually authored an
    # alternative (``expansion_alternative_branch_enabled``), since without a
    # ``has_alternative_node`` back-pointer there is nothing to promote.
    #
    #   * ``_on_fail_``: the primary ended FAILED. Unambiguous, needs no other signal.
    #   * ``_on_unverified_``: the primary ended DONE and its step contract is satisfied but
    #     verified no measurable DATUM — F35's distinction between "opened a page matching
    #     the words of my own goal" and "actually answered the ask". That signal only exists
    #     when ``contract_reexpand_enabled`` and ``contract_veto_requires_datum_enabled`` are
    #     both on, so this trigger requires them rather than silently reading a verdict whose
    #     datum half nothing else in the run trusts.
    alternative_branch_promote_on_fail_enabled: bool = False
    alternative_branch_promote_on_unverified_enabled: bool = False
    reexpand_corrective_context_enabled: bool = False
    # F6 (narrow MVP): re-plan a parent whose whole expansion collapsed to the single guessed
    # candidate `_create_fallback_candidate` emits. Only fires when the parent's ENTIRE child
    # set is that one `FALLBACK_EXPANSION`-tagged leaf, so a parent can be repaired at most
    # once. Opt-in, default OFF for byte-identity.
    reexpand_fallback_nodes_enabled: bool = False
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
    # Feed the dynamic beam's score pool (the adaptive p25/p75 spread, and the legacy
    # average branch that shares it) the judge's PRE-CAP opinion (`raw_score`,
    # recorded by the evaluation policies since 9b710b91) instead of the recorded `node.score`.
    # Every candidate is scored while still pending, so `evaluate_batch` clips it at
    # `evaluation_no_action_result_score_cap` -- 45.8% of pooled nodes in the recorded corpus
    # sit on a value the engine wrote rather than the judge, which shrinks the measured spread
    # by construction and reads as "converged". `scripts/analyze_beam_spread_contamination.py`
    # replays 506 post-9b710b91 runs: mean spread 0.209 -> 0.410 and the beam changes in 50.6%
    # of end-of-run pools, always WIDER. Nodes whose `raw_score` is None (the base-score
    # shortcut, which never calls the judge) keep their `node.score`. Only this spread reads
    # the field; `node.score` and every other consumer are untouched. Opt-in, default OFF for
    # byte-identity -- the widening is unvalidated live and costs fan-out.
    beam_spread_uses_raw_score_enabled: bool = False
    beam_score_high: float = 0.7
    beam_score_low: float = 0.3
    prune_enabled: bool = True
    prune_min_nodes_before_prune: int = 6
    prune_stddev_factor: float = 1.0
    prune_score_threshold: float = 0.15
    backtrack_enabled: bool = False
    backtrack_dead_end_threshold: int = 5
    backtrack_low_score_threshold: float = 0.3
    # T1-6: `backtrack_dead_end_threshold` is an ABSOLUTE node count compared against the
    # leading low-score run of `path_to_root`, which is bounded above by the node's depth.
    # `scripts/analyze_prune_backtrack_deadzone.py` measures that depth over 11121 recorded
    # non-root nodes: the maximum anywhere in the corpus is 3, so 5 is unreachable by
    # construction on the graphs this engine actually builds. With this flag on, the limit is
    # derived from the path the walk just took -- `max(2, ceil(fraction * scored path length))`
    # -- so it means "the whole scored path from here to the root is low" on a graph of any
    # depth, instead of a magic number tuned for depths that never occur. The floor of 2 keeps
    # a single low score from ever triggering a backtrack. `backtrack_enabled` is itself default
    # OFF, so this is doubly gated; default OFF for byte-identity pending a live A/B.
    backtrack_dead_end_relative_enabled: bool = False
    #: Absent from `idea_dag_settings.json` on purpose, like `prune_stddev_factor` above.
    backtrack_dead_end_path_fraction: float = 0.75
    # A6: the SYMMETRIC counterpart of the step-confidence trigger above. A1 only ever adds
    # compute (a distrusted step re-expands); this stops the loop and finalizes when the
    # accumulated confidence prefix clears a CALIBRATED bar, so an easy mandate does not pay
    # for steps it does not need. The bar is not hand-picked: it is derived from held-out
    # (confidence-sequence, eventual-label) pairs with certified false-stop rate and shipped
    # in confidence_early_exit_calibration.json (see idea_policies/confidence_early_exit.py).
    # Opt-in, default OFF for byte-identity. Margin is extra conservatism on top of the
    # calibrated threshold; min_judged_steps is a floor below which no rule fires.
    confidence_early_exit_enabled: bool = False
    confidence_early_exit_margin: float = 0.05
    confidence_early_exit_min_judged_steps: int = 2
    telemetry_routing_enabled: bool = False
    telemetry_routing_score_model: Optional[str] = None
    telemetry_routing_generate_model: Optional[str] = None

    #: Fields whose JSON key is NOT the got_-prefixed field name. The A6 early-exit knobs
    #: are GoT control-loop decisions (siblings of backtrack_enabled/step_confidence_*) but
    #: named with the native_ prefix that the native-engine A-series uses (native_vote_k_enabled,
    #: native_reasoning_effort_discipline_enabled). Deliberately NOT called _KEYS: that ClassVar
    #: means "override the bare field name" in every _build group. Reusing the name here with
    #: a got_-prefixed fallback would give it two meanings.
    _NATIVE_KEYS: ClassVar[dict] = {
        "confidence_early_exit_enabled": "native_confidence_early_exit_enabled",
        "confidence_early_exit_margin": "native_confidence_early_exit_margin",
        "confidence_early_exit_min_judged_steps": "native_confidence_early_exit_min_judged_steps",
    }

    @classmethod
    def json_key(cls, field_name: str) -> str:
        """The JSON key a field reads: the ``got_`` prefix, or a ``_NATIVE_KEYS`` override."""
        return cls._NATIVE_KEYS.get(field_name, f"got_{field_name}")

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "GoTConfig":
        # Every key is the field name prefixed with ``got_``, except ``_NATIVE_KEYS``.
        defaults = cls()
        values: dict[str, Any] = {}
        for f in fields(cls):
            current = getattr(defaults, f.name)
            values[f.name] = _coerce(settings.get(cls.json_key(f.name), current), current)
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
    # Merge previously had no dedicated field here, so idea_engine.py's _action_timeout_for
    # returned None and fell back to generic action timeout (20s). A merge call synthesizes from
    # all children's raw page content concatenated in one LLM call (confirmed: 150-210KB for
    # 4-leaf tasks). This exceeds 20s for local models. Found via barrage: every qwen2.5:14b
    # merge node timed out at 20s, stranding otherwise-grounded runs to finalize fallback.
    # 180 matches final/expansion's generous default for single-call-over-large-context.
    merge: int = 180

    _KEYS: ClassVar[dict] = {
        "action": "action_timeout_seconds",
        "search": "search_timeout_seconds",
        "visit": "visit_timeout_seconds",
        "fetch": "fetch_timeout_seconds",
        "chroma": "chroma_timeout_seconds",
        "llm": "llm_timeout_seconds",
        "final": "final_timeout_seconds",
        "expansion": "expansion_timeout_seconds",
        "merge": "merge_timeout_seconds",
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
    # Per-ancestor page content the planner may see. Was a bare literal `1000` with no knob,
    # against `sequential_react`'s 6000 chars per page for EVERY page in its linear history --
    # a 6x per-page and much larger aggregate disadvantage in evidence visible at decision
    # time. The DAG's problem on the baseline was starvation, not context overload (it visited
    # 1.3 pages to the flat arms' 3.8), so this exists to be raised and measured, not trimmed.
    # Default preserves the historical value exactly.
    ancestor_content_chars: int = 1000
    expect_contract_enabled: bool = False
    # Prompt hygiene for weak models: label the context blob as read-only INPUT and restate the
    # {candidates: [...]} output shape right after it. Live-proven 2026-08-06 (eliminated 5/8
    # raw-completion echo failures on qwen2.5:0.5b/llama3.2:1b, 0/16 after) and defaulted ON
    # 2026-08-14: the `baseline` benchmark arm pins it back to `False` explicitly so the
    # "adaptive OFF" ladder rung stays byte-identical regardless of this default.
    # ``echo_retry_enabled`` is a separate lever on purpose so the prompt fix and the retry
    # safety net can be ablated independently. It stays OFF (no measurable benefit shown yet).
    # See expansion.py's ``_INPUT_FRAMING_HEADER`` block for the telemetry that motivated them.
    input_output_framing_enabled: bool = True
    echo_retry_enabled: bool = False
    # Ask the author, once, at ordinary expansion time, for two OPTIONAL structural hints per
    # candidate: ``alternative_of`` (this candidate is a fallback for that one) and
    # ``race_group`` (these candidates are different routes to the SAME fact). Swaps in the
    # ``EXPANSION_JSON_SCHEMA_WITH_BRANCHING`` schema variant and appends a short prompt
    # addendum; ``idea_policies/alternative_branch.py`` resolves the hints into real node
    # relationships after ``graph.expand()``. Opt-in, default OFF -> byte-identical prompt.
    alternative_branch_enabled: bool = False
    # Recover the SAME race relationship as ``alternative_branch_enabled``'s ``race_group`` tag
    # without asking the model for a tag at all: ``alternative_branch.infer_race_groups`` reads
    # it off plan shape (near-duplicate ``expect`` contracts, or title overlap, AND disjoint
    # approaches) after ``graph.expand()``. Checked INDEPENDENTLY of the flag above, because
    # the point is to work with the branching schema variant switched off — the live emission
    # probe found the authored tag is simply never emitted below the 14b tier. Opt-in, default
    # OFF; on its own it only populates the ``race_groups_inferred`` registry (instrumentation),
    # since merge-time consumption is gated separately by
    # ``merge_race_winner_selection_includes_inferred_groups_enabled``.
    race_group_structural_inference_enabled: bool = False

    _KEYS: ClassVar[dict] = {
        "model": "expansion_model",
        "temperature": "expansion_temperature",
        "max_tokens": "expansion_max_tokens",
        "max_context_nodes": "expansion_max_context_nodes",
        "max_detail_chars": "expansion_max_detail_chars",
        "ancestor_content_chars": "expansion_ancestor_content_chars",
        "expect_contract_enabled": "expansion_expect_contract_enabled",
        "input_output_framing_enabled": "expansion_input_output_framing_enabled",
        "echo_retry_enabled": "expansion_echo_retry_enabled",
        "alternative_branch_enabled": "expansion_alternative_branch_enabled",
        "race_group_structural_inference_enabled": (
            "expansion_race_group_structural_inference_enabled"
        ),
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
    # Capped at provider RESERVATION ceiling (see idea_dag_settings._MAX_TOKENS_RESERVATION_CAP).
    # Old 120000 reserved ~30x the largest deliverable ever observed and triggered HTTP 402 errors
    # when daily credit cap drained.
    max_tokens: Optional[int] = 32768
    chroma_results: int = 10
    max_prompt_chars: int = 200000  # absent from JSON; original call-site default
    allow_partial_success: bool = True
    # F31: hard grounding gate before finalize (opt-in, default OFF for byte-identity). When on
    # and the run opened ZERO pages on a grounded-research mandate, the answer is not presented
    # as researched. It is banner-flagged as ungrounded, unverifiable URLs are stripped, and
    # success/goal_achieved are forced False rather than laundering parametric memory.
    require_grounding: bool = False
    # F37: page-identity relevance signal layered onto the grounding gate above (opt-in,
    # default OFF). `evaluate_grounding` today is pure set arithmetic over visited URLs — any
    # 2 pages (or 1 followed link) satisfy it, with zero title/URL/content relevance check, so
    # two completely off-topic visits trivially "ground" the answer. When on, a visit only
    # counts toward the requirement if it ALSO passes `waypoint.page_identity_ok` (the same
    # h1/title/url-only guard build_waypoint already uses to reject a wrong-page fetch) against
    # the subject tokens of the leaf that performed it. Changes pass/fail semantics of a gate
    # that is on in every arm today (`final_require_grounding: True` everywhere) — needs a live
    # A/B before flipping the default, since some currently-passing runs could newly fail if
    # subject-token extraction is noisy on certain task phrasings.
    require_grounding_page_identity: bool = False
    # C1b: approximator-stripped k-sample vote for terminal answer (opt-in). When
    # native_vote_k_enabled and native_vote_k >= 2, finalize answer is extracted k times
    # (anchor temp-0 + diverse temps), normalized via approximator-stripped vote key, and
    # majority wins (tie-break toward anchor). k=1 or flag off equals one extraction, current
    # behavior, for byte-identity default.
    native_vote_k_enabled: bool = False
    native_vote_k: int = 1
    # Capability-tiered vote-k (opt-in, layered on top of native_vote_k_enabled; this flag
    # alone without the master switch is a no-op). When both are on, native_vote_k is overridden
    # by a band picked via model_tiers.capability_tier: weak models get more redundant finalize
    # votes, strong models taper toward k=1 (fully off downstream). Pattern ported (not numbers)
    # from compiled-scaffold's _votes_for_model auto-tapering (cheap=5/unknown=3/mid=3/premium=2)
    # but not copied verbatim. Those were calibrated for cheap per-page thin-extraction vote, not
    # full finalize-prompt rerun. That harness's 4-bucket unknown≠cheap split has no equivalent
    # here (capability_tier deliberately collapses unknown into weak). These are placeholders
    # pending live calibration (see capability-continuum plan); do not treat as tuned values.
    native_vote_k_tiered_enabled: bool = False
    native_vote_k_weak: int = 3
    native_vote_k_standard: int = 2
    native_vote_k_strong: int = 1
    # Size-band refinement within the weak band for LOCAL (unpriced) models only (opt-in, layered
    # on top of native_vote_k_tiered_enabled; this flag alone is a no-op). When on, a model whose
    # tag encodes parameter count (model_tiers.local_model_size_band) uses the matching band
    # instead of flat native_vote_k_weak. Unparseable tags (phi3:mini, tinyllama) or any priced
    # model keeps flat tiered value, so this can only refine, never reinterpret. Rationale for
    # the taper (badmodel-lab reachable-tier): <2B scores 0.25-0.54, failures often malformed
    # extraction that one more sample rescues; >=12B scores 0.97 (paid-API ceiling), where
    # blanket finalize redundancy buys nothing and costs most wall-clock (big local models are
    # slowest to re-run, small ones cheapest). Big local models still need mitigation stack for
    # specific failure modes (targeted levers, not this blanket one). Placeholders pending live
    # calibration, like tier bands above.
    native_vote_k_size_band_enabled: bool = False
    native_vote_k_local_tiny: int = 4
    native_vote_k_local_small: int = 3
    native_vote_k_local_medium: int = 2
    native_vote_k_local_large: int = 1
    # Post-synthesis reconcile chain (opt-in, default OFF for byte-identity). Each pass runs
    # only for answer-shaped tasks (see final_recompute_shapes) and fails open (keeps prior draft
    # on empty/error/timeout). final_recompute_enabled: re-list exact source values (verbatim
    # quote + URL) and re-derive answer, correcting arithmetic/extraction slip. final_verify_enabled:
    # demand verbatim passage that supports draft, else replace with what evidence actually says.
    # final_variations_enabled (with final_variations_k framings): decorrelated alternative to
    # k-vote. Answer K differently-framed versions of question independently, then reconcile to
    # surface correct value where one framing misreads. Order when several are on: variations,
    # then recompute, then verify.
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
        "require_grounding_page_identity": "final_require_grounding_page_identity",
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
    # Emit ``goal_evaluation`` BEFORE the ``goal_achieved`` boolean it justifies, in both
    # of merge's ordering sources (the prompt template and the merge schema hint). Opt-in,
    # default OFF: the assembled merge system message is byte-identical when false.
    #
    # promptbench v2 (2026-08-19): the shipped boolean-first ordering is DEGENERATE on 5/5
    # models -- each answered ACHIEVED on 100% of a balanced set, scoring chance while
    # judging nothing. No paired contrast is even computable against a constant. Reason-first
    # scored up to 0.929. Authorized by the pre-registered gate as flagged and default OFF;
    # end-to-end transfer is unmeasured.
    goal_evaluation_first_enabled: bool = False
    # Resolve an authored race group (2+ siblings that are different routes to the SAME fact)
    # down to ONE winner before aggregating, instead of synthesizing over every route's
    # result. Purely mechanical: status + contract comparisons, never an LLM judgment call.
    # Opt-in, default OFF -> ``merge()`` aggregates everything exactly as today.
    race_winner_selection_enabled: bool = False
    # Let the flag above also resolve STRUCTURALLY INFERRED groups (``race_groups_inferred``,
    # from ``expansion_race_group_structural_inference_enabled``), not just authored ones. A
    # separate flag because the two registries carry different evidence: an authored tag is a
    # choice the model made, an inferred group is a heuristic that has never been live
    # validated — and winner selection actively DISCARDS the loser's findings, so a false
    # positive here silently drops a legitimate independent result. Opt-in, default OFF ->
    # inferred groups stay pure instrumentation no matter what the expansion flag is set to.
    # Reaches TIER 1 groups only: tier 2 measured 50% precision live, so it is unconsumable
    # regardless of this flag (see ``SimpleMergePolicy._race_registry``).
    race_winner_selection_includes_inferred_groups_enabled: bool = False
    # Hard ceiling on the serialized ``merged_results`` blob spliced into the merge user
    # prompt. Truncation here is blind (a mid-JSON chop), so a child past the cap is simply
    # invisible to synthesis -- the cap is a last resort behind ``_compact_merged_results``.
    max_json_chars: int = 100000
    # Only mint a merge node when at least TWO children actually produced something to
    # synthesize, instead of gating on structural child count alone. The 2026-08-21
    # diagnostic found 4 of 7 real merge calls "combining" one content-bearing child with a
    # Serper-403 search the connector had recorded as ``success: true, content: ""`` -- an
    # LLM call that can only restate its single source, and whose ``goal_achieved`` verdict
    # then terminates the branch. Default ON with a kill switch: declining routes into the
    # engine's existing no-merge path (parent marked DONE, children's content still reaches
    # finalization through the graph-wide collectors), and the check is deliberately
    # generous -- any non-echoed, non-empty result field counts (see
    # ``SimpleMergePolicy._payload_is_substantive``), so genuine multi-source merges are
    # untouched.
    require_substantive_children_enabled: bool = True
    # Let a parent whose merge was SKIPPED mint a second merge node once genuinely new
    # evidence has landed. ``should_create_merge_node``'s dedup check returns False for ANY
    # existing merge child, skipped or not, so the first skip is an irreversible lockout: a
    # branch the synthesis judged incomplete can never be re-judged, however much its
    # remaining siblings go on to find (ENGINE_DESIGN_REVIEW D4). Retry is bounded by the
    # substantive-child count stamped when the skipped merge was created -- the count must
    # have GROWN -- so re-running the same evidence is still impossible. Opt-in, default OFF:
    # unlike the pure bugfixes it composes with, this creates new nodes and new LLM calls on
    # branches that were previously terminal.
    retry_after_skip_enabled: bool = False
    # Refuse a ``goal_achieved: true`` whose only goal-relevant evidence is unvisited
    # search-result snippets (``goal_evidence_provenance`` == ``snippet``). The 2026-08-21
    # reason-first A/B caught qwen2.5:14b electing the right entity from snippets alone,
    # never fetching the keystone datum, and claiming victory in BOTH prompt-ordering arms --
    # so this is a gap in what counts as achieved, not a prompt problem. Opt-in, default OFF:
    # unlike the consistency guard (which resolves a contradiction the completion states
    # itself), this overrules a self-consistent verdict on external grounds, and a goal whose
    # answer genuinely IS a search snippet -- or one phrased so the fetched page clears the
    # overlap bar only by paraphrase -- would be wrongly held incomplete. Detection is
    # unconditional either way: the ``goal_achieved_snippet_only`` marker and its warning are
    # written whatever this flag says, so the failure mode is measurable before it is acted on.
    require_visited_evidence_enabled: bool = False
    # Refuse a ``goal_achieved: true`` whose asserted MEASUREMENTS appear in nothing the run
    # fetched (``grounding.answer_numeric_provenance``). Narrower than the flag above and
    # paraphrase-proof: numbers do not re-word, so "1,310 metres" / "1310 m" / "1 310 metres"
    # are one fact under normalization. Targets a live-observed completion that narrated "all
    # three routes confirm 575 meters" -- a figure in no fetched page anywhere -- as achieved,
    # with zero cited URLs. It fires only when EVERY measurement in the completion is absent
    # from every visited page's raw text (see ``NumericProvenanceResult.unsupported``), and a
    # completion asserting no measurement at all is a no-op.
    #
    # Opt-in, default OFF for the same reason as the flag above, on a residual risk this check
    # narrows but cannot eliminate: a page carrying the figure in OTHER units (answer "1.31
    # km" vs page "1,310 m"), a deliberately rounded restatement, or a value the model
    # correctly computed from two fetched numbers all normalize to a value no page states.
    # Detection is unconditional either way -- the ``goal_achieved_numeric_unverified`` marker
    # records which figures failed -- so the false-positive rate is measurable live before the
    # downgrade is turned on anywhere.
    require_numeric_provenance_enabled: bool = False
    # Act on ``alternative_branch.race_value_agreement``: do the members of a race group
    # actually agree on the VALUE they raced for? ``race_route_evidence`` establishes only
    # that the routes are DIFFERENT and never compares what they returned, so "all three
    # routes confirm X" has until now been narration no mechanism could contradict.
    #
    # Two effects when on, both mechanical: a ``disagree`` group is NOT resolved to a winner
    # (contradicting routes are the one case where discarding N-1 of them is indefensible) and
    # its conflict is appended to the merge's ``missing_requirements``, which the existing
    # consistency guard already turns into a not-achieved verdict; an ``agree`` verdict admits
    # a TIER 2 inferred group into winner selection, since routes that returned the same value
    # cost nothing to collapse whether or not the group was a genuine race.
    #
    # Opt-in, default OFF: the verdict is computed and stamped (``race_value_agreement`` on the
    # shared ancestor) unconditionally, so its live behaviour -- especially the false-disagree
    # risk from equivalent-but-differently-derived figures, or one route reading a page updated
    # after the other -- is measurable in report captures before it decides anything.
    race_value_agreement_enabled: bool = False
    # Require one page-attributable verdict per candidate the mandate ENUMERATES before an
    # achieved verdict stands (``candidate_roster.audit_candidate_roster``). Targets the
    # 2026-08-21 diagnosis of task 122: the score gain there came from a config cap that FORCED
    # all-four coverage, never from the engine deciding to check every candidate, and a
    # "checked" candidate is recorded as a bare visit -- no field anywhere says WHY it was
    # eliminated, so a silent skip and a real disqualification are indistinguishable.
    #
    # Three findings from the same strong-agent trace ride this one flag, because all three are
    # the same record: the per-candidate disqualifier quote (B6), the magnitude tripwire that
    # blocks electing a survivor while a LARGER candidate carries no quotable disqualifier
    # (B7 -- RATAN-600 is physically bigger than FAST and is still correctly eliminated, on a
    # geometry predicate pure argmax cannot see), and the year token a time-indexed superlative
    # demands (B8 -- Arecibo collapsed in December 2020, straddling most training cutoffs).
    #
    # Scope is deterministic and narrow: an enumerated NAME list plus an individual-disposition
    # cue, which over the 165 task modules selects 39 mandates -- every branch-eliminate /
    # survivor / AND-filter task and no breadth-argmax, question or logic-constraint list.
    #
    # Opt-in, default OFF. Residual risk: the corroboration bar is token overlap against the
    # candidate's own page, so a verdict paraphrased past recognition, or one resting on a page
    # the run reached under a different title, reads as unsourced. Detection is unconditional
    # either way -- the audit is stamped on the root (``candidate_roster``) and the gaps on the
    # merge node -- so the false-positive rate is measurable before the downgrade is used.
    candidate_roster_enabled: bool = False
    # Require a CHAIN-shaped mandate's hops to be relation-linked to each other before an
    # achieved verdict stands (``chain_closure.audit_chain_closure``): some page the run
    # fetched must state an EARLIER fetched page's entity near the relation the mandate hops
    # on ("birthplace"), which is the back-reference a careful researcher checks before
    # trusting a hop hand-off.
    #
    # The failure it targets is invisible to every other check in this method: a wrong entity
    # at hop k produces a REAL page with REAL content, so the visit happens, visit-count
    # grounding passes, the page-identity guard passes and ``answer_numeric_provenance``
    # passes -- the 2026-08-21/22 trace's live decoy (Temuco, real infobox elevation 360 m,
    # described in a search snippet as Neruda's "native town") would satisfy all of them while
    # answering about the wrong town. The error is in the RELATION between two hops rather
    # than in either hop's execution.
    #
    # Opt-in, default OFF, on a false-positive risk that is genuinely unmeasured rather than
    # merely residual: a CORRECT hop's page need not mention the entity that led to it at all
    # (plenty of town pages list no notable residents), and the relation vocabulary is a small
    # curated list. Detection is unconditional either way -- the audit is stamped on the merge
    # node (``chain_closure``) and a failed closure marked (``chain_closure_open``) -- so the
    # rate is countable from report captures before the downgrade is used anywhere.
    chain_closure_enabled: bool = False

    _KEYS: ClassVar[dict] = {
        "model": "merge_model",
        "temperature": "merge_temperature",
        "max_tokens": "merge_max_tokens",
        "max_json_chars": "merge_max_json_chars",
        "require_substantive_children_enabled": "merge_require_substantive_children_enabled",
        "retry_after_skip_enabled": "merge_retry_after_skip_enabled",
        "require_visited_evidence_enabled": "merge_require_visited_evidence_enabled",
        "require_numeric_provenance_enabled": "merge_require_numeric_provenance_enabled",
        "goal_evaluation_first_enabled": "merge_goal_evaluation_first_enabled",
        "race_winner_selection_enabled": "merge_race_winner_selection_enabled",
        "race_winner_selection_includes_inferred_groups_enabled": (
            "merge_race_winner_selection_includes_inferred_groups_enabled"
        ),
        "race_value_agreement_enabled": "merge_race_value_agreement_enabled",
        "candidate_roster_enabled": "merge_candidate_roster_enabled",
        "chain_closure_enabled": "merge_chain_closure_enabled",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "MergeConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class VerifyConfig:
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = 1024
    # Emit ``reasoning`` BEFORE the ``verdict`` it justifies. Opt-in, default OFF.
    #
    # promptbench v2 (2026-08-19): pooled A2-A1 = +0.142, CI [+0.053, +0.232],
    # permutation p = 0.0119, positive on 5/5 models -- the pre-registered primary.
    # End-to-end transfer is unmeasured.
    #
    # Interacts with ``max_tokens`` above: reasoning-first spends the 1024-token budget on
    # prose before the verdict, so a live A/B must report parse-failure rate alongside
    # accuracy. See ``VerifyLeafAction._REASON_FIRST_SYSTEM_PROMPT``.
    reason_first_enabled: bool = False

    _KEYS: ClassVar[dict] = {
        "model": "verify_model",
        "temperature": "verify_temperature",
        "max_tokens": "verify_max_tokens",
        "reason_first_enabled": "verify_reason_first_enabled",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "VerifyConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class PlanLibraryConfig:
    """Retrieval-augmented planning. The ``plan_library_*`` keys.

    A new subsystem rather than a GoT-optimisation knob, so it gets its own group and does
    NOT use ``GoTConfig``'s ``got_`` auto-prefix. ``enabled`` is the master switch;
    ``auto_enabled`` governs the automatic pre-expansion short-circuit (a confident template
    match replaces the LLM's invented decomposition) and ``action_enabled`` the on-demand
    ``plan_library_search`` leaf action (the model asks for a strategy itself, and an adopted
    one grows children through ``_maybe_plan_library_reexpand``). The two sub-flags are
    independent (auto-only, on-demand-only or both) mirroring the
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
class StrategyLibraryConfig:
    """Retrieval-augmented *advice*. The ``strategy_library_*`` keys.

    The sibling of :class:`PlanLibraryConfig`, modelled on it directly, for the OTHER library:
    ``strategy_library/`` holds generalized prose notes rather than slot-parameterized DAG
    blueprints, and is consumed on the ``graph_compiled`` path (spliced into the offline
    authoring meta-prompt ``testing/scaffold_compiler`` and into that path's aggregation
    prompt) rather than through the native engine's expansion.

    One flag, not three: unlike the plan library there is no auto-vs-on-demand choice to make,
    because a note is never *applied*, only appended to a prompt. Default OFF -> byte-identical.

    Deliberately carries NO similarity threshold, for the same reason ``PlanLibraryConfig``
    does not: the decision constant lives next to the retrieval that owns it
    (``strategy_library/retrieval.APPLY_THRESHOLD``), and a second engine-level gate would
    silently disagree with it. Nor does it carry a leak-gate switch. The gate is not optional.
    """

    enabled: bool = False

    #: Second, narrower gate for the native engine's own expansion prompt (idea_engine.py /
    #: idea_policies/expansion.py). Kept separate from ``enabled`` so turning this on for a
    #: native arm profile can never silently also turn on retrieval for the ``graph_compiled``
    #: path — the two consumers must be independently controllable.
    native_expansion_enabled: bool = False

    _KEYS: ClassVar[dict] = {
        "enabled": "strategy_library_enabled",
        "native_expansion_enabled": "strategy_library_native_expansion_enabled",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "StrategyLibraryConfig":
        return _build(cls, settings)


def _action_names(raw: Any) -> Tuple[str, ...]:
    """Normalise an action-name sequence (JSON array / tuple) to a tuple of strings.

    JSON has no tuple type, so a shipped list arrives as a list and ``_coerce`` passes it
    straight through; the frozen views need something hashable and immutable. A bare string is
    treated as a one-element menu rather than silently exploding into characters.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw)


@dataclass(frozen=True)
class ToolsConfig:
    """Which tools a run may use. The ``tools_*`` keys.

    Tool availability used to be a bare content key (``allowed_actions``, a literal list set at
    each entry point) plus an unused extension point: ``IdeaDagEngine.install_action_pack`` was
    built and unit-tested but had no production call site, so "run this task with the calculator
    but no file writes" was not expressible without editing code. This group makes it declarative,
    modelled on :class:`PlanLibraryConfig`. Master flag per pack, independent sub-flags, all
    default OFF -> byte-identical.

    ``core_actions`` is the always-available menu (today's hard-coded six, in their order).
    ``tools_core_actions`` is deliberately ABSENT from ``idea_dag_settings.json``: the shipped
    source of truth for that list is still the legacy ``allowed_actions`` key, which callers
    (``idea_test_runner``, ``debug_runner``, tests) already override per run. Shipping both would
    make a JSON default silently clobber those overrides, so ``from_settings`` reads the legacy
    key when the new one is unset, and the new one only exists to *override* it.

    ``sandbox_pack_actions`` is the subset of :class:`SandboxToolPack` that actually gets
    permitted when the pack is on. The pack is always installed whole (the registry learns every
    class it ships); narrowing happens at the ``allowed_actions`` gate, so "sandbox on, read_file
    only" is a config change rather than an edit to ``SandboxToolPack.ACTION_CLASSES``.
    """

    core_actions: Tuple[str, ...] = ("search", "visit", "save", "think", "merge", "verify")
    sandbox_pack_enabled: bool = False
    sandbox_pack_actions: Tuple[str, ...] = (
        "read_file", "write_file", "list_dir", "count_lines",
        "word_count", "head_file", "disk_usage", "find_files",
    )
    calculator_pack_enabled: bool = False

    _KEYS: ClassVar[dict] = {
        "core_actions": "tools_core_actions",
        "sandbox_pack_enabled": "tools_sandbox_pack_enabled",
        "sandbox_pack_actions": "tools_sandbox_pack_actions",
        "calculator_pack_enabled": "tools_calculator_pack_enabled",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "ToolsConfig":
        built = _build(cls, settings)
        core = settings.get(cls._KEYS["core_actions"])
        if core is None:
            # Legacy key still wins when the typed override is unset (see the class docstring):
            # a run that narrows `allowed_actions` keeps its narrowed menu.
            core = settings.get("allowed_actions", built.core_actions)
        return replace(
            built,
            core_actions=_action_names(core),
            sandbox_pack_actions=_action_names(built.sandbox_pack_actions),
        )


@dataclass(frozen=True)
class ActionConfig:
    max_retries: int = 2
    retry_backoff_steps: int = 1
    max_observation_chars: int = 100000
    default_search_count: int = 10
    max_links_per_visit: int = 20
    visit_max_sites_per_action: int = 20
    visit_link_query_top_k: int = 15
    # Links a single visited page may index into Chroma. ChromaDB's default embedding
    # function runs CLIENT-side (ONNX MiniLM, ~19ms/doc) inside the async client's
    # `add` coroutine, so an uncapped store of a link-dense page (a Wikipedia article
    # yields ~990 links after chrome filtering) burns ~19s of on-loop CPU and eats the
    # whole 20s visit budget. 100 keeps a page's link index at ~2s while still giving
    # the link query a wide field to choose from.
    visit_link_store_max: int = 100
    visit_page_concurrency: int = 5
    visit_link_selection_model: Optional[str] = None
    visit_empty_content_retryable: bool = True
    # A3b: reasoning-effort/token discipline for native leaf micro-prompts (opt-in). When on,
    # a reasoning-model executor's perception/selection micro-prompt uses reasoning_effort=minimal
    # and its token budget is floored so hidden reasoning can't starve the completion (the
    # content=None bug fixed on the compiled path). Default OFF -> byte-identical.
    native_reasoning_effort_discipline_enabled: bool = False
    native_reasoning_min_tokens_floor: int = 2048
    # A5: price-tier parameter tiering for native executor micro-prompts (opt-in). When on, a
    # micro-prompt's token budget scales by the executor model's price tier (cheap stays tight,
    # mid/premium get headroom). Default OFF -> byte-identical.
    price_tier_param_tiering_enabled: bool = False
    # C1a: tool-failure recovery (opt-in). ``connector_retry_on_failure_enabled``: when a leaf
    # action returns a TOOL failure (empty/timeout/HTTP-error fetch, no search results), retry the
    # SAME action in place with bounded backoff before deciding the node's fate. So a TRANSIENT
    # failure recovers at the source instead of the re-expansion loop spawning a subtree that
    # repeats the failing fetch. ``tool_failure_recovery_enabled``: route the low-confidence
    # re-expansion trigger AWAY from re-expanding a leaf whose low score was caused by a tool
    # failure (a fresh subtree would just repeat it). Both default OFF -> byte-identical.
    connector_retry_on_failure_enabled: bool = False
    connector_retry_max_attempts: int = 2
    connector_retry_backoff_seconds: float = 0.5
    tool_failure_recovery_enabled: bool = False
    # Sibling visit-URL dedup (opt-in). A visit leaf that arrives without a URL of its own is
    # handed one by a sibling-blind cascade (parents/sibling results/Chroma links), which ranks
    # the same pool the same way for every such sibling -- so a fan-out of per-entity page reads
    # collapses onto ONE page (16.2% of recorded sibling-visit batches contain a duplicate and
    # half of those have EVERY sibling on one page; ASSUMPTION_AUDIT.md T1-4). When on, a
    # fallback-resolved URL a sibling already claimed is dropped and the next candidate is used
    # instead. An explicitly declared URL is never dropped -- that duplicate is the planner's
    # own instruction, a different defect. Default OFF -> byte-identical.
    visit_sibling_url_dedup: bool = False
    # Dead declared-URL fallback (kill-switch, default ON). ``io.fetch_url`` RAISES on a permanent
    # HTTP status (404/403), and that exception used to escape the whole visit action -- skipping
    # the URL-recovery cascade (parent search hits / sibling results / stored link index) the code
    # already falls through to when the declared URL merely RETURNS a failure. A planner-guessed
    # Wikipedia title is the common cause (all but a handful of the 107 permanent visit failures in
    # the recorded corpus are guessed en.wikipedia.org titles), and the cascade routinely holds the
    # real page, so a 404 threw away a recoverable hop. When on, the raised failure falls through to
    # the cascade; if the cascade resolves nothing NEW the original error is re-raised, so the
    # action's failure surface is unchanged. Off -> byte-identical to the old abort.
    visit_dead_url_fallback_enabled: bool = True
    # Site-chrome filtering of the visit URL POOL (opt-in). The chrome test added for the dead-URL
    # recovery harvest is applied there only, so every OTHER pool a URL-less visit resolves from --
    # ancestor page links (``_extract_url_from_parents``), sibling results, the Chroma link index --
    # still offers donation appeals, login/create-account forms and portal plumbing as if they were
    # content. They out-rank real pages because a chrome link routinely carries the leaf's own words
    # in a campaign/``returnto=`` parameter, and they sit FIRST in a Wikipedia page's link order, so
    # a score tie resolves to them. 64 of 2134 executed sibling visits in the recorded corpus (3.0%,
    # across 35 runs) fetched one -- a donation page read as evidence. When on, chrome URLs are
    # dropped from the resolved pool before selection. A DECLARED URL is never dropped (same
    # principle as the sibling dedup above: that is the planner's own instruction).
    # Default OFF -> byte-identical.
    visit_chrome_link_filter: bool = False

    _KEYS: ClassVar[dict] = {
        "max_retries": "action_max_retries",
        "retry_backoff_steps": "action_retry_backoff_steps",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "ActionConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class SandboxActionConfig:
    """Limits for the CODE-domain sandbox actions (the ``sandbox_*`` settings keys).

    The web-research arms act on the world through ``search``/``visit``; the coding arm
    (``graph_compiled_code``, see ``testing/execution_compiled_code.py``) acts on a writable
    workdir through ``connector_sandbox.SandboxConnector``. These are that connector's cost/blast
    bounds: where the workdir lives, how long one subprocess action may run, and how much a single
    leaf may write. They are DEFENCE IN DEPTH, not the security boundary. The container itself
    (read-only rootfs, dropped caps, tmpfs ``/work``, outer 900s wall-clock kill) is that.
    """

    workdir_root: str = "/work"
    run_pytest_timeout_seconds: int = 30
    run_python_timeout_seconds: int = 15
    #: Wall-clock bound on ONE allow-listed read-only shell command (``connector_sandbox``'s
    #: ``run_readonly``: wc/grep/du/find/head). Small on purpose. These inspect a scratch workdir,
    #: so anything slower than this is a pathological pattern or tree, not useful work. Matches the
    #: 10s bound the ported ``badmodel-lab/localagent/tools/shell.py`` used.
    shell_timeout_seconds: int = 10
    max_file_bytes: int = 200000
    max_files_per_leaf: int = 10
    #: Wall-clock bound on ONE leaf-loop decision call. The subprocess actions above are bounded by
    #: their own timeouts, but an LLM completion is not: a hung backend would otherwise stall a step
    #: forever and silently eat the outer 900s container budget with zero step-budget progress.
    #: Sized between the two: generous next to a 30s pytest run (a ``write_file`` decision carries a
    #: whole source file), yet small enough that a leaf's full step budget still fits in the run.
    llm_call_timeout_seconds: int = 90
    #: Wall-clock bound on the CALCULATOR ``run_python`` leaf action
    #: (``idea_policies/extra_actions/calculator_tools.py``), deliberately tighter than the coding
    #: arm's 15s ``run_python_timeout_seconds``. That budget covers a generated module importing
    #: third-party packages and doing real work; the calculator only ever recomputes over facts the
    #: run already gathered (max of six numbers, a ratio, a subset sum), which is milliseconds even
    #: with interpreter start-up. Anything slower is a runaway loop, and inside a latency-sensitive
    #: web-research leaf it is cheaper to hand the model a fast "timed out" observation it can retry
    #: than to stall the step. Matches ``shell_timeout_seconds``. Same "inspect something small"
    #: class of work.
    calculator_timeout_seconds: int = 10

    _KEYS: ClassVar[dict] = {
        "workdir_root": "sandbox_workdir_root",
        "run_pytest_timeout_seconds": "sandbox_run_pytest_timeout_seconds",
        "run_python_timeout_seconds": "sandbox_run_python_timeout_seconds",
        "shell_timeout_seconds": "sandbox_shell_timeout_seconds",
        "max_file_bytes": "sandbox_max_file_bytes",
        "max_files_per_leaf": "sandbox_max_files_per_leaf",
        "llm_call_timeout_seconds": "sandbox_llm_call_timeout_seconds",
        "calculator_timeout_seconds": "sandbox_calculator_timeout_seconds",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "SandboxActionConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class MemoryConfig:
    document_chunk_threshold: int = 200000
    document_chunk_size: int = 4000
    document_chunk_overlap: int = 400
    expansion_chroma_internal: int = 5
    expansion_chroma_observations: int = 5
    # ``leaf_chroma_results`` / ``default_semantic_results`` lived here and in all three
    # settings files with ZERO readers, in every commit since the keys were introduced
    # (git-archaeology, ASSUMPTION_AUDIT.md T1-2): declared surface that never had a
    # consumer to lose. Deleted 2026-08-20 rather than wired, since there is no evidence of
    # an intended call site. Leaf/semantic retrieval keeps using its own explicit top-k.
    max_available_links_for_expansion: int = 50
    grep_context_window: int = 80
    # Minimum cosine similarity a retrieved memory must clear before it may enter a prompt
    # (ASSUMPTION_AUDIT.md T3-1: a k-NN query returns k rows however far away they are).
    # 0.0 = admit everything = the behaviour every measurement to date was taken under, so
    # this ships OFF; it is a lever for E3, not a tuned value. Raise it only with an A/B.
    retrieval_similarity_floor: float = 0.0
    # Rank the finalize prompt's pooled chroma context by similarity before the
    # final_chroma_results cap, instead of by the order its four query batches were issued
    # (ASSUMPTION_AUDIT.md T3-3). Off = the historical arrival order.
    final_context_rank_by_similarity: bool = False
    # Restrict expansion's vector retrieval to memories written by the node's OWN lineage
    # (``graph.path_to_root``), instead of every node in the run. ``write_memory`` has always
    # stamped ``node_id`` on each chunk, but retrieval only ever filtered on ``memory_type``,
    # so a node could read a sibling branch's results -- which makes the race-and-merge
    # premise ("racing branches are independent routes to the same fact") untrue by
    # construction. ``GoTOperations.hybrid_retrieve`` already scopes its graph-walk half this
    # exact way; this applies the same scope to the vector half.
    #
    # Opt-in, default OFF: narrowing scope can plausibly help (less cross-branch noise, a
    # genuinely blind race) or hurt (a useful cross-branch precedent -- "that search already
    # failed elsewhere" -- gets cut off). That is an A/B, not a blind default flip.
    branch_scoped_retrieval_enabled: bool = False

    _KEYS: ClassVar[dict] = {
        "retrieval_similarity_floor": "memory_retrieval_similarity_floor",
        "final_context_rank_by_similarity": "final_context_rank_by_similarity",
        "branch_scoped_retrieval_enabled": "memory_branch_scoped_retrieval_enabled",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "MemoryConfig":
        return _build(cls, settings)


@dataclass(frozen=True)
class EngineConfig:
    """Execution / graph-shape / parallelism / logging knobs read by the engine."""

    max_branching: int = 5
    # Demand-driven root fan-out. ``max_branching`` is a GLOBAL budget shared by every task
    # shape, and at 5 it capped the whole graph below the size of a 7-candidate question --
    # the four-way baseline's mean node count was 4.6, i.e. the cap WAS the graph. Raising the
    # constant would hand chain and narrow tasks a width they have no use for, so instead the
    # ROOT (only) may widen to the number of candidates the mandate actually enumerates, via
    # the same parser ``candidate_coverage`` uses. Off by default so it can be A/B'd.
    breadth_aware_branching_enabled: bool = False
    #: Hard ceiling on the widened root fan-out, so a pathological enumeration cannot mint an
    #: unbounded number of children.
    breadth_branching_max: int = 8
    # Let a remediation path re-expand a node that already has children. ``step()`` gated
    # expansion on ``not node.children``, so the root expanded exactly once per run -- and
    # every remediation path (coverage extension, grounding replan, budget extension) ends by
    # re-activating the root, which then fell through to the intermediate handler and could
    # only pick among nodes that already existed. The engine could detect incompleteness and
    # was structurally unable to act on it. Off by default.
    root_reexpansion_enabled: bool = False
    #: How many times one node may be re-expanded in a run. Bounds the remediation loop.
    root_reexpansion_max: int = 2
    # Make coverage remediation create VISITS. The gate counts only successful visits, but the
    # only remediation it could reach was the mandate-phrase/navigation hooks, which decline on
    # an ordinary enumerated mandate -- so a detected gap produced more SEARCHING. Measured: 46
    # searches / 1 visit (n=24 A/B, null result), and 55 searches / 2 visits once the structural
    # caps were lifted. Off by default.
    coverage_visit_injection_enabled: bool = False
    #: Ceiling on candidate visit-pairs minted per remediation pass.
    coverage_visit_injection_max: int = 8
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
    # Deterministic (LLM-free) value threading: extract the datum a completed VISIT leaf's
    # page carries (a number near its contract's own cue wording, or an entity/place via
    # `action_result.link_contexts`) and stash it on the node as `details["waypoint"]` for a
    # downstream hop. See `idea_policies/waypoint.py`. Opt-in and default OFF: a wrong-page
    # false-positive rate measurement (`scripts/replay_waypoints.py --precision`) gates this on
    # a page-identity guard rather than emitting freely.
    waypoint_enabled: bool = False  # absent from JSON
    # Deterministic (LLM-free) unresolved-slot detection for `search`/`visit` candidates
    # (idea_policies/dataflow.py's `unresolved_slots`) and the two engine call sites it
    # feeds (idea_sequencing.py's `siblings_are_independent` /
    # `defer_unresolved_slot_candidates`). A read-only census
    # (scripts/measure_dataflow_slots.py) found 19/418 chain-task search/visit leaves
    # executed with a literal unfilled placeholder (e.g. `<to be determined after previous
    # visit>`) still in `optional_url`/`query`/`link_idea` -- 0% false positives (incl. on
    # fan-out batches), 3 failed outright, 16 were "silently repaired" by
    # VisitLeafAction's unscoped sibling-URL scavenging fallback. Both opt-in and default
    # OFF pending a live A/B.
    parallel_requires_evidence: bool = False  # absent from JSON
    defer_unresolved_slots: bool = False  # absent from JSON
    # The resolved-value channel: before dispatching a node that declares
    # `requires_data.slot`, fill that slot from its NAMED source's structured output --
    # the source's `waypoint` first, else the contract's own `value_for` reader
    # (`idea_policies/data_contracts.py`). This is the write-back the deferral mechanism
    # above lacks: `defer_unresolved_slots` changes WHEN a node runs, never what it
    # resolves to, so a deferred node came back and hit the same unscoped sibling-URL
    # scavenging in `VisitLeafAction`. Only nodes whose writer declared a `slot` are
    # touched, and an unresolvable one is a logged no-op, so the existing fallback still
    # owns every path this does not fill. Opt-in and default OFF pending a live A/B; see
    # docs/handoffs/RESOLVED_VALUE_CHANNEL_DESIGN_2026-08-16.md. Wants `waypoint_enabled`
    # on too (the engine warns once at construction otherwise): with it off there is no
    # waypoint to read and only the weaker contract fallback remains.
    resolved_value_channel_enabled: bool = False  # absent from JSON
    # The evaluation-ordering invariant (ENGINE_DESIGN_REVIEW.md PART 3): "a decision that
    # consumes a score must run after that score exists". Two sites violate it, and each gets
    # one opt-in, default-OFF flag here because the fix changes graph shape and token spend,
    # so turning it on invalidates comparisons against in-flight benchmark numbers.
    #
    # `beam_after_evaluation`: the expansion beam (`candidates[:max_branching]`) truncates
    # before any candidate is scored, so the "beam" is really the model's emission order. On:
    # every surviving candidate becomes a node, the batch evaluator scores them all, and the
    # beam keeps the top `max_branching` by score (the rest are SKIPPED, marked
    # `__beam_pruned`). Only binds where selection happens, i.e. the sequential path: an
    # auto-parallel batch executes every sibling by design, so there is no beam to apply.
    #
    # `evaluate_parallel_siblings`: the auto-parallel batch path returns before the
    # evaluation block, so its siblings keep `score is None` forever -- which is the
    # documented root cause behind dormant prune/backtrack/confidence machinery
    # (TECHNIQUE_INVENTORY.md:42-50). On: the batch is scored once its results are in
    # (one batch call), which is strictly better-informed than the sequential path's
    # pre-execution scoring (that one is capped by `no_action_result_score_cap`).
    beam_after_evaluation: bool = False  # absent from JSON
    evaluate_parallel_siblings: bool = False  # absent from JSON
    # A6's calibrated early exit (`got.confidence_early_exit_enabled`) decides from a
    # confidence statistic alone, with no awareness of the mandate's substantiation
    # requirements -- and the hooks that would inject the missing visit only fire during
    # normal step expansion. So an early exit on a grounded-research mandate can reach
    # finalize with zero opened pages, where `final_require_grounding` (default OFF) is the
    # only thing that would catch it (ENGINE_DESIGN_REVIEW.md D6). On: an earned early exit
    # is DECLINED while `idea_finalize.grounding_gate_would_refuse` still holds, so the run
    # keeps expanding and the grounding-injection hooks get their chance. Declining is the
    # whole mechanism -- the engine never forces a visit action itself, which would couple
    # the control loop to one policy action. Independent of `final_require_grounding`: this
    # asks the gate's question earlier, it does not turn the gate on.
    early_exit_respects_grounding_enabled: bool = False

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
class RunPolicy:
    """Run-scoped semantics decided ONCE per mandate. The ``run_policy_*`` keys.

    Every other group here tunes a *stage* (expansion, merge, a leaf action). This one
    holds the decisions a profile makes about the run as a whole, so a later run-level
    subsystem does not have to smuggle its switch into whichever stage group happens to
    read it first.

    ``ledger_mode`` is the first such switch: ``"off"`` (today's behaviour) or ``"observe"``
    (the task ledger records the run without steering it). Deliberately a mode string rather
    than a bool, because the intended progression is off -> observe -> enforce and a bool
    would have to be replaced the moment a third state exists. Absent from
    ``idea_dag_settings.json`` on purpose: nothing reads it yet, so the dataclass default is
    the only source of truth and the shipped settings stay byte-identical.

    ``search_must_yield_visit`` states the run-level contract that a completed SEARCH has to
    hand the run at least one page worth opening. When it does not (empty results, or every
    result already visited elsewhere), the empty-search remediation in
    ``post_expansion_hooks.inject_empty_search_followup`` mints one broadened search plus a
    visit that depends on it. Off by default and absent from the shipped settings for the same
    reason as above: flag-off behaviour is byte-identical to not having the mechanism.

    ``sibling_context_delta`` lets an expansion prompt see, in one bounded line, which of the
    run's requirements OTHER branches have already resolved. Expansion context is root-ward
    only (``IdeaDag.path_to_root``), so today a node being expanded cannot tell that a sibling
    already covered half the roster. DEPENDS ON ``ledger_mode == "observe"``: the block is a
    rendering of the task ledger snapshot at ``root.details["task_ledger_v1"]``, so with the
    ledger off there is nothing to render and the flag is inert on its own. Off by default and
    absent from the shipped settings, same as the two above.

    ``evidence_store_mode`` decides whether a completed VISIT also records a structured
    ``Evidence`` (free, deterministic) and its extracted ``Claim`` triples (ONE cheap LLM call
    per visited page) as sidecar node details — see ``agent/app/evidence_store.py``. A mode
    string for the same reason as ``ledger_mode``: off -> observe -> (later) a mode where the
    claims are verified and consumed. ``"observe"`` costs one extra LLM call per successful
    visit and changes no decision. Off by default and absent from the shipped settings, same as
    the three above.

    ``deterministic_merge_view`` records, ALONGSIDE the LLM merge synthesis and without
    touching it, what the merge node's sources actually claimed: the descendants' ``Claim``
    triples grouped by subject — see ``evidence_store.aggregate_claims_for_merge``. DEPENDS ON
    ``evidence_store_mode == "observe"``: the view is an aggregation of claim sidecars, so with
    the store off there is nothing to aggregate and the flag is inert on its own (same
    dependency shape as ``sibling_context_delta`` on ``ledger_mode``). The merge's
    ``action_result``, deliverable, ``goal_achieved`` and prompts are untouched either way. Off
    by default and absent from the shipped settings, same as the four above.

    ``merge_uses_evidence_view`` is the first flag in this group that CONSUMES rather than
    observes: it appends one bounded ``[Evidence]`` block — subjects with their claim and
    distinct-source counts, no raw claim text — to the merge synthesis system prompt, so the
    model aggregating the branches can see what its own sources actually recorded. DEPENDS ON
    BOTH ``deterministic_merge_view`` AND ``evidence_store_mode == "observe"``, a three-flag
    chain that reads bottom-up: the store records the ``Claim`` sidecars, the view is their
    per-subject aggregation, and this flag renders that aggregation into the prompt. With any
    link off, no block is built and the merge prompt is byte-identical to today's. The LLM
    stays the decision-maker — nothing here overrides ``goal_achieved`` or the deliverable, it
    only changes what the model is shown. Off by default and absent from the shipped settings,
    same as the five above.
    """

    ledger_mode: str = "off"
    search_must_yield_visit: bool = False
    sibling_context_delta: bool = False
    evidence_store_mode: str = "off"
    deterministic_merge_view: bool = False
    merge_uses_evidence_view: bool = False

    _KEYS: ClassVar[dict] = {
        "ledger_mode": "run_policy_ledger_mode",
        "search_must_yield_visit": "run_policy_search_must_yield_visit",
        "sibling_context_delta": "run_policy_sibling_context_delta",
        "evidence_store_mode": "run_policy_evidence_store_mode",
        "deterministic_merge_view": "run_policy_deterministic_merge_view",
        "merge_uses_evidence_view": "run_policy_merge_uses_evidence_view",
    }

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "RunPolicy":
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
    strategy_library: StrategyLibraryConfig
    tools: ToolsConfig
    action: ActionConfig
    sandbox: SandboxActionConfig
    memory: MemoryConfig
    engine: EngineConfig
    policy: PolicyConfig
    run_policy: RunPolicy

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
            strategy_library=StrategyLibraryConfig.from_settings(settings),
            tools=ToolsConfig.from_settings(settings),
            action=ActionConfig.from_settings(settings),
            sandbox=SandboxActionConfig.from_settings(settings),
            memory=MemoryConfig.from_settings(settings),
            engine=EngineConfig.from_settings(settings),
            policy=PolicyConfig.from_settings(settings),
            run_policy=RunPolicy.from_settings(settings),
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
