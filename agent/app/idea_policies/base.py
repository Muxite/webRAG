from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode


class ExpansionPolicy(ABC):
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    async def expand(self, graph: IdeaDag, node_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError()


class EvaluationPolicy(ABC):
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    async def evaluate(self, graph: IdeaDag, node_id: str) -> float:
        raise NotImplementedError()


class SelectionPolicy(ABC):
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    def select(self, graph: IdeaDag, parent_id: str) -> Optional[IdeaNode]:
        raise NotImplementedError()


class DecompositionPolicy(ABC):
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    def should_decompose(self, graph: IdeaDag, node_id: str) -> bool:
        raise NotImplementedError()


class MergePolicy(ABC):
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    def merge(self, graph: IdeaDag, node_id: str) -> Dict[str, Any]:
        raise NotImplementedError()


class MemoizationPolicy(ABC):
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    def get_key(self, graph: IdeaDag, node_id: str) -> Optional[str]:
        raise NotImplementedError()

    @abstractmethod
    def should_reuse(self, graph: IdeaDag, node_id: str) -> bool:
        raise NotImplementedError()


class IdeaNodeStatus(str, Enum):
    """
    Status values for idea nodes.
    """
    PENDING = "pending"
    ACTIVE = "active"
    BLOCKED = "blocked"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class IdeaActionType(str, Enum):
    THINK = "think"
    SEARCH = "search"
    VISIT = "visit"
    SAVE = "save"
    MERGE = "merge"
    VERIFY = "verify"
    # On-demand plan-library retrieval (opt-in, gated by ``plan_library_enabled`` +
    # ``plan_library_action_enabled``): ask the pre-authored template corpus for a
    # composition strategy for this node. Read-only — the action ranks, fills and reports;
    # ``IdeaDagEngine._maybe_plan_library_reexpand`` is what turns an adopted template into
    # real children. Reachable only when the engine patches it into ``allowed_actions``, so
    # the default action menu is unchanged.
    PLAN_LIBRARY_SEARCH = "plan_library_search"


class DetailKey(str, Enum):
    ACTION = "action"
    QUERY = "query"
    PROMPT = "prompt"
    COUNT = "count"
    URL = "url"
    LINK = "link"
    PATTERN = "pattern"
    TEXT = "text"
    FLAGS = "flags"
    DOCUMENTS = "documents"
    DOCUMENT = "document"
    METADATAS = "metadatas"
    QUERIES = "queries"
    N_RESULTS = "n_results"
    EVALUATION = "evaluation"
    RATIONALE = "rationale"
    ACTION_RESULT = "action_result"
    ACTION_RESULTS = "action_results"
    ACTION_ATTEMPTS = "action_attempts"
    ACTION_MAX_RETRIES = "action_max_retries"
    ACTION_COOLDOWN_UNTIL = "action_cooldown_until"
    ACTION_RETRYABLE = "action_retryable"
    ACTION_ERROR = "action_error"
    MERGED_RESULTS = "merged_results"
    MERGE_SUMMARY = "merge_summary"
    MERGE_FAILURE = "merge_failure"
    EXPANSION_META = "expansion_meta"
    EXECUTE_ALL_CHILDREN = "execute_all_children"
    MEMO_NAMESPACE = "memo_namespace"
    INTENT = "intent"
    # Optional structured output contract for a LEAF candidate (opt-in, gated by
    # ``expansion_expect_contract_enabled``): a one-line measurable target — the exact
    # value to report AND that its source URL must accompany it. Threads from the
    # expansion candidate into leaf execution (as an extraction-target addendum to the
    # intent) and is auto-serialized into the evaluation context. Absent by default.
    EXPECT = "expect"
    PARENT_GOAL = "parent_goal"
    IS_LEAF = "is_leaf"
    JUSTIFICATION = "justification"
    WHY_THIS_NODE = "why_this_node"
    PARENT_JUSTIFICATION = "parent_justification"
    # A cross-node dependency record, written as a raw dict (no schema class):
    #   {"type": <DataContract name>, "source_node_id": <str>, "slot": <str | None>}
    # ``type``/``source_node_id`` are the readiness gate ``_has_required_data`` reads.
    # ``slot`` is the OPTIONAL resolved-value channel: the name of THIS node's own detail
    # field ("url" today) that the engine fills from the source's structured output at
    # dispatch time (``IdeaDagEngine._resolve_slot``, gated by
    # ``EngineConfig.resolved_value_channel_enabled``). ``slot`` is deliberately absent on
    # writers whose value is already concrete at authoring time; only a genuinely
    # unresolved-at-authoring field declares one. Readiness never reads ``slot``.
    REQUIRES_DATA = "requires_data"
    PROVIDES_DATA = "provides_data"
    # Deterministic value a completed VISIT leaf's page carried, for a downstream hop
    # (``idea_policies/waypoint.py``; written only when ``waypoint_enabled`` is on).
    WAYPOINT = "waypoint"
    DATA_SOURCE_NODE = "data_source_node"
    GOAL = "goal"
    # The AUTHORITATIVE verdict. Written only by ``MergeLeafAction`` (which demotes it on
    # missing requirements, snippet-only provenance and unverified numerics) and by the
    # upward propagation that follows it. ``idea_finalize.resolve_goal_achieved`` trusts it.
    GOAL_ACHIEVED = "goal_achieved"
    # ``SimpleMergePolicy._validate_goal_achievement``'s cheap keyword-overlap pre-check.
    # Deliberately a SEPARATE key: ``merge()`` recurses to root, so writing this one to
    # ``GOAL_ACHIEVED`` stamped an optimistic verdict on the ROOT before any merge node's LLM
    # call ran, and finalize's root-first read could never retract it -- a run covering 2 of 7
    # candidates still reported success. Diagnostic only; nothing gates on it.
    GOAL_ACHIEVED_PROVISIONAL = "goal_achieved_provisional"
    CHUNK_INDEX = "chunk_index"
    TOTAL_CHUNKS = "total_chunks"
    CHUNK_CONTENT = "chunk_content"
    ORIGINAL_GOAL = "original_goal"
    CLAIM = "claim"
    # Single-use human steer injected via the interactive debugger (agent-debug
    # `f`/`feedback`). Surfaced in the next expansion of the node and immediately
    # consumed-and-cleared, so it never persists onto the node/subtree afterward.
    HUMAN_FEEDBACK = "human_feedback"
    # Single-use corrective hint threaded onto a re-expanded node (opt-in, gated by
    # ``got_reexpand_corrective_context_enabled``): the triggering judge/detector's own
    # ``reason`` for why the prior step was inadequate (low confidence or a genuine
    # follow-up), surfaced once in the re-expansion prompt and immediately
    # consumed-and-cleared, mirroring ``HUMAN_FEEDBACK``. Absent by default.
    REEXPAND_REASON = "reexpand_reason"
    # Structural-degeneracy marker, written by ``ExpansionPolicy._create_fallback_candidate``
    # onto the ONE candidate it emits when the model returned no usable candidates at all.
    # That single candidate becomes the parent's entire expansion, so a subtree that should
    # have fanned out collapsed to one leaf derived from a regex/keyword guess. Pure
    # instrumentation today: it is counted into the final payload
    # (``degenerate_fallback_count``) so a run's structural starvation is visible without
    # re-parsing logs, and (opt-in via ``got_reexpand_fallback_nodes_enabled``) it is the
    # trigger for re-planning the collapsed parent.
    FALLBACK_EXPANSION = "fallback_expansion"
    # Written onto a fallback leaf whose PARENT was successfully re-planned by
    # ``_maybe_reexpand_fallback_parent``: the guessed action has been superseded by a real
    # decomposition, so the leaf is also marked ``SKIPPED``. Absent by default (the trigger
    # is opt-in).
    FALLBACK_SUPERSEDED = "fallback_superseded"
    # Sequential A->B fallback (opt-in, gated by ``expansion_alternative_branch_enabled``):
    # the node id of the PRIMARY this node is a pre-authored fallback for. Written by
    # ``idea_policies/alternative_branch.link_alternatives`` once ``graph.expand()`` has
    # minted real ids, resolved from the authored ``alternative_of`` title hint.
    ALTERNATIVE_OF_NODE = "alternative_of_node"
    # The back-pointer of ``ALTERNATIVE_OF_NODE``, written on the PRIMARY: the node id of the
    # fallback to promote when the primary fails or lands unverified. Its absence (every
    # existing arm) is what makes ``_maybe_promote_alternative_branch`` a no-op.
    HAS_ALTERNATIVE_NODE = "has_alternative_node"
    # Concurrent race-and-merge (same flag): the authored label shared by 2+ siblings that
    # are different routes to the SAME fact. Read by ``idea_sequencing.siblings_are_independent``
    # to dispatch the group concurrently and by ``SimpleMergePolicy.select_winner`` to resolve
    # one winner at the merge point. Absent by default.
    RACE_GROUP = "race_group"