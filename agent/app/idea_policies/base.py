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
    REQUIRES_DATA = "requires_data"
    PROVIDES_DATA = "provides_data"
    DATA_SOURCE_NODE = "data_source_node"
    GOAL = "goal"
    GOAL_ACHIEVED = "goal_achieved"
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