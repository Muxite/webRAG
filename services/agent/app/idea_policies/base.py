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
    PARENT_GOAL = "parent_goal"
    IS_LEAF = "is_leaf"
    JUSTIFICATION = "justification"
    WHY_THIS_NODE = "why_this_node"
    PARENT_JUSTIFICATION = "parent_justification"