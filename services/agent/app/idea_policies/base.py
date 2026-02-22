from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from agent.app.idea_dag import IdeaDag, IdeaNode


class ExpansionPolicy(ABC):
    """
    Base class for idea expansion behavior.
    :param settings: Settings dictionary.
    :returns: ExpansionPolicy instance.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    async def expand(self, graph: IdeaDag, node_id: str) -> List[Dict[str, Any]]:
        """
        Generate candidate child ideas for a node.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: List of idea dicts.
        """
        raise NotImplementedError()


class EvaluationPolicy(ABC):
    """
    Base class for idea evaluation behavior.
    :param settings: Settings dictionary.
    :returns: EvaluationPolicy instance.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    async def evaluate(self, graph: IdeaDag, node_id: str) -> float:
        """
        Score a node based on graph context.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: Score value.
        """
        raise NotImplementedError()


class SelectionPolicy(ABC):
    """
    Base class for idea selection behavior.
    :param settings: Settings dictionary.
    :returns: SelectionPolicy instance.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    def select(self, graph: IdeaDag, parent_id: str) -> Optional[IdeaNode]:
        """
        Choose the next node to execute.
        :param graph: IdeaDag instance.
        :param parent_id: Parent node identifier.
        :returns: Selected IdeaNode or None.
        """
        raise NotImplementedError()


class DecompositionPolicy(ABC):
    """
    Base class for decomposition behavior.
    :param settings: Settings dictionary.
    :returns: DecompositionPolicy instance.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    def should_decompose(self, graph: IdeaDag, node_id: str) -> bool:
        """
        Decide whether a node should be decomposed.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: True if decomposition should occur.
        """
        raise NotImplementedError()


class MergePolicy(ABC):
    """
    Base class for merge behavior.
    :param settings: Settings dictionary.
    :returns: MergePolicy instance.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    def merge(self, graph: IdeaDag, node_id: str) -> Dict[str, Any]:
        """
        Merge child results into a parent summary payload.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: Merge payload.
        """
        raise NotImplementedError()


class MemoizationPolicy(ABC):
    """
    Base class for memoization behavior.
    :param settings: Settings dictionary.
    :returns: MemoizationPolicy instance.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = dict(settings or {})

    @abstractmethod
    def get_key(self, graph: IdeaDag, node_id: str) -> Optional[str]:
        """
        Derive memoization key for a node.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: Memoization key or None.
        """
        raise NotImplementedError()

    @abstractmethod
    def should_reuse(self, graph: IdeaDag, node_id: str) -> bool:
        """
        Decide whether memoized data should be reused.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: True if reuse should occur.
        """
        raise NotImplementedError()


class IdeaActionType(str, Enum):
    """
    Supported leaf action types.
    """
    THINK = "think"
    SEARCH = "search"
    VISIT = "visit"
    SAVE = "save"
    MERGE = "merge"


class DetailKey(str, Enum):
    """
    Canonical node detail keys.
    """
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