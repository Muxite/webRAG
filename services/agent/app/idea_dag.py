from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Union
import uuid


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


@dataclass
class IdeaNode:
    """
    Represents a single unit of work in the problem-solving graph.
    
    Each node can be an expansion (breaks into sub-problems), a leaf (executes actions),
    or a merge (combines results). Nodes track their status, relationships, and results.
    
    **Node Types (determined by `details`):**
    - **Expansion**: Has `action=null`, will be expanded into children
    - **Leaf**: Has `action` field (search/visit/save/think), executes immediately
    - **Merge**: Has `action=MERGE`, combines children's results
    
    **Status Values:**
    - `PENDING`: Not yet processed
    - `ACTIVE`: Currently being worked on
    - `DONE`: Completed successfully
    - `BLOCKED`: Temporarily blocked (e.g., rate limit)
    - `FAILED`: Execution failed
    - `SKIPPED`: Intentionally skipped
    
    **Key Fields:**
    - `node_id`: Unique UUID identifier
    - `title`: Human-readable label (e.g., "Search for X")
    - `details`: Dict containing action, results, merged data, etc.
    - `status`: Current processing status
    - `score`: Evaluation score (0.0-1.0) for selection
    - `children`: List of child node IDs
    - `parent_id`: Primary parent node ID
    
    **Usage:**
    Users typically access nodes via `graph.get_node(node_id)`, then check:
    - `node.status` to see if work is complete
    - `node.details.get("action")` to see what action it performs
    - `node.details.get("result")` to see execution results
    - `node.is_leaf()` to check if it's a terminal node
    """
    node_id: str
    title: str
    details: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)
    status: IdeaNodeStatus = IdeaNodeStatus.PENDING
    children: List[str] = field(default_factory=list)
    score: Optional[float] = None
    memo_key: Optional[str] = None

    def is_leaf(self) -> bool:
        """
        Indicate whether the node is a leaf.
        :returns: True if node has no children.
        """
        return len(self.children) == 0


class IdeaDag:
    """
    Directed Acyclic Graph (DAG) structure for organizing problem-solving nodes.
    
    Manages a tree-like hierarchy of nodes where each node represents a unit of work.
    The graph starts with a root node and grows as problems are expanded into sub-problems.
    Users typically don't modify the graph directly; the engine manages it.
    
    **Usage Pattern:**
    ```python
    graph = IdeaDag(root_title="Research topic", root_details={"mandate": "..."})
    root_id = graph.root_id()
    node = graph.get_node(root_id)
    node_count = graph.node_count()
    ```
    
    **What It Stores:**
    - Node hierarchy with parent-child relationships
    - Node status (pending, active, done, blocked, failed)
    - Action results and merged data
    - Blocked sites and execution history
    
    **Key Methods:**
    - `root_id()`: Get the root node's unique ID
    - `get_node(node_id)`: Retrieve a node by ID
    - `node_count()`: Total number of nodes in the graph
    - `to_dict()`: Serialize graph to dictionary for storage/analysis
    
    **Important Behavior:**
    - Each node has a unique UUID
    - Nodes track their parent(s) and children
    - Graph automatically manages node lifecycle
    - Can be serialized to JSON for persistence
    """
    def __init__(self, root_title: str, root_details: Optional[Dict[str, Any]] = None):
        self._nodes: Dict[str, IdeaNode] = {}
        self._root_id = self._new_id()
        self._executed_actions: Dict[str, str] = {}
        self._blocked_sites: Dict[str, str] = {}
        root = IdeaNode(
            node_id=self._root_id,
            title=root_title,
            details=dict(root_details or {}),
            parent_id=None,
            parent_ids=[],
            status=IdeaNodeStatus.ACTIVE,
            children=[],
        )
        self._nodes[self._root_id] = root

    def _new_id(self) -> str:
        """
        Create a unique node identifier.
        :returns: Node identifier.
        """
        return str(uuid.uuid4())

    def root_id(self) -> str:
        """
        Return the root node identifier.
        :returns: Root node id.
        """
        return self._root_id

    def node_count(self) -> int:
        """
        Return the number of nodes in the graph.
        :returns: Node count.
        """
        return len(self._nodes)

    def get_node(self, node_id: str) -> Optional[IdeaNode]:
        """
        Retrieve a node by id.
        :param node_id: Node identifier.
        :returns: IdeaNode or None.
        """
        return self._nodes.get(node_id)

    def depth(self, node_id: str) -> int:
        """
        Return the depth of a node from the root.
        :param node_id: Node identifier.
        :returns: Depth value.
        """
        depth = 0
        current = self._nodes.get(node_id)
        seen = set()
        while current and current.node_id not in seen:
            seen.add(current.node_id)
            parents = current.parent_ids or ([] if current.parent_id is None else [current.parent_id])
            if not parents:
                break
            depth += 1
            current = self._nodes.get(parents[0])
        return depth

    def add_child(
        self,
        parent_id: str,
        title: str,
        details: Optional[Dict[str, Any]] = None,
        status: Union[IdeaNodeStatus, str] = IdeaNodeStatus.PENDING,
        score: Optional[float] = None,
        memo_key: Optional[str] = None,
    ) -> IdeaNode:
        """
        Add a child node under a parent.
        :param parent_id: Parent node identifier.
        :param title: Title for the child.
        :param details: Optional details payload.
        :param status: Status enum or string.
        :param score: Optional evaluation score.
        :param memo_key: Optional memoization key.
        :returns: Newly created IdeaNode.
        """
        parent = self._nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Unknown parent_id: {parent_id}")
        node_id = self._new_id()
        node = IdeaNode(
            node_id=node_id,
            title=title,
            details=dict(details or {}),
            parent_id=parent_id,
            parent_ids=[parent_id],
            status=self._coerce_status(status),
            children=[],
            score=score,
            memo_key=memo_key,
        )
        self._nodes[node_id] = node
        parent.children.append(node_id)
        return node

    def merge_nodes(
        self,
        parent_ids: List[str],
        title: str,
        details: Optional[Dict[str, Any]] = None,
        status: Union[IdeaNodeStatus, str] = IdeaNodeStatus.PENDING,
        score: Optional[float] = None,
        memo_key: Optional[str] = None,
    ) -> IdeaNode:
        """
        Merge multiple parents into a new node.
        :param parent_ids: Parent node identifiers.
        :param title: Title for the merged node.
        :param details: Optional details payload.
        :param status: Status enum or string.
        :param score: Optional evaluation score.
        :param memo_key: Optional memoization key.
        :returns: Newly created IdeaNode.
        """
        if not parent_ids:
            raise ValueError("parent_ids required")
        missing = [pid for pid in parent_ids if pid not in self._nodes]
        if missing:
            raise ValueError(f"Unknown parent_ids: {missing}")
        node_id = self._new_id()
        node = IdeaNode(
            node_id=node_id,
            title=title,
            details=dict(details or {}),
            parent_id=None,
            parent_ids=list(parent_ids),
            status=self._coerce_status(status),
            children=[],
            score=score,
            memo_key=memo_key,
        )
        self._nodes[node_id] = node
        for parent_id in parent_ids:
            parent = self._nodes.get(parent_id)
            if parent:
                parent.children.append(node_id)
        return node

    def expand(
        self,
        parent_id: str,
        ideas: List[Dict[str, Any]],
    ) -> List[IdeaNode]:
        """
        Expand a node into multiple candidate child ideas.
        :param parent_id: Parent node identifier.
        :param ideas: List of idea dicts with title/details/score/memo_key.
        :returns: List of created nodes.
        """
        created: List[IdeaNode] = []
        for idea in ideas:
            created.append(
                self.add_child(
                    parent_id=parent_id,
                    title=str(idea.get("title", "")),
                    details=dict(idea.get("details") or {}),
                    status=idea.get("status", IdeaNodeStatus.PENDING),
                    score=idea.get("score"),
                    memo_key=idea.get("memo_key"),
                )
            )
        return created

    def evaluate(
        self,
        node_id: str,
        score: float,
        status: Optional[Union[IdeaNodeStatus, str]] = None,
    ) -> None:
        """
        Evaluate a node by assigning a score and optional status.
        :param node_id: Node identifier.
        :param score: Evaluation score.
        :param status: Optional status update.
        :returns: None
        """
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Unknown node_id: {node_id}")
        node.score = score
        if status is not None:
            node.status = self._coerce_status(status)

    def select_best_child(
        self,
        parent_id: str,
        require_score: bool = True,
    ) -> Optional[IdeaNode]:
        """
        Select the best child node based on score.
        :param parent_id: Parent node identifier.
        :param require_score: Require scored nodes.
        :returns: Best child node or None.
        """
        parent = self._nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Unknown parent_id: {parent_id}")
        candidates = [self._nodes[child_id] for child_id in parent.children if child_id in self._nodes]
        if require_score:
            candidates = [node for node in candidates if node.score is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda node: node.score or float("-inf"))

    def leaf_nodes(self, start_id: Optional[str] = None) -> List[IdeaNode]:
        """
        Return leaf nodes from a start node.
        :param start_id: Optional start node id.
        :returns: List of leaf nodes.
        """
        leaves: List[IdeaNode] = []
        for node in self.iter_depth_first(start_id):
            if node.is_leaf():
                leaves.append(node)
        return leaves

    def merge_details(
        self,
        node_id: str,
        child_ids: Optional[List[str]] = None,
        merge_key: str = "merged",
    ) -> None:
        """
        Merge child details into a parent node.
        :param node_id: Node identifier.
        :param child_ids: Optional child list override.
        :param merge_key: Destination key for merged list.
        :returns: None
        """
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Unknown node_id: {node_id}")
        ids = child_ids if child_ids is not None else node.children
        merged = []
        for child_id in ids:
            child = self._nodes.get(child_id)
            if child is None:
                continue
            merged.append(
                {
                    "node_id": child.node_id,
                    "title": child.title,
                    "details": child.details,
                    "score": child.score,
                    "status": child.status.value,
                }
            )
        node.details[merge_key] = merged

    def update_status(self, node_id: str, status: Union[IdeaNodeStatus, str]) -> None:
        """
        Update the status of a node.
        :param node_id: Node identifier.
        :param status: New status value.
        :returns: None
        """
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Unknown node_id: {node_id}")
        node.status = self._coerce_status(status)

    @staticmethod
    def _sanitize_for_storage(obj: Any) -> Any:
        """
        Recursively sanitize data to ensure JSON serializability.
        Converts all non-serializable types to strings or removes them.
        :param obj: Object to sanitize.
        :returns: Sanitized object.
        """
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): IdeaDag._sanitize_for_storage(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [IdeaDag._sanitize_for_storage(item) for item in obj]
        return str(obj)

    def update_details(self, node_id: str, updates: Dict[str, Any]) -> None:
        """
        Merge detail updates into a node.
        All updates are sanitized to ensure JSON serializability.
        :param node_id: Node identifier.
        :param updates: Dictionary of updates.
        :returns: None
        """
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Unknown node_id: {node_id}")
        sanitized = self._sanitize_for_storage(updates or {})
        node.details.update(sanitized)

    def set_title(self, node_id: str, title: str) -> None:
        """
        Update the node title.
        :param node_id: Node identifier.
        :param title: New title.
        :returns: None
        """
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Unknown node_id: {node_id}")
        node.title = title

    def path_to_root(self, node_id: str) -> List[IdeaNode]:
        """
        Return the path from a node to the root.
        :param node_id: Node identifier.
        :returns: List of nodes from node to root.
        """
        path: List[IdeaNode] = []
        current = self._nodes.get(node_id)
        seen = set()
        while current and current.node_id not in seen:
            path.append(current)
            seen.add(current.node_id)
            parents = current.parent_ids or ([] if current.parent_id is None else [current.parent_id])
            if not parents:
                break
            current = self._nodes.get(parents[0])
        return path

    def iter_depth_first(self, start_id: Optional[str] = None) -> Iterable[IdeaNode]:
        """
        Iterate nodes depth-first.
        :param start_id: Optional start node id.
        :returns: Iterator of nodes.
        """
        start = start_id or self._root_id
        stack = [start]
        while stack:
            node_id = stack.pop()
            node = self._nodes.get(node_id)
            if not node:
                continue
            yield node
            for child_id in reversed(node.children):
                stack.append(child_id)

    def iter_breadth_first(self, start_id: Optional[str] = None) -> Iterable[IdeaNode]:
        """
        Iterate nodes breadth-first.
        :param start_id: Optional start node id.
        :returns: Iterator of nodes.
        """
        start = start_id or self._root_id
        queue = [start]
        idx = 0
        while idx < len(queue):
            node_id = queue[idx]
            idx += 1
            node = self._nodes.get(node_id)
            if not node:
                continue
            yield node
            queue.extend(node.children)

    def find_by_status(self, status: Union[IdeaNodeStatus, str]) -> List[IdeaNode]:
        """
        Return all nodes matching a status.
        :param status: Status value.
        :returns: List of matching nodes.
        """
        expected = self._coerce_status(status)
        return [node for node in self._nodes.values() if node.status == expected]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize graph to a dictionary.
        :returns: Serialized payload.
        """
        return {
            "root_id": self._root_id,
            "nodes": {
                node_id: {
                    "node_id": node.node_id,
                    "title": node.title,
                    "details": node.details,
                    "parent_id": node.parent_id,
                    "parent_ids": list(node.parent_ids),
                    "status": node.status.value,
                    "children": list(node.children),
                    "score": node.score,
                    "memo_key": node.memo_key,
                }
                for node_id, node in self._nodes.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> IdeaDag:
        """
        Deserialize graph from a dictionary.
        :param payload: Serialized payload.
        :returns: IdeaDag instance.
        """
        root_id = payload.get("root_id")
        nodes = payload.get("nodes", {})
        if not root_id or root_id not in nodes:
            raise ValueError("Invalid payload: missing root node")
        root = nodes[root_id]
        graph = cls(root_title=root.get("title", "root"), root_details=root.get("details"))
        graph._root_id = root_id
        graph._nodes = {}
        graph._executed_actions = dict(payload.get("executed_actions", {}))
        graph._blocked_sites = dict(payload.get("blocked_sites", {}))
        
        for node_id, data in nodes.items():
            node = IdeaNode(
                node_id=data.get("node_id", node_id),
                title=data.get("title", ""),
                details=dict(data.get("details") or {}),
                parent_id=data.get("parent_id"),
                parent_ids=list(data.get("parent_ids") or []),
                status=graph._coerce_status(data.get("status", IdeaNodeStatus.PENDING)),
                children=list(data.get("children") or []),
                score=data.get("score"),
                memo_key=data.get("memo_key"),
            )
            graph._nodes[node_id] = node
            
            if node.status == IdeaNodeStatus.DONE:
                action_type = node.details.get("action")
                if action_type:
                    action_key = graph._build_action_key(str(action_type), node.details)
                    if action_key:
                        graph._executed_actions[action_key] = node_id
        
        return graph

    @staticmethod
    def _coerce_status(status: Union[IdeaNodeStatus, str]) -> IdeaNodeStatus:
        """
        Convert status input to IdeaNodeStatus.
        :param status: Status value.
        :returns: IdeaNodeStatus.
        """
        if isinstance(status, IdeaNodeStatus):
            return status
        try:
            return IdeaNodeStatus(str(status))
        except ValueError:
            raise ValueError(f"Unknown status: {status}")
    
    def _build_action_key(self, action_type: str, details: Dict[str, Any]) -> Optional[str]:
        """
        Build a deduplication key for an action.
        :param action_type: Action type string.
        :param details: Node details dict.
        :returns: Action key or None.
        """
        if action_type == "visit":
            url = details.get("url") or details.get("link")
            if url:
                return f"visit:{url.lower().strip()}"
        elif action_type == "search":
            query = details.get("query") or details.get("prompt")
            if query:
                return f"search:{query.lower().strip()}"
        return None
    
    def has_executed_action(self, action_type: str, details: Dict[str, Any]) -> Optional[str]:
        """
        Check if an action has already been executed.
        :param action_type: Action type string.
        :param details: Node details dict.
        :returns: Node ID that executed this action, or None.
        """
        action_key = self._build_action_key(action_type, details)
        if not action_key:
            return None
        return self._executed_actions.get(action_key)
    
    def mark_action_executed(self, node_id: str, action_type: str, details: Dict[str, Any]) -> None:
        """
        Mark an action as executed.
        :param node_id: Node identifier.
        :param action_type: Action type string.
        :param details: Node details dict.
        :returns: None.
        """
        action_key = self._build_action_key(action_type, details)
        if action_key:
            self._executed_actions[action_key] = node_id
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """
        Extract domain from URL for blocking detection.
        :param url: URL string.
        :returns: Domain or None.
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if ':' in domain:
                domain = domain.split(':')[0]
            return domain
        except Exception:
            return None
    
    def is_site_blocked(self, url: str) -> Optional[str]:
        """
        Check if a site is blocked.
        :param url: URL to check.
        :returns: Block reason or None.
        """
        domain = self._extract_domain(url)
        if domain:
            return self._blocked_sites.get(domain)
        return None
    
    def mark_site_blocked(self, url: str, reason: str) -> None:
        """
        Mark a site as blocked.
        :param url: URL that was blocked.
        :param reason: Reason for blocking.
        :returns: None.
        """
        domain = self._extract_domain(url)
        if domain:
            self._blocked_sites[domain] = reason
    
    def build_event_log_table(self, node_id: str, max_events: int = 20) -> str:
        """
        Build a concise event log table of actions and outcomes up to a node.
        This provides branch-aware context: each branch only sees events in its path.
        Branch-to-branch communication happens via vector database.
        
        :param node_id: Node identifier to build log up to.
        :param max_events: Maximum number of events to include.
        :returns: Formatted event log table string.
        """
        path = self.path_to_root(node_id)
        if not path:
            return "No events in path."
        
        path = list(reversed(path))
        
        events = []
        for node in path:
            if node.node_id == self._root_id and not node.details.get("action"):
                continue
            
            action_type = node.details.get("action", "")
            status = node.status.value
            result = node.details.get("action_result")
            
            # Determine success status
            if result and isinstance(result, dict):
                success = result.get("success", False)
            elif status == IdeaNodeStatus.DONE.value:
                success = True
            elif status == IdeaNodeStatus.FAILED.value:
                success = False
            else:
                success = None
            
            # Build event summary
            event_summary = []
            if action_type:
                if action_type == "visit":
                    url = result.get("url") if result and isinstance(result, dict) else node.details.get("url") or node.details.get("link")
                    if url:
                        event_summary.append(f"URL: {url[:60]}")
                elif action_type == "search":
                    query = result.get("query") if result and isinstance(result, dict) else node.details.get("query") or node.details.get("prompt")
                    if query:
                        event_summary.append(f"Query: {query[:60]}")
                    if result and isinstance(result, dict):
                        results_count = len(result.get("results", []))
                        if results_count > 0:
                            event_summary.append(f"Found {results_count} results")
                elif action_type == "think":
                    event_summary.append("Internal reasoning")
            
            # Add error if failed
            if success is False:
                error = None
                if result and isinstance(result, dict):
                    error = result.get("error", "")
                if not error:
                    error = node.details.get("action_error", "")
                if error:
                    event_summary.append(f"Error: {error[:80]}")
            
            # Build event row
            title_short = node.title[:50] if len(node.title) > 50 else node.title
            action_display = action_type if action_type else "planning"
            status_display = "✓" if success is True else "✗" if success is False else "○"
            summary = " | ".join(event_summary) if event_summary else ""
            
            events.append({
                "status": status_display,
                "action": action_display,
                "title": title_short,
                "summary": summary,
            })
        
        # Limit events
        events = events[-max_events:] if len(events) > max_events else events
        
        if not events:
            return "No events in path."
        
        # Format as table
        lines = ["Event Log (branch context):"]
        lines.append("=" * 80)
        lines.append(f"{'Status':<8} {'Action':<12} {'Title':<50}")
        lines.append("-" * 80)
        
        for event in events:
            status = event["status"]
            action = event["action"]
            title = event["title"]
            summary = event["summary"]
            
            # Main row
            lines.append(f"{status:<8} {action:<12} {title:<50}")
            
            # Summary row if present
            if summary:
                lines.append(f"{'':<8} {'':<12}   └─ {summary}")
        
        lines.append("=" * 80)
        lines.append(f"Note: This branch focuses on its sub-problem. Other branches handle")
        lines.append(f"      their sub-problems independently. Cross-branch data sharing")
        lines.append(f"      happens via vector database when relevant.")
        
        return "\n".join(lines)