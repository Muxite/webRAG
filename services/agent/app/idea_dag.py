from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union
import uuid

from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus


@dataclass
class IdeaNode:
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
        return len(self.children) == 0


class IdeaDag:
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
        return str(uuid.uuid4())

    def root_id(self) -> str:
        return self._root_id

    def node_count(self) -> int:
        return len(self._nodes)

    def get_node(self, node_id: str) -> Optional[IdeaNode]:
        return self._nodes.get(node_id)

    def depth(self, node_id: str) -> int:
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
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Unknown node_id: {node_id}")
        node.status = self._coerce_status(status)

    @staticmethod
    def _sanitize_for_storage(obj: Any) -> Any:
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
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Unknown node_id: {node_id}")
        sanitized = self._sanitize_for_storage(updates or {})
        node.details.update(sanitized)

    def set_title(self, node_id: str, title: str) -> None:
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"Unknown node_id: {node_id}")
        node.title = title

    def path_to_root(self, node_id: str) -> List[IdeaNode]:
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
        expected = self._coerce_status(status)
        return [node for node in self._nodes.values() if node.status == expected]

    def to_dict(self) -> Dict[str, Any]:
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
                action_type = node.details.get(DetailKey.ACTION.value)
                if action_type:
                    action_key = graph._build_action_key(str(action_type), node.details)
                    if action_key:
                        graph._executed_actions[action_key] = node_id
        
        return graph

    @staticmethod
    def _coerce_status(status: Union[IdeaNodeStatus, str]) -> IdeaNodeStatus:
        if isinstance(status, IdeaNodeStatus):
            return status
        try:
            return IdeaNodeStatus(str(status))
        except ValueError:
            raise ValueError(f"Unknown status: {status}")
    
    def _build_action_key(self, action_type: str, details: Dict[str, Any]) -> Optional[str]:
        from agent.app.idea_policies.base import IdeaActionType
        if action_type == IdeaActionType.VISIT.value:
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            url = NodeDetailsExtractor.get_url(details)
            if url:
                url_str = url if isinstance(url, str) else str(url)
                return f"visit:{url_str.lower().strip()}"
        elif action_type == IdeaActionType.SEARCH.value:
            query = details.get(DetailKey.QUERY.value) or details.get(DetailKey.PROMPT.value)
            if query:
                query_str = query if isinstance(query, str) else str(query)
                return f"search:{query_str.lower().strip()}"
        return None
    
    def has_executed_action(self, action_type: str, details: Dict[str, Any]) -> Optional[str]:
        action_key = self._build_action_key(action_type, details)
        if not action_key:
            return None
        return self._executed_actions.get(action_key)
    
    def mark_action_executed(self, node_id: str, action_type: str, details: Dict[str, Any]) -> None:
        action_key = self._build_action_key(action_type, details)
        if action_key:
            self._executed_actions[action_key] = node_id
    
    def _extract_domain(self, url: str) -> Optional[str]:
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
        domain = self._extract_domain(url)
        if domain:
            return self._blocked_sites.get(domain)
        return None
    
    def mark_site_blocked(self, url: str, reason: str) -> None:
        domain = self._extract_domain(url)
        if domain:
            self._blocked_sites[domain] = reason
    
    def build_event_log_table(self, node_id: str, max_events: int = 20) -> str:
        path = self.path_to_root(node_id)
        if not path:
            return "No events in path."
        
        path = list(reversed(path))
        
        events = []
        for node in path:
            if node.node_id == self._root_id and not node.details.get(DetailKey.ACTION.value):
                continue
            
            action_type = node.details.get(DetailKey.ACTION.value, "")
            status = node.status.value
            result = node.details.get(DetailKey.ACTION_RESULT.value)
            
            from agent.app.idea_policies.action_constants import ActionResultKey
            from agent.app.idea_policies.base import IdeaActionType
            if result and isinstance(result, dict):
                success = result.get(ActionResultKey.SUCCESS.value, False)
            elif status == IdeaNodeStatus.DONE.value:
                success = True
            elif status == IdeaNodeStatus.FAILED.value:
                success = False
            else:
                success = None
            
            event_summary = []

            justification = (
                node.details.get(DetailKey.JUSTIFICATION.value)
                or node.details.get(DetailKey.WHY_THIS_NODE.value)
                or ""
            )
            if justification:
                event_summary.append(f"Why: {str(justification)[:80]}")

            if action_type:
                if action_type == IdeaActionType.VISIT.value:
                    from agent.app.idea_policies.action_constants import NodeDetailsExtractor
                    url = result.get(ActionResultKey.URL.value) if result and isinstance(result, dict) else NodeDetailsExtractor.get_url(node.details)
                    if url:
                        event_summary.append(f"URL: {url[:60]}")
                    if success and result and isinstance(result, dict):
                        page_title = result.get("page_title", "")
                        content_chars = result.get("content_total_chars", 0)
                        links_count = result.get("links_count", 0)
                        if page_title:
                            event_summary.append(f"Page: {page_title[:50]}")
                        if content_chars:
                            event_summary.append(f"{content_chars} chars, {links_count} links")
                elif action_type == IdeaActionType.SEARCH.value:
                    from agent.app.idea_policies.action_constants import NodeDetailsExtractor
                    query = result.get(ActionResultKey.QUERY.value) if result and isinstance(result, dict) else NodeDetailsExtractor.get_query(node.details)
                    if query:
                        event_summary.append(f"Query: {query[:60]}")
                    if result and isinstance(result, dict):
                        search_results = result.get(ActionResultKey.RESULTS.value, [])
                        results_count = len(search_results) if isinstance(search_results, list) else 0
                        if results_count > 0:
                            event_summary.append(f"Found {results_count} results")
                            top_urls = []
                            for sr in (search_results[:3] if isinstance(search_results, list) else []):
                                if isinstance(sr, dict) and sr.get("url"):
                                    top_urls.append(str(sr["url"])[:60])
                            if top_urls:
                                event_summary.append(f"Top URLs: {', '.join(top_urls)}")
                elif action_type == IdeaActionType.THINK.value:
                    event_summary.append("Internal reasoning")
            
            if success is False:
                error = None
                if result and isinstance(result, dict):
                    error = result.get(ActionResultKey.ERROR.value, "")
                if not error:
                    error = node.details.get(DetailKey.ACTION_ERROR.value, "")
                if error:
                    event_summary.append(f"Error: {error[:80]}")
            
            title_short = node.title[:50] if len(node.title) > 50 else node.title
            action_display = action_type if action_type else "planning"
            status_display = "[OK]" if success is True else "[FAIL]" if success is False else "[-]"
            summary = " | ".join(event_summary) if event_summary else ""
            
            events.append({
                "status": status_display,
                "action": action_display,
                "title": title_short,
                "summary": summary,
            })
        
        events = events[-max_events:] if len(events) > max_events else events
        
        if not events:
            return "No events in path."
        
        lines = ["Ancestor Decision Trail (this branch only):"]
        lines.append("=" * 80)
        lines.append(f"{'Status':<8} {'Action':<12} {'Title':<50}")
        lines.append("-" * 80)
        
        for event in events:
            status = event["status"]
            action = event["action"]
            title = event["title"]
            summary = event["summary"]
            
            lines.append(f"{status:<8} {action:<12} {title:<50}")
            
            if summary:
                lines.append(f"{'':<8} {'':<12}   └─ {summary}")
        
        lines.append("=" * 80)
        lines.append(f"Use the trail above to understand what has been done and decide next steps.")
        lines.append(f"Do NOT repeat completed work. Build on ancestor outcomes.")
        
        return "\n".join(lines)