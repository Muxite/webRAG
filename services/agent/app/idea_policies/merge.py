from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.app.idea_dag import IdeaDag, IdeaNodeStatus
from agent.app.idea_policies.base import MergePolicy, DetailKey, IdeaActionType


class SimpleMergePolicy(MergePolicy):
    """
    Merge child action results into the parent details with recursive merging support.
    :param settings: Settings dictionary.
    :returns: SimpleMergePolicy instance.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings=settings)
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _sanitize_data(obj: Any) -> Any:
        """
        Recursively sanitize data to ensure JSON serializability.
        :param obj: Object to sanitize.
        :returns: Sanitized object.
        """
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): SimpleMergePolicy._sanitize_data(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [SimpleMergePolicy._sanitize_data(item) for item in obj]
        return str(obj)

    def are_children_ready_to_merge(self, graph: IdeaDag, node_id: str) -> bool:
        """
        Check if all children are complete and ready to merge.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: True if all children are done/failed/blocked/skipped.
        """
        node = graph.get_node(node_id)
        if not node or not node.children:
            return False
        
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            # Check if child is a merge node that's already done
            if child.details.get(DetailKey.ACTION.value) == IdeaActionType.MERGE.value:
                if child.status == IdeaNodeStatus.DONE:
                    continue
                # If merge node failed, still consider it "ready" to merge
                if child.status in (IdeaNodeStatus.FAILED, IdeaNodeStatus.BLOCKED):
                    continue
            # For regular action nodes, check if they're complete
            if child.status not in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.BLOCKED, IdeaNodeStatus.SKIPPED):
                return False
        
        return True

    def should_create_merge_node(self, graph: IdeaDag, node_id: str) -> bool:
        """
        Determine if a merge node should be created for this parent.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: True if merge node should be created.
        """
        if not self.settings.get("enable_recursive_merge", True):
            return False
        
        node = graph.get_node(node_id)
        if not node or len(node.children) < 2:
            return False
        
        # Check if merge node already exists
        for child_id in node.children:
            child = graph.get_node(child_id)
            if child and child.details.get(DetailKey.ACTION.value) == IdeaActionType.MERGE.value:
                return False
        
        # Check if all children are ready
        return self.are_children_ready_to_merge(graph, node_id)

    def create_merge_node(self, graph: IdeaDag, parent_id: str) -> Optional[str]:
        """
        Create a merge node for a parent with completed children.
        :param graph: IdeaDag instance.
        :param parent_id: Parent node identifier.
        :returns: Created merge node ID or None.
        """
        parent = graph.get_node(parent_id)
        if not parent:
            return None
        
        # First, merge the children results into parent
        self.merge(graph, parent_id)
        
        # Create merge node
        merge_title = f"Merge: {parent.title}"
        merge_node = graph.add_child(
            parent_id=parent_id,
            title=merge_title,
            details={
                DetailKey.ACTION.value: IdeaActionType.MERGE.value,
            },
            status=IdeaNodeStatus.PENDING,
            score=None,
        )
        
        self._logger.info(f"Created merge node {merge_node.node_id} for parent {parent_id}")
        return merge_node.node_id

    def merge(self, graph: IdeaDag, node_id: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Merge child action results into a parent summary payload.
        Supports recursive merging up the tree.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :param recursive: If True, recursively merge parent's parent.
        :returns: Merge payload.
        """
        node = graph.get_node(node_id)
        if not node:
            return {}
        
        merged: List[Dict[str, Any]] = []
        success_count = 0
        failed_count = 0
        blocked_count = 0
        skipped_count = 0
        
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            
            # For merge nodes, use their synthesized result
            if child.details.get(DetailKey.ACTION.value) == IdeaActionType.MERGE.value:
                result = child.details.get(DetailKey.ACTION_RESULT.value)
                if result and result.get("success"):
                    # Extract synthesized content from merge result
                    synthesized = result.get("synthesized", {})
                    merged.append(
                        {
                            "node_id": child.node_id,
                            "title": child.title,
                            "status": child.status.value,
                            "score": child.score,
                            "result": synthesized,
                            "is_merge": True,
                        }
                    )
                    if child.status == IdeaNodeStatus.DONE:
                        success_count += 1
                    continue
            
            # For regular action nodes, use their action result
            result = child.details.get(DetailKey.ACTION_RESULT.value)
            if result is None:
                result = child.details.get(DetailKey.ACTION_RESULTS.value)
            
            # Track status counts
            if child.status == IdeaNodeStatus.DONE:
                success_count += 1
            elif child.status == IdeaNodeStatus.FAILED:
                failed_count += 1
            elif child.status == IdeaNodeStatus.BLOCKED:
                blocked_count += 1
            elif child.status == IdeaNodeStatus.SKIPPED:
                skipped_count += 1
            
            # Sanitize result to ensure JSON serializability
            sanitized_result = self._sanitize_data(result) if result else None
            
            merged.append(
                {
                    "node_id": child.node_id,
                    "title": child.title,
                    "status": child.status.value,
                    "score": child.score,
                    "evaluation": child.details.get(DetailKey.EVALUATION.value),
                    "result": sanitized_result,
                    "is_merge": False,
                }
            )
        
        # Store merge summary with failure tracking
        merge_summary = {
            "total": len(merged),
            "success": success_count,
            "failed": failed_count,
            "blocked": blocked_count,
            "skipped": skipped_count,
        }
        node.details[DetailKey.MERGED_RESULTS.value] = self._sanitize_data(merged)
        node.details[DetailKey.MERGE_SUMMARY.value] = merge_summary
        
        # Propagate failure state to parent if all children failed
        if failed_count > 0 and success_count == 0 and blocked_count == 0:
            if node.status == IdeaNodeStatus.ACTIVE:
                node.status = IdeaNodeStatus.FAILED
                node.details[DetailKey.MERGE_FAILURE.value] = f"All {failed_count} children failed"
        
        # Recursive merge: merge parent's parent if applicable
        if recursive and node.parent_id:
            parent = graph.get_node(node.parent_id)
            if parent:
                self.merge(graph, node.parent_id, recursive=True)
        
        return {"merged": merged, "summary": merge_summary}
