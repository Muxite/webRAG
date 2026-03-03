from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag

from agent.app.idea_policies.base import MergePolicy, DetailKey, IdeaActionType, IdeaNodeStatus


class SimpleMergePolicy(MergePolicy):
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings=settings)
        self._logger = logging.getLogger(__name__)

    @staticmethod
    def _sanitize_data(obj: Any) -> Any:
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
        node = graph.get_node(node_id)
        if not node or not node.children:
            return False
        
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            # Skip already-finished merge nodes
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            if NodeDetailsExtractor.is_merge_action(child.details):
                if child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                    continue
                # Merge node still running -> not ready
                return False
            # For regular action nodes: BLOCKED means still retrying -> not ready
            if child.status not in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                return False
        
        return True

    def should_create_merge_node(self, graph: IdeaDag, node_id: str) -> bool:
        if not self.settings.get("enable_recursive_merge", True):
            return False
        
        node = graph.get_node(node_id)
        if not node or len(node.children) < 2:
            return False
        
        # Check if merge node already exists
        for child_id in node.children:
            child = graph.get_node(child_id)
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            if child and NodeDetailsExtractor.is_merge_action(child.details):
                # If existing merge node indicates skip, don't create another
                if child.details.get("merge_should_skip", False):
                    return False
                return False
        
        # Check if all children are ready
        return self.are_children_ready_to_merge(graph, node_id)

    def create_merge_node(self, graph: IdeaDag, parent_id: str) -> Optional[str]:
        parent = graph.get_node(parent_id)
        if not parent:
            return None
        
        # First, merge the children results into parent (populates parent.details.merged_results)
        self.merge(graph, parent_id)
        
        # Create merge node with parent justification
        merge_title = f"Merge: {parent.title}"
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor
        parent_justification = NodeDetailsExtractor.get_justification(parent.details)
        merge_details = {
            DetailKey.ACTION.value: IdeaActionType.MERGE.value,
        }
        if parent_justification:
            merge_details[DetailKey.PARENT_JUSTIFICATION.value] = parent_justification
            merge_details[DetailKey.WHY_THIS_NODE.value] = f"Merge results from children to synthesize findings for: {parent.title}. {parent_justification}"
        
        # Copy merged_results from parent into the merge node so MergeLeafAction can read them
        parent_merged = parent.details.get(DetailKey.MERGED_RESULTS.value)
        if parent_merged:
            merge_details[DetailKey.MERGED_RESULTS.value] = parent_merged
        parent_merge_summary = parent.details.get(DetailKey.MERGE_SUMMARY.value)
        if parent_merge_summary:
            merge_details[DetailKey.MERGE_SUMMARY.value] = parent_merge_summary
        
        merge_node = graph.add_child(
            parent_id=parent_id,
            title=merge_title,
            details=merge_details,
            status=IdeaNodeStatus.PENDING,
            score=None,
        )
        
        self._logger.info(f"Created merge node {merge_node.node_id} for parent {parent_id}")
        return merge_node.node_id

    def merge(self, graph: IdeaDag, node_id: str, recursive: bool = True) -> Dict[str, Any]:
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
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            if NodeDetailsExtractor.is_merge_action(child.details):
                result = child.details.get(DetailKey.ACTION_RESULT.value)
                from agent.app.idea_policies.action_constants import ActionResultKey
                from agent.app.idea_policies.action_constants import ActionResultExtractor
                if result and ActionResultExtractor.is_success(result):
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
        
        # Validate goal achievement if this is a merge node
        goal_achieved = self._validate_goal_achievement(graph, node, merged)
        node.details[DetailKey.GOAL_ACHIEVED.value] = goal_achieved
        
        if not goal_achieved and success_count > 0:
            self._logger.warning(f"[MERGE] Goal not fully achieved for node {node_id} - may need additional work")
        
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
        
        return {"merged": merged, "summary": merge_summary, "goal_achieved": goal_achieved}
    
    def _validate_goal_achievement(self, graph: IdeaDag, node: IdeaNode, merged_results: List[Dict[str, Any]]) -> bool:
        original_goal = node.details.get(DetailKey.GOAL.value) or node.details.get(DetailKey.ORIGINAL_GOAL.value) or node.details.get(DetailKey.INTENT.value)
        if not original_goal:
            return True
        
        from agent.app.idea_policies.action_constants import ActionResultKey
        
        has_relevant_content = False
        for result_item in merged_results:
            result = result_item.get("result")
            if not isinstance(result, dict):
                continue
            
            content = result.get(ActionResultKey.CONTENT.value) or ""
            query = result.get(ActionResultKey.QUERY.value) or ""
            results = result.get(ActionResultKey.RESULTS.value) or []
            
            if content and original_goal.lower() in content.lower():
                has_relevant_content = True
                break
            
            if query and original_goal.lower() in query.lower():
                has_relevant_content = True
                break
            
            if isinstance(results, list) and len(results) > 0:
                has_relevant_content = True
                break
        
        return has_relevant_content
