from __future__ import annotations

from typing import Any, Dict, Optional

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DecompositionPolicy


class ScoreThresholdDecompositionPolicy(DecompositionPolicy):
    """
    Decompose when score is below a threshold and depth is allowed.
    :param settings: Settings dictionary.
    :returns: ScoreThresholdDecompositionPolicy instance.
    """
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings=settings)

    def should_decompose(self, graph: IdeaDag, node_id: str) -> bool:
        """
        Decide whether a node should be decomposed.
        Don't decompose if node is already a leaf (has action ready).
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :returns: True if decomposition should occur.
        """
        node = graph.get_node(node_id)
        if not node:
            return False
        
        from agent.app.idea_policies.base import DetailKey
        if node.details.get(DetailKey.ACTION.value):
            return False
        
        threshold = float(self.settings.get("decomposition_threshold", 0.5))
        max_depth = int(self.settings.get("max_depth", 6))
        depth = graph.depth(node_id)
        score = node.score if node.score is not None else 0.0
        
        # Prefer decomposition when score is low (problem not well understood)
        # But avoid over-decomposition - prefer fewer, larger steps
        # Only decompose if we're not too deep and the problem needs breaking down
        should_decompose = depth < max_depth and score < threshold
        
        # If node already has children that are complete, don't decompose further
        # Let merging happen instead
        if node.children:
            from agent.app.idea_policies.base import IdeaNodeStatus
            all_children_complete = all(
                (child := graph.get_node(child_id)) and 
                child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.BLOCKED, IdeaNodeStatus.SKIPPED)
                for child_id in node.children
            )
            if all_children_complete:
                return False
        
        return should_decompose
