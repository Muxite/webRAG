"""
Expansion-Merge Pair Abstraction

Every branch in the DAG follows the pattern:
  Expansion Node → [Layers of nodes] → Merge Node

This module provides abstractions to make this pattern explicit and ensure
merges always progress toward completion (toward the root).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from agent.app.idea_dag import IdeaDag, IdeaNode, IdeaNodeStatus
from agent.app.idea_policies.base import DetailKey, IdeaActionType


class BranchPair:
    """
    Represents an expansion-merge pair.
    
    Structure:
      expansion_node (creates sub-problems)
        ↓
      [intermediate layers - can have their own expansion-merge pairs]
        ↓
      merge_node (combines results toward completion)
    
    :param expansion_node_id: Node that expands (breaks into sub-problems).
    :param merge_node_id: Node that merges (combines results).
    :param graph: IdeaDag instance.
    :returns: BranchPair instance.
    """
    
    def __init__(self, expansion_node_id: str, merge_node_id: Optional[str], graph: IdeaDag):
        self.expansion_node_id = expansion_node_id
        self.merge_node_id = merge_node_id
        self.graph = graph
    
    @property
    def expansion_node(self) -> Optional[IdeaNode]:
        """
        Get the expansion node.
        :returns: Expansion node or None.
        """
        return self.graph.get_node(self.expansion_node_id)
    
    @property
    def merge_node(self) -> Optional[IdeaNode]:
        """
        Get the merge node.
        :returns: Merge node or None.
        """
        if not self.merge_node_id:
            return None
        return self.graph.get_node(self.merge_node_id)
    
    def is_complete(self) -> bool:
        """
        Check if the pair is complete (merge node exists and is done).
        :returns: True if complete.
        """
        if not self.merge_node_id:
            return False
        merge_node = self.merge_node
        if not merge_node:
            return False
        return merge_node.status == IdeaNodeStatus.DONE
    
    def needs_expansion(self) -> bool:
        """
        Check if expansion node needs to expand (has no children or should decompose).
        :returns: True if needs expansion.
        """
        expansion_node = self.expansion_node
        if not expansion_node:
            return False
        
        # Already has an action - it's a leaf, not an expansion node
        if expansion_node.details.get(DetailKey.ACTION.value):
            return False
        
        # Has no children - needs expansion
        if not expansion_node.children:
            return True
        
        # Has children but they're all complete - ready for merge
        if all(
            (child := self.graph.get_node(child_id)) and
            child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.BLOCKED, IdeaNodeStatus.SKIPPED)
            for child_id in expansion_node.children
        ):
            return False
        
        return False
    
    def needs_merge(self) -> bool:
        """
        Check if children are ready to merge.
        :returns: True if ready to merge.
        """
        expansion_node = self.expansion_node
        if not expansion_node or not expansion_node.children:
            return False
        
        # Check if all children are complete
        all_complete = all(
            (child := self.graph.get_node(child_id)) and
            child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.BLOCKED, IdeaNodeStatus.SKIPPED)
            for child_id in expansion_node.children
        )
        
        return all_complete and not self.merge_node_id
    
    def get_intermediate_nodes(self) -> List[str]:
        """
        Get all nodes between expansion and merge (the layers).
        :returns: List of intermediate node IDs.
        """
        expansion_node = self.expansion_node
        if not expansion_node or not expansion_node.children:
            return []
        
        intermediate = []
        for child_id in expansion_node.children:
            child = self.graph.get_node(child_id)
            if child and child.details.get(DetailKey.ACTION.value) != IdeaActionType.MERGE.value:
                intermediate.append(child_id)
        
        return intermediate


def find_branch_pair(graph: IdeaDag, node_id: str) -> Optional[BranchPair]:
    """
    Find the expansion-merge pair for a given node.
    
    If node is an expansion node (has children), finds its merge node.
    If node is a merge node, finds its expansion node (parent).
    If node is intermediate, finds the pair it belongs to.
    
    :param graph: IdeaDag instance.
    :param node_id: Node identifier.
    :returns: BranchPair or None.
    """
    node = graph.get_node(node_id)
    if not node:
        return None
    
    # If this is a merge node, find its expansion node (parent)
    if node.details.get(DetailKey.ACTION.value) == IdeaActionType.MERGE.value:
        # Merge nodes have parent_ids pointing to the children they merge
        # The expansion node is the parent of those children
        if node.parent_ids:
            first_child_id = node.parent_ids[0]
            first_child = graph.get_node(first_child_id)
            if first_child and first_child.parent_id:
                return BranchPair(expansion_node_id=first_child.parent_id, merge_node_id=node_id, graph=graph)
        return None
    
    # If this node has children, it's an expansion node
    # Check if it has a merge node as a child
    if node.children:
        for child_id in node.children:
            child = graph.get_node(child_id)
            if child and child.details.get(DetailKey.ACTION.value) == IdeaActionType.MERGE.value:
                return BranchPair(expansion_node_id=node_id, merge_node_id=child_id, graph=graph)
        
        # No merge node yet, but this is an expansion node
        return BranchPair(expansion_node_id=node_id, merge_node_id=None, graph=graph)
    
    # This is an intermediate/leaf node - find its expansion node (parent)
    if node.parent_id:
        parent = graph.get_node(node.parent_id)
        if parent:
            # Check if parent has a merge node
            if parent.children:
                for child_id in parent.children:
                    child = graph.get_node(child_id)
                    if child and child.details.get(DetailKey.ACTION.value) == IdeaActionType.MERGE.value:
                        return BranchPair(expansion_node_id=node.parent_id, merge_node_id=child_id, graph=graph)
            
            # Parent is expansion node, no merge node yet
            return BranchPair(expansion_node_id=node.parent_id, merge_node_id=None, graph=graph)
    
    return None


def get_completion_path(graph: IdeaDag, node_id: str) -> List[str]:
    """
    Get the path from a node toward completion (toward root).
    
    This represents the merge chain that will eventually reach the root.
    
    :param graph: IdeaDag instance.
    :param node_id: Starting node identifier.
    :returns: List of node IDs from node to root (for merging).
    """
    path = []
    current_id = node_id
    
    while current_id:
        node = graph.get_node(current_id)
        if not node:
            break
        
        path.append(current_id)
        
        # Move toward root
        if node.parent_id:
            current_id = node.parent_id
        else:
            break
    
    return path
