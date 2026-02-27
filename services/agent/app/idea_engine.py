from __future__ import annotations

from typing import Any, Dict, Optional, List
import hashlib
import logging

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import IdeaNodeStatus
from agent.app.agent_io import AgentIO
from agent.app.idea_dag_settings import load_idea_dag_settings
from agent.app.idea_memory import MemoryManager
from agent.app.idea_policies import (
    DetailKey,
    IdeaActionType,
    LlmExpansionPolicy,
    LlmBatchEvaluationPolicy,
    BestScoreSelectionPolicy,
    ScoreThresholdDecompositionPolicy,
    SimpleMergePolicy,
    LeafActionRegistry,
    NodeDetailsExtractor,
)
from agent.app.idea_finalize import build_final_payload
from agent.app.idea_branch_pair import BranchPair, find_branch_pair, get_completion_path


class IdeaDagEngine:
    """
    Main execution engine for the IdeaDAG agent system.
    
    Orchestrates the complete problem-solving lifecycle: expansion, evaluation, selection,
    action execution, and merging. Users create an engine, then call `step()` repeatedly
    or use `run()` for automatic execution until completion.
    
    **Usage Pattern:**
    ```python
    engine = IdeaDagEngine(io=agent_io, settings=settings, model_name="gpt-5-mini")
    graph = IdeaDag(root_title="Task", root_details={"mandate": "Solve X"})
    current_id = graph.root_id()
    
    # Option 1: Manual step-by-step control
    for step_num in range(max_steps):
        current_id = await engine.step(graph, current_id, step_num)
        if current_id is None:
            break
    
    # Option 2: Automatic execution
    result = await engine.run(mandate="Solve X", max_steps=50)
    ```
    
    **What It Does:**
    - Expands problems into sub-problems using LLM
    - Evaluates candidate solutions
    - Executes actions (search, visit, save, think)
    - Merges results from parallel branches
    - Manages memory and context retrieval
    
    **Key Parameters:**
    - `io`: AgentIO instance providing connectors (LLM, search, HTTP, ChromaDB)
    - `settings`: Optional dict overriding defaults from `idea_dag_settings.json`
    - `model_name`: Optional LLM model override (e.g., "gpt-5-mini", "gpt-4o")
    
    **Returns:**
    - `run()`: Complete result dict with `final_deliverable`, `graph`, `success`
    - `step()`: Next node ID to process, or None if complete/failed
    
    **Important Behavior:**
    - Automatically queries vector DB for context on every node
    - Handles parallel execution when `execute_all_children=true`
    - Creates merge nodes when all children are complete
    - Stops at `max_steps` or when graph reaches `max_total_nodes`
    """
    def __init__(
        self,
        io: AgentIO,
        settings: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        expansion: Optional[LlmExpansionPolicy] = None,
        evaluation: Optional[LlmBatchEvaluationPolicy] = None,
        selection: Optional[BestScoreSelectionPolicy] = None,
        decomposition: Optional[ScoreThresholdDecompositionPolicy] = None,
        merge: Optional[SimpleMergePolicy] = None,
        actions: Optional[LeafActionRegistry] = None,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self.settings = settings or load_idea_dag_settings()
        self.io = io
        self.model_name = model_name
        self.expansion = expansion or LlmExpansionPolicy(io=io, settings=self.settings, model_name=model_name)
        self.evaluation = evaluation or LlmBatchEvaluationPolicy(io=io, settings=self.settings, model_name=model_name)
        self.selection = selection or BestScoreSelectionPolicy(settings=self.settings)
        self.decomposition = decomposition or ScoreThresholdDecompositionPolicy(settings=self.settings)
        self.merge = merge or SimpleMergePolicy(settings=self.settings)
        self.actions = actions or LeafActionRegistry(settings=self.settings)
        self._step_index = 0
        self._memory_manager: Optional[MemoryManager] = None

    async def run(self, mandate: str, max_steps: int = 50) -> Dict[str, Any]:
        """
        Execute the complete problem-solving cycle automatically.
        
        Creates a new DAG, runs steps until completion or max_steps, then generates
        the final output. This is the high-level entry point for automatic execution.
        
        **What It Does:**
        1. Creates a new IdeaDag with the mandate as root
        2. Runs `step()` repeatedly until complete or max_steps reached
        3. Generates final output using merged results
        4. Returns complete result with graph structure
        
        **Parameters:**
        - `mandate`: The task/problem statement to solve
        - `max_steps`: Maximum number of steps before stopping (default: 50)
        
        **Returns:**
        ```python
        {
            "final_deliverable": str|dict,  # Final answer/output
            "action_summary": str,          # Summary of actions taken
            "success": bool,                # Whether execution succeeded
            "graph": dict                   # Complete DAG structure
        }
        ```
        
        **Usage:**
        ```python
        result = await engine.run("Find the capital of France", max_steps=30)
        print(result["final_deliverable"])
        ```
        
        **Important:**
        - Stops early if agent completes before max_steps
        - Creates isolated memory namespace for this mandate
        - Logs DAG visualization at intervals (if enabled)
        - Graph structure is included for analysis/debugging
        """
        mandate_short = mandate.split("\n\nTask Statement")[0] if "\n\nTask Statement" in mandate else mandate[:100]
        self._logger.info(f"[RUN] Starting idea DAG engine with mandate: {mandate_short}..., max_steps={max_steps}")
        namespace = self._memo_namespace(mandate)
        self.settings[DetailKey.MEMO_NAMESPACE.value] = namespace
        # Initialize memory manager
        self._memory_manager = MemoryManager(
            connector_chroma=self.io.connector_chroma,
            namespace=namespace,
        )
        self._current_mandate = mandate  # Store for memory queries
        # Remove "Task Statement" from root title for cleaner logging
        root_title = mandate.split("\n\nTask Statement")[0] if "\n\nTask Statement" in mandate else mandate
        graph = IdeaDag(root_title=root_title, root_details={"mandate": mandate, "memo_namespace": namespace})
        current_id = graph.root_id()
        self._logger.info(f"[RUN] Created graph with root_id={current_id}")
        steps = 0
        while steps < max_steps:
            self._logger.info(f"[RUN] === STEP {steps}/{max_steps} ===")
            current_id = await self.step(graph, current_id, steps)
            steps += 1
            self._step_index = steps
            self._maybe_log_dag(graph, steps)
            
            # Validation checkpoint: After step 1, verify root expanded
            if steps == 1:
                root = graph.get_node(graph.root_id())
                if root and not root.children:
                    self._logger.error(f"[RUN] VALIDATION FAILED: Root has no children after step 1!")
                    self._logger.error(f"[RUN] Root status: {root.status.value}, Root details keys: {list(root.details.keys())}")
                    self._logger.error(f"[RUN] Attempting emergency root expansion...")
                    emergency_result = await self._handle_expansion_node(graph, graph.root_id(), steps, None)
                    if emergency_result and root.children:
                        self._logger.info(f"[RUN] Emergency expansion succeeded: {len(root.children)} children created")
                        current_id = emergency_result
                    else:
                        self._logger.error(f"[RUN] Emergency expansion failed - root still has no children")
            
            # Validation checkpoint: After step 3, verify actions exist
            if steps == 3:
                action_count = sum(1 for n in graph.nodes.values() if NodeDetailsExtractor.get_action(n.details))
                if action_count == 0:
                    self._logger.warning(f"[RUN] VALIDATION WARNING: No actions created after step 3 (total nodes: {graph.node_count()})")
            
            if current_id is None:
                self._logger.warning(f"[RUN] Step {steps} returned None, breaking loop")
                break
        self._logger.info(f"[RUN] Completed {steps} steps, checking for pending nodes before finalizing")
        
        pending_nodes = self._get_pending_executable_nodes(graph)
        if pending_nodes:
            pending_ids = [n.node_id for n in pending_nodes]
            self._logger.warning(f"[RUN] GUARDRAIL: {len(pending_nodes)} nodes still pending execution: {pending_ids[:5]}...")
            self._logger.warning(f"[RUN] Cannot finalize with pending nodes. These nodes need action execution:")
            for node in pending_nodes[:5]:
                action = NodeDetailsExtractor.get_action(node.details)
                self._logger.warning(f"[RUN]   - {node.node_id}: {node.title[:60]}... (action={action}, status={node.status.value})")
        
        final_payload = await build_final_payload(self.io, self.settings, graph, mandate, self.model_name)
        final_payload["graph"] = graph.to_dict()
        final_payload["pending_nodes_count"] = len(pending_nodes) if pending_nodes else 0
        if pending_nodes:
            final_payload["warning"] = f"Finalized with {len(pending_nodes)} pending nodes - execution incomplete"
        self._logger.info(f"[RUN] Final payload created, graph has {graph.node_count()} nodes, {len(pending_nodes) if pending_nodes else 0} pending")
        self._maybe_log_dag(graph, steps, force=True)
        return final_payload

    async def step(self, graph: IdeaDag, current_id: str, step_index: int) -> Optional[str]:
        """
        Execute a single step of the problem-solving process.
        
        This is the core method that advances the DAG one step forward. It handles
        expansion, evaluation, action execution, and merging automatically based on
        the current node's state.
        
        **Execution Flow:**
        1. **Merge Node**: If current node is a merge, execute merge and move to parent
        2. **Leaf Node**: If node has an action, execute it (search/visit/save/think)
        3. **Expansion**: If node has no children, expand into sub-problems
        4. **Evaluation**: If children exist but aren't evaluated, score them
        5. **Selection**: If children are evaluated, select best and execute
        6. **Merge Creation**: If all children are done, create merge node
        
        **Parameters:**
        - `graph`: The IdeaDag instance being processed
        - `current_id`: ID of the node to process in this step
        - `step_index`: Current step number (for logging/debugging)
        
        **Returns:**
        - `str`: Next node ID to process (usually a child, parent, or merge node)
        - `None`: If execution is complete, failed, or max nodes reached
        
        **What Users See:**
        - Node status changes (pending → active → done)
        - Children created during expansion
        - Actions executed (search results, visited URLs)
        - Merge nodes created when branches complete
        - Progress toward root when merging
        
        **Important Behavior:**
        - Automatically handles parallel execution (`execute_all_children=true`)
        - Retries blocked nodes when cooldown expires
        - Stops if graph exceeds `max_total_nodes` (default: 500)
        - Returns None if node not found or execution fails
        """
        self._logger.info(f"[STEP {step_index}] Starting step with current_id={current_id}, node_count={graph.node_count()}")
        if graph.node_count() >= int(self.settings.get("max_total_nodes", 500)):
            self._logger.warning(f"[STEP {step_index}] Max nodes reached, stopping")
            return None
        
        node = graph.get_node(current_id)
        if not node:
            self._logger.warning(f"[STEP {step_index}] Node {current_id} not found")
            return None
        
        self._logger.info(f"[STEP {step_index}] Processing node: {node.title[:80]}... (status={node.status.value}, children={len(node.children)}, action={NodeDetailsExtractor.get_action(node.details)})")
        
        # Special handling for root node: ensure it expands if it has no children
        is_root = current_id == graph.root_id()
        if is_root and not node.children and step_index == 0:
            self._logger.info(f"[STEP {step_index}] ROOT NODE: No children detected, forcing expansion")
            result = await self._handle_expansion_node(graph, current_id, step_index, None)
            if result is None:
                self._logger.error(f"[STEP {step_index}] ROOT EXPANSION FAILED: Expansion returned None")
            elif not node.children:
                self._logger.error(f"[STEP {step_index}] ROOT EXPANSION FAILED: Expansion returned but no children created")
            else:
                self._logger.info(f"[STEP {step_index}] ROOT EXPANSION SUCCESS: Created {len(node.children)} children")
            return result
        
        # 1. Handle merge nodes (execute merge, progress toward root)
        if NodeDetailsExtractor.is_merge_action(node.details):
            self._logger.info(f"[STEP {step_index}] Node is merge node, handling merge")
            return await self._handle_merge_node(graph, current_id, step_index, None)
        
        # 2. Handle leaf nodes (execute actions)
        # Leaf nodes are marked with IS_LEAF=True and have an action
        is_leaf = node.details.get(DetailKey.IS_LEAF.value, False)
        action = NodeDetailsExtractor.get_action(node.details)
        has_action = action and not NodeDetailsExtractor.is_merge_action(node.details)
        if is_leaf or has_action:
            self._logger.info(f"[STEP {step_index}] Node is leaf node (is_leaf={is_leaf}, has_action={has_action}, action={action}), executing action")
            return await self._handle_leaf_node(graph, current_id, step_index, None)
        
        # 3. If node has no children → expand it
        if not node.children:
            self._logger.info(f"[STEP {step_index}] Node has no children, expanding into sub-problems")
            result = await self._handle_expansion_node(graph, current_id, step_index, None)
            if result is None:
                self._logger.warning(f"[STEP {step_index}] Expansion returned None for node {current_id}")
            elif not node.children:
                self._logger.warning(f"[STEP {step_index}] Expansion completed but no children created for node {current_id}")
            else:
                self._logger.info(f"[STEP {step_index}] Expansion created {len(node.children)} children")
            return result
        
        # 4. Check if all children are complete (leaf nodes) → merge
        # Leaf nodes are marked with IS_LEAF=True
        all_children_are_leaves = all(
            (child := graph.get_node(child_id)) and
            (child.details.get(DetailKey.IS_LEAF.value, False) or 
             (NodeDetailsExtractor.get_action(child.details) and not NodeDetailsExtractor.is_merge_action(child.details)))
            for child_id in node.children
        )
        all_children_complete = all(
            (child := graph.get_node(child_id)) and
            (child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED) or
             (child.status == IdeaNodeStatus.BLOCKED and not self._is_action_ready(child, step_index)))
            for child_id in node.children
        )
        if all_children_complete and all_children_are_leaves:
            branch_pair = find_branch_pair(graph, current_id)
            if branch_pair and branch_pair.needs_merge():
                return await self._handle_merge_creation(graph, current_id, step_index, branch_pair)
        
        # 5. Check if any children have data dependencies that aren't satisfied
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            if not self._has_required_data(graph, child):
                required_data = child.details.get(DetailKey.REQUIRES_DATA.value)
                if required_data and isinstance(required_data, dict):
                    source_node_id = required_data.get("source_node_id")
                    if source_node_id:
                        source_node = graph.get_node(source_node_id)
                        if source_node and source_node.status != IdeaNodeStatus.DONE:
                            self._logger.info(f"[STEP {step_index}] Child {child_id} waiting for data from {source_node_id} - executing source first")
                            return source_node_id
        
        # 6. Handle intermediate nodes (evaluate and select)
        return await self._handle_intermediate_node(graph, current_id, step_index, None)
    
    async def _handle_merge_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        """
        Handle a merge node - executes merge and progresses toward completion.
        :param graph: IdeaDag instance.
        :param node_id: Merge node identifier.
        :param step_index: Current step index.
        :param branch_pair: Branch pair context.
        :returns: Next node id (toward root for completion).
        """
        node = graph.get_node(node_id)
        if not node:
            return None
        
        # Execute merge if not done
        if node.details.get(DetailKey.ACTION_RESULT.value) is None:
            if self._is_action_ready(node, step_index):
                result = await self._execute_action(graph, node.parent_id or graph.root_id(), node_id)
                if result is not None:
                    self._handle_action_result(graph, node_id, step_index)
        
        # After merge, progress toward completion (move toward root)
        # This ensures merges always head toward the final stage
        if node.status == IdeaNodeStatus.DONE:
            completion_path = get_completion_path(graph, node_id)
            if len(completion_path) > 1:
                # Move to parent to continue merging up
                next_id = completion_path[1] if len(completion_path) > 1 else completion_path[0]
                self._logger.info(f"[STEP {step_index}] MERGE COMPLETE: Progressing toward completion via {next_id}")
                return next_id
        
        return node_id
    
    async def _handle_leaf_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        """
        Handle a leaf node - executes action.
        After execution, returns to parent to check for merge.
        :param graph: IdeaDag instance.
        :param node_id: Leaf node identifier.
        :param step_index: Current step index.
        :param branch_pair: Branch pair context (unused, kept for compatibility).
        :returns: Next node id (parent if done, or same node if still executing).
        """
        node = graph.get_node(node_id)
        if not node:
            return None
        
        if not self._has_required_data(graph, node):
            self._logger.warning(f"[STEP {step_index}] Node {node_id} missing required data - cannot execute yet")
            return node.parent_id if node.parent_id else None
        
        # Execute action if not done, or retry if blocked and ready
        has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None
        is_blocked_ready = node.status == IdeaNodeStatus.BLOCKED and self._is_action_ready(node, step_index)
        
        if not has_result or is_blocked_ready:
            if self._is_action_ready(node, step_index):
                result = await self._execute_action(graph, node.parent_id or graph.root_id(), node_id)
                if result is not None:
                    self._handle_action_result(graph, node_id, step_index)
        
        # After leaf execution, check if document needs chunking
        if node.status == IdeaNodeStatus.DONE:
            should_chunk = self._should_chunk_document(graph, node)
            if should_chunk:
                self._logger.info(f"[STEP {step_index}] Large document detected - creating chunk sub-problems")
                chunk_nodes = await self._create_chunk_subproblems(graph, node)
                if chunk_nodes:
                    return chunk_nodes[0] if chunk_nodes else node.parent_id
        
        # After leaf execution, return to parent to check for merge
        if node.status == IdeaNodeStatus.DONE and node.parent_id:
            return node.parent_id
        
        return node_id
    
    def _has_required_data(self, graph: IdeaDag, node: IdeaNode) -> bool:
        """
        Check if node has all required data dependencies available.
        :param graph: IdeaDag instance.
        :param node: Node to check.
        :returns: True if all required data is available.
        """
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor
        from agent.app.idea_policies.base import IdeaActionType
        
        requires_data = node.details.get(DetailKey.REQUIRES_DATA.value)
        if not requires_data:
            return True
        
        if not isinstance(requires_data, dict):
            return True
        
        required_type = requires_data.get("type")
        data_source_node_id = requires_data.get("source_node_id")
        
        if not data_source_node_id:
            return True
        
        source_node = graph.get_node(data_source_node_id)
        if not source_node:
            return False
        
        if source_node.status != IdeaNodeStatus.DONE:
            return False
        
        source_result = source_node.details.get(DetailKey.ACTION_RESULT.value)
        if not source_result:
            return False
        
        from agent.app.idea_policies.action_constants import ActionResultKey, ActionResultExtractor
        if not ActionResultExtractor.is_success(source_result):
            return False
        
        if required_type == "urls_from_search":
            results = source_result.get(ActionResultKey.RESULTS.value) or []
            if not results or len(results) == 0:
                return False
            return True
        
        if required_type == "urls_from_visit":
            links = source_result.get(ActionResultKey.LINKS.value) or source_result.get(ActionResultKey.LINKS_FULL.value) or []
            if not links or len(links) == 0:
                return False
            return True
        
        if required_type == "url_from_think":
            extracted_url = source_result.get(ActionResultKey.URL.value) or source_result.get("extracted_url")
            if not extracted_url or not isinstance(extracted_url, str) or not extracted_url.startswith(("http://", "https://")):
                url_from_details = NodeDetailsExtractor.get_url(source_node.details)
                if not url_from_details or not isinstance(url_from_details, str) or not url_from_details.startswith(("http://", "https://")):
                    return False
            return True
        
        if required_type == "chunk_from_visit":
            content_full = source_result.get(ActionResultKey.CONTENT_FULL.value) or ""
            if not content_full:
                return False
            return True
        
        return True
        
        if required_type == "chunk_from_visit" and data_source_node_id:
            source_node = graph.get_node(data_source_node_id)
            if not source_node:
                return False
            
            source_result = source_node.details.get(DetailKey.ACTION_RESULT.value)
            if not source_result:
                return False
            
            from agent.app.idea_policies.action_constants import ActionResultKey, ActionResultExtractor
            if not ActionResultExtractor.is_success(source_result):
                return False
            
            content_full = source_result.get(ActionResultKey.CONTENT_FULL.value) or ""
            if not content_full:
                return False
            
            return True
        
        return True
    
    async def _handle_expansion_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        """
        Handle an expansion node - breaks problem into sub-problems.
        After expansion, continues to evaluation.
        :param graph: IdeaDag instance.
        :param node_id: Expansion node identifier.
        :param step_index: Current step index.
        :param branch_pair: Branch pair context (unused, kept for compatibility).
        :returns: Next node id (same node to continue to evaluation).
        """
        node = graph.get_node(node_id)
        if not node:
            return None
        
        # Retrieve relevant memories from vector DB before expansion (split by type)
        memories = []
        if self._memory_manager:
            justification = NodeDetailsExtractor.get_justification(node.details)
            parent_goal = node.details.get(DetailKey.PARENT_GOAL.value) or ""
            
            query_parts = [node.title]
            if justification:
                query_parts.append(justification[:100])
            if parent_goal:
                query_parts.append(parent_goal[:100])
            if hasattr(self, '_current_mandate') and self._current_mandate:
                query_parts.append(self._current_mandate[:100])
            
            query = " ".join(query_parts)
            
            n_internal = int(self.settings.get("expansion_max_context_nodes", 5))
            n_observations = int(self.settings.get("expansion_max_context_nodes", 5))
            split_memories = await self._memory_manager.retrieve_memories_split(
                query=query,
                node_context={
                    "title": node.title,
                    "action": node.details.get(DetailKey.ACTION.value),
                    "error": node.details.get(DetailKey.ACTION_ERROR.value),
                    "justification": justification,
                },
                n_internal=n_internal,
                n_observations=n_observations,
            )
            memories = split_memories["internal_thoughts"] + split_memories["observations"]
            self._logger.info(
                f"[STEP {step_index}] EXPANSION: Retrieved {len(split_memories['internal_thoughts'])} internal thoughts, "
                f"{len(split_memories['observations'])} observations from vector DB"
            )
            if split_memories["observations"]:
                obs_preview = "\n".join([str(obs.get("content", "") if isinstance(obs, dict) else obs)[:200] for obs in split_memories["observations"][:3]])
                self._logger.info(f"[STEP {step_index}] Observations preview:\n{obs_preview}")
        
        # Expand: break problem into sub-problems
        self._logger.info(f"[STEP {step_index}] EXPANSION: Calling expansion policy for node '{node.title[:60]}...'")
        try:
            candidates = await self.expansion.expand(graph, node_id, memories=memories)
            self._logger.info(f"[STEP {step_index}] EXPANSION: Policy returned {len(candidates) if candidates else 0} candidates")
            if not candidates:
                self._logger.error(f"[STEP {step_index}] EXPANSION FAILED: Expansion policy returned no candidates!")
                self._logger.error(f"[STEP {step_index}] Node details: {list(node.details.keys())}")
                self._logger.error(f"[STEP {step_index}] Node title: {node.title}")
                return None
        except Exception as exc:
            self._logger.error(f"[STEP {step_index}] EXPANSION EXCEPTION: {exc}", exc_info=True)
            return None
        
        max_branching = int(self.settings.get("max_branching", len(candidates)))
        graph.expand(node_id, candidates[:max_branching])
        
        # Store parent and local goals in children for evaluation/merge context.
        # Every branch should be able to function as an independent sub-problem:
        # a subtree rooted at "learn about strawberries" should look the same
        # whether it appears alone or inside a larger mandate.
        parent_goal = node.details.get(DetailKey.GOAL.value) or node.title
        if not node.details.get(DetailKey.GOAL.value):
            node.details[DetailKey.GOAL.value] = parent_goal
        if not node.details.get(DetailKey.ORIGINAL_GOAL.value):
            node.details[DetailKey.ORIGINAL_GOAL.value] = parent_goal
        parent_original_goal = node.details.get(DetailKey.ORIGINAL_GOAL.value) or parent_goal
        parent_justification = NodeDetailsExtractor.get_justification(node.details)
        
        for child_id in node.children[-max_branching:]:
            child = graph.get_node(child_id)
            if not child:
                continue
            
            # Record hierarchical relationship
            child.details[DetailKey.PARENT_GOAL.value] = parent_goal
            
            # Determine the child's own local goal (branch-level mandate)
            child_goal = (
                child.details.get(DetailKey.GOAL.value)
                or child.details.get(DetailKey.ORIGINAL_GOAL.value)
                or child.title
            )
            if child_goal:
                # Local branch goal
                if not child.details.get(DetailKey.GOAL.value):
                    child.details[DetailKey.GOAL.value] = child_goal
                if not child.details.get(DetailKey.ORIGINAL_GOAL.value):
                    # Treat the child's local goal as its original goal so the
                    # subtree can be evaluated and merged independently.
                    child.details[DetailKey.ORIGINAL_GOAL.value] = child_goal
            
            # Preserve overall/root intent separately via parent_original_goal
            # (available through ancestor chain) without overwriting branch goals.
            if parent_justification:
                child.details[DetailKey.PARENT_JUSTIFICATION.value] = parent_justification
            
            child_justification = NodeDetailsExtractor.get_justification(child.details)
            if not child_justification and parent_justification:
                child.details[DetailKey.WHY_THIS_NODE.value] = f"Parent goal: {parent_goal}. {parent_justification}"
            
            # Mark as leaf if it has an action (search, visit, save)
            if NodeDetailsExtractor.get_action(child.details) and not NodeDetailsExtractor.is_merge_action(child.details):
                child.details[DetailKey.IS_LEAF.value] = True
        
        self._logger.info(f"[STEP {step_index}] EXPANSION: {len(candidates)} candidates → {min(len(candidates), max_branching)} children (total nodes: {graph.node_count()})")
        
        # After expansion, continue to evaluation (return same node_id)
        return node_id
    
    async def _handle_merge_creation(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        """
        Handle merge node creation - creates merge node when children are complete.
        :param graph: IdeaDag instance.
        :param node_id: Parent node identifier.
        :param step_index: Current step index.
        :param branch_pair: Branch pair context.
        :returns: Merge node id or next node id.
        """
        if not self.merge.should_create_merge_node(graph, node_id):
            return node_id
        
        # Retrieve the parent node to access its goal/intent for merge context
        node = graph.get_node(node_id)
        
        merge_node_id = self.merge.create_merge_node(graph, node_id)
        if merge_node_id:
            self._logger.info(f"[STEP {step_index}] MERGE CREATED: {merge_node_id} for parent {node_id}")
            merge_node = graph.get_node(merge_node_id)
            if merge_node and node:
                original_goal = (
                    node.details.get(DetailKey.GOAL.value)
                    or node.details.get(DetailKey.ORIGINAL_GOAL.value)
                    or node.title
                )
                merge_node.details[DetailKey.GOAL.value] = original_goal
                merge_node.details[DetailKey.ORIGINAL_GOAL.value] = original_goal
                
            if merge_node and self._is_action_ready(merge_node, step_index):
                # Check if merge should be skipped (goal not achieved)
                if merge_node.details.get("merge_should_skip", False):
                    self._logger.warning(f"[STEP {step_index}] MERGE SKIPPED: Goal not achieved for node {merge_node_id}")
                    merge_node.status = IdeaNodeStatus.SKIPPED
                    merge_node.details["merge_skipped_reason"] = "Goal not achieved according to evaluation"
                    return node_id
                
                # Execute merge immediately
                result = await self._execute_action(graph, node_id, merge_node_id)
                if result is not None:
                    self._handle_action_result(graph, merge_node_id, step_index)
                
                # After merge, check if goal was achieved
                goal_achieved = merge_node.details.get(DetailKey.GOAL_ACHIEVED.value, False)
                if not goal_achieved and merge_node.details.get("merge_should_skip", False):
                    self._logger.warning(f"[STEP {step_index}] MERGE INCOMPLETE: Goal not achieved, marking as incomplete")
                    merge_node.status = IdeaNodeStatus.SKIPPED
                    merge_node.details["merge_skipped_reason"] = merge_node.details.get("goal_evaluation", "Goal not achieved")
                    return node_id
                
                # After merge, progress toward completion
                if merge_node.status == IdeaNodeStatus.DONE:
                    completion_path = get_completion_path(graph, merge_node_id)
                    if len(completion_path) > 1:
                        next_id = completion_path[1]
                        self._logger.info(f"[STEP {step_index}] MERGE COMPLETE: Progressing toward completion via {next_id}")
                        return next_id
        
        return merge_node_id or node_id
    
    async def _handle_intermediate_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        """
        Handle intermediate nodes - evaluate and select next action.
        :param graph: IdeaDag instance.
        :param node_id: Intermediate node identifier.
        :param step_index: Current step index.
        :param branch_pair: Branch pair context (unused, kept for compatibility).
        :returns: Next node id.
        """
        node = graph.get_node(node_id)
        if not node or not node.children:
            self._logger.warning(f"[STEP {step_index}] No children to evaluate, stopping")
            return None
        
        # Check if children need evaluation (no scores yet)
        needs_evaluation = any(
            (child := graph.get_node(child_id)) and child.score is None
            for child_id in node.children
        )
        
        if needs_evaluation:
            # Evaluate children
            self._logger.debug(f"[STEP {step_index}] Evaluating {len(node.children)} children")
            if hasattr(self.evaluation, "evaluate_batch"):
                await self.evaluation.evaluate_batch(graph, node_id, list(node.children))
            else:
                for child_id in node.children:
                    await self.evaluation.evaluate(graph, child_id)
        
        # Select eligible children
        min_score = float(self.settings.get("min_score_threshold", 0.0))
        allow_unscored = bool(self.settings.get("allow_unscored_selection", True))
        eligible = []
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            # Check if action is ready (handles blocked nodes that are ready to retry)
            if not self._is_action_ready(child, step_index):
                continue
            # Merge nodes don't need scores
            if NodeDetailsExtractor.is_merge_action(child.details):
                eligible.append(child_id)
                continue
            # Blocked nodes that are ready to retry are eligible (they already have scores from before)
            if child.status == IdeaNodeStatus.BLOCKED:
                eligible.append(child_id)
                continue
            if child.score is None and not allow_unscored:
                continue
            if child.score is not None and child.score < min_score:
                continue
            eligible.append(child_id)
        
        if not eligible:
            self._logger.warning(f"[STEP {step_index}] No eligible children after evaluation")
            return None
        
        self._logger.debug(f"[STEP {step_index}] Found {len(eligible)} eligible children")
        original_children = list(node.children)
        node.children = eligible
        try:
            if bool(self.settings.get("best_first_global", False)):
                selected, parent_id = self._select_best_global(graph, min_score, allow_unscored)
            else:
                selected = self.selection.select(graph, node_id)
                parent_id = node_id
        finally:
            node.children = original_children
        
        if selected is None:
            self._logger.warning(f"[STEP {step_index}] Selection returned None")
            return None
        
        meta = node.details.get(DetailKey.EXPANSION_META.value) or {}
        execute_all = bool(meta.get(DetailKey.EXECUTE_ALL_CHILDREN.value)) if isinstance(meta, dict) else False
        
        has_dependencies = self._detect_state_dependencies(graph, eligible)
        has_chunk_dependencies = self._detect_chunk_dependencies(graph, eligible)
        
        if has_dependencies or has_chunk_dependencies:
            self._logger.info(f"[STEP {step_index}] DEPENDENCY DETECTED: Forcing sequential execution (child depends on sibling result)")
            execute_all = False
        
        # For chunk-based searches, allow parallel execution of independent chunks
        chunk_nodes = [nid for nid in eligible if self._is_chunk_node(graph, nid)]
        if chunk_nodes and not has_dependencies:
            self._logger.info(f"[STEP {step_index}] CHUNK MODE: {len(chunk_nodes)} chunk searches can run in parallel")
            execute_all = True
        
        # PARALLEL MODE: Execute all children (only if no dependencies)
        if execute_all and bool(self.settings.get("allow_execute_all_children", True)) and not has_dependencies:
            self._logger.info(f"[STEP {step_index}] PARALLEL: Executing all {len(eligible)} children")
            for child_id in eligible:
                child = graph.get_node(child_id)
                if not child or not self._is_action_ready(child, step_index):
                    continue
                result = await self._execute_action(graph, parent_id, child_id)
                if result is not None:
                    self._handle_action_result(graph, child_id, step_index)
            # Return to parent to check for merge
            return parent_id
        
        # SEQUENTIAL MODE: Execute only the selected (best) child
        # Override: Prefer data-producing nodes (search/visit with valid URLs) over
        # data-consuming nodes (think/save) to ensure correct execution order.
        best_to_execute = self._reorder_for_sequential(graph, selected, eligible, step_index)
        if best_to_execute and best_to_execute.node_id != selected.node_id:
            self._logger.info(f"[STEP {step_index}] SEQUENTIAL REORDER: {selected.title[:40]}... → {best_to_execute.title[:40]}...")
            selected = best_to_execute
        
        self._logger.info(f"[STEP {step_index}] SEQUENTIAL: Executing best child: {selected.title[:50]}...")
        result = await self._execute_action(graph, parent_id, selected.node_id)
        if result is not None:
            self._handle_action_result(graph, selected.node_id, step_index)
        
        # Return to parent to check for merge
        return node_id

    async def _execute_action(self, graph: IdeaDag, parent_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Execute action for a node and store results.
        :param graph: IdeaDag instance.
        :param parent_id: Parent node id.
        :param node_id: Node identifier.
        :returns: None
        """
        node = graph.get_node(node_id)
        if not node:
            return None
        action_type = node.details.get(DetailKey.ACTION.value)
        if not action_type:
            return None
        
        # Check for duplicate actions
        existing_node_id = graph.has_executed_action(str(action_type), node.details)
        if existing_node_id and existing_node_id != node_id:
            existing_node = graph.get_node(existing_node_id)
            if existing_node and existing_node.status == IdeaNodeStatus.DONE:
                self._logger.info(f"[ACTION] Skipping duplicate action (already executed by node {existing_node_id})")
                # Copy result from existing node
                existing_result = existing_node.details.get(DetailKey.ACTION_RESULT.value)
                if existing_result:
                    graph.update_details(node_id, {DetailKey.ACTION_RESULT.value: existing_result})
                    node.status = IdeaNodeStatus.DONE
                    return existing_result
        
        # Check for blocked sites (for visit actions)
        from agent.app.idea_policies.base import IdeaActionType
        if action_type == IdeaActionType.VISIT.value:
            from agent.app.idea_policies.action_constants import NodeDetailsExtractor
            url = NodeDetailsExtractor.get_url(node.details)
            if url:
                blocked_reason = graph.is_site_blocked(str(url))
                if blocked_reason:
                    self._logger.warning(f"[ACTION] Site blocked, skipping: {url} - {blocked_reason}")
                    result = {
                        "action": action_type,
                        "success": False,
                        "url": url,
                        "error": f"Site blocked: {blocked_reason}",
                        "retryable": False,
                    }
                    graph.update_details(node_id, {DetailKey.ACTION_RESULT.value: result})
                    node.status = IdeaNodeStatus.FAILED
                    return result
        
        try:
            allowed = self.settings.get("allowed_actions") or [a.value for a in IdeaActionType]
            if str(action_type) not in [str(item) for item in allowed]:
                action_enum = IdeaActionType.THINK
            else:
                action_enum = IdeaActionType(str(action_type))
        except Exception:
            action_enum = IdeaActionType.THINK
        action = self.actions.get(action_enum)
        attempts = int(node.details.get(DetailKey.ACTION_ATTEMPTS.value, 0)) + 1
        max_retries = int(self.settings.get("action_max_retries", 0))
        graph.update_details(
            node_id,
            {
                DetailKey.ACTION_ATTEMPTS.value: attempts,
                DetailKey.ACTION_MAX_RETRIES.value: max_retries,
            },
        )
        node.status = IdeaNodeStatus.ACTIVE
        result = await action.execute(graph, node_id, self.io)
        # Sanitize result before storing to prevent circular references
        sanitized_result = self._sanitize_action_result(result) if result else None
        graph.update_details(node_id, {DetailKey.ACTION_RESULT.value: sanitized_result})
        
        # Write memory after action execution
        if self._memory_manager and result:
            await self._memory_manager.write_node_result(
                node_id=node_id,
                node_title=node.title,
                action_type=str(action_type),
                result=result,
            )
        
        from agent.app.idea_policies.action_constants import ActionResultKey
        # Mark action as executed if successful
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        if result and ActionResultExtractor.is_success(result):
            graph.mark_action_executed(node_id, str(action_type), node.details)
        
        return result

    def _handle_action_result(self, graph: IdeaDag, node_id: str, step_index: int) -> str:
        """
        Handle action result status and retries.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier.
        :param step_index: Current step index.
        :returns: Outcome label.
        """
        node = graph.get_node(node_id)
        if not node:
            return "missing"
        from agent.app.idea_policies.action_constants import ActionResultKey, ResultStatus
        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            node.status = IdeaNodeStatus.FAILED
            return ResultStatus.FAILED.value
        from agent.app.idea_policies.action_constants import ActionResultExtractor, ResultStatus
        from agent.app.idea_policies.base import IdeaActionType
        success = ActionResultExtractor.is_success(result)
        if success:
            node.status = IdeaNodeStatus.DONE
            
            action = NodeDetailsExtractor.get_action(node.details)
            if action == IdeaActionType.SEARCH.value:
                node.details[DetailKey.PROVIDES_DATA.value] = {"type": "urls_from_search"}
                self._logger.debug(f"[DATA_FLOW] Node {node_id} (search) now provides URLs")
            elif action == IdeaActionType.VISIT.value:
                node.details[DetailKey.PROVIDES_DATA.value] = {"type": "urls_from_visit"}
                self._logger.debug(f"[DATA_FLOW] Node {node_id} (visit) now provides URLs")
            
            return ResultStatus.SUCCESS.value
        retryable = ActionResultExtractor.is_retryable(result)
        node.details[DetailKey.ACTION_RETRYABLE.value] = retryable
        attempts = int(node.details.get(DetailKey.ACTION_ATTEMPTS.value, 0))
        max_retries = int(self.settings.get("action_max_retries", 0))
        if retryable and attempts <= max_retries:
            backoff = int(self.settings.get("action_retry_backoff_steps", 1))
            next_step = step_index + max(1, backoff)
            node.details[DetailKey.ACTION_COOLDOWN_UNTIL.value] = next_step
            node.status = IdeaNodeStatus.BLOCKED
            return "retry"
        
        # Store comprehensive error information for root cause analysis
        from agent.app.idea_policies.action_constants import ActionResultExtractor, ActionResultKey
        error_info = ActionResultExtractor.get_error(result)
        error_type = result.get(ActionResultKey.ERROR_TYPE.value)
        root_cause = result.get(ActionResultKey.ROOT_CAUSE.value)
        http_status = result.get(ActionResultKey.HTTP_STATUS.value)
        traceback_summary = result.get(ActionResultKey.TRACEBACK_SUMMARY.value)
        
        error_details = {
            "error": error_info,
            "error_type": error_type,
            "root_cause": root_cause,
            "http_status": http_status,
            "traceback_summary": traceback_summary,
        }
        node.details[DetailKey.ACTION_ERROR.value] = error_info
        node.details[f"{DetailKey.ACTION_ERROR.value}_details"] = error_details
        node.status = IdeaNodeStatus.FAILED
        return ResultStatus.FAILED.value

    def _reorder_for_sequential(
        self,
        graph: IdeaDag,
        selected: IdeaNode,
        eligible: List[str],
        step_index: int,
    ) -> Optional[IdeaNode]:
        """
        In sequential mode, ensure data-producing nodes (search/visit with URLs)
        execute before data-consuming nodes (think/save).
        
        If the LLM-selected node is a think/save and there are unexecuted
        visit/search nodes, prefer the visit/search node instead.
        
        :param graph: IdeaDag instance.
        :param selected: Node selected by score-based policy.
        :param eligible: List of eligible node IDs.
        :param step_index: Current step index.
        :returns: Reordered node to execute, or None to keep selection.
        """
        selected_action = NodeDetailsExtractor.get_action(selected.details) or ""
        
        # Only reorder if the selected node is a data-consuming type
        data_consuming = {"think", "save", "merge"}
        if selected_action.lower() not in data_consuming:
            return None
        
        # Check if there are unexecuted data-producing siblings
        data_producing_candidates: List[IdeaNode] = []
        for nid in eligible:
            if nid == selected.node_id:
                continue
            child = graph.get_node(nid)
            if not child or child.status.value == "done":
                continue
            child_action = NodeDetailsExtractor.get_action(child.details) or ""
            if child_action.lower() in ("visit", "search"):
                # Prefer visit/search nodes that have a valid URL or no dependency issues
                url = child.details.get("optional_url") or child.details.get("url") or child.details.get("link") or ""
                has_url = isinstance(url, str) and url.startswith(("http://", "https://"))
                has_link_idea = bool(child.details.get("link_idea"))
                if has_url or has_link_idea:
                    data_producing_candidates.append(child)
        
        if not data_producing_candidates:
            return None
        
        # Among data-producing candidates, prefer those with valid URLs
        for candidate in data_producing_candidates:
            url = candidate.details.get("optional_url") or candidate.details.get("url") or candidate.details.get("link") or ""
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return candidate
        
        # Fall back to first data-producing candidate
        return data_producing_candidates[0]
    
    def _detect_state_dependencies(self, graph: IdeaDag, candidate_ids: List[str]) -> bool:
        """
        Detect if candidates have state dependencies (e.g., visit needs URL from search).
        Checks both implicit dependencies (missing URLs) and explicit data dependencies.
        :param graph: IdeaDag instance.
        :param candidate_ids: List of candidate node IDs to check.
        :returns: True if dependencies detected.
        """
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor
        from agent.app.idea_policies.base import IdeaActionType
        
        has_search = False
        has_visit = False
        visit_needs_url = False
        has_data_dependencies = False
        
        for node_id in candidate_ids:
            node = graph.get_node(node_id)
            if not node:
                continue
            
            action = NodeDetailsExtractor.get_action(node.details)
            if action == IdeaActionType.SEARCH.value:
                has_search = True
            elif action == IdeaActionType.VISIT.value:
                has_visit = True
                url = node.details.get(DetailKey.URL.value) or node.details.get(DetailKey.LINK.value) or node.details.get("url") or node.details.get("link")
                if not url or not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    visit_needs_url = True
                
                requires_data = node.details.get(DetailKey.REQUIRES_DATA.value)
                if requires_data and isinstance(requires_data, dict):
                    source_node_id = requires_data.get("source_node_id")
                    if source_node_id and source_node_id in candidate_ids:
                        has_data_dependencies = True
                        self._logger.info(f"[DEPENDENCY] Node {node_id} requires data from sibling {source_node_id} - forcing sequential")
        
        if has_search and has_visit and visit_needs_url:
            return True
        
        if has_data_dependencies:
            return True
        
        return False
    
    def _should_chunk_document(self, graph: IdeaDag, node: IdeaNode) -> bool:
        """
        Determine if a visit node's document should be chunked into sub-problems.
        :param graph: IdeaDag instance.
        :param node: Visit node to check.
        :returns: True if document should be chunked.
        """
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor, ActionResultKey
        from agent.app.idea_policies.base import IdeaActionType
        
        action = NodeDetailsExtractor.get_action(node.details)
        if action != IdeaActionType.VISIT.value:
            return False
        
        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            return False
        
        if not ActionResultExtractor.is_success(result):
            return False
        
        content_total = result.get(ActionResultKey.CONTENT_TOTAL_CHARS.value) or 0
        chunk_threshold = int(self.settings.get("document_chunk_threshold", 10000))
        
        if content_total > chunk_threshold:
            self._logger.info(f"[CHUNKING] Document size {content_total} chars exceeds threshold {chunk_threshold}")
            return True
        
        return False
    
    async def _create_chunk_subproblems(self, graph: IdeaDag, visit_node: IdeaNode) -> Optional[List[str]]:
        """
        Create sub-problems for chunking a large document.
        :param graph: IdeaDag instance.
        :param visit_node: Visit node with large document.
        :returns: List of created chunk node IDs.
        """
        from agent.app.idea_policies.action_constants import ActionResultKey
        from agent.app.idea_policies.base import IdeaActionType
        
        result = visit_node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            return None
        
        content_full = result.get(ActionResultKey.CONTENT_FULL.value) or ""
        if not content_full or not isinstance(content_full, str):
            return None
        
        chunk_size = int(self.settings.get("document_chunk_size", 2000))
        chunk_overlap = int(self.settings.get("document_chunk_overlap", 200))
        
        chunks = self._chunk_text(content_full, chunk_size, chunk_overlap)
        if len(chunks) <= 1:
            return None
        
        original_goal = visit_node.details.get(DetailKey.GOAL.value) or visit_node.details.get(DetailKey.INTENT.value) or visit_node.title
        url = result.get(ActionResultKey.URL.value) or ""
        
        chunk_nodes = []
        for i, chunk in enumerate(chunks):
            chunk_title = f"Search chunk {i+1}/{len(chunks)} of {visit_node.title[:40]}..."
            chunk_details = {
                DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                DetailKey.QUERY.value: original_goal,
                DetailKey.CHUNK_INDEX.value: i,
                DetailKey.TOTAL_CHUNKS.value: len(chunks),
                DetailKey.CHUNK_CONTENT.value: chunk,
                DetailKey.ORIGINAL_GOAL.value: original_goal,
                DetailKey.GOAL.value: f"Find information in chunk {i+1} relevant to: {original_goal}",
                DetailKey.REQUIRES_DATA.value: {
                    "type": "chunk_from_visit",
                    "source_node_id": visit_node.node_id
                },
                DetailKey.JUSTIFICATION.value: f"Document too large ({len(content_full)} chars) - searching chunk {i+1}/{len(chunks)} for: {original_goal}",
            }
            
            chunk_node = graph.add_child(
                parent_id=visit_node.node_id,
                title=chunk_title,
                details=chunk_details,
                status=IdeaNodeStatus.PENDING,
            )
            chunk_nodes.append(chunk_node.node_id)
        
        self._logger.info(f"[CHUNKING] Created {len(chunk_nodes)} chunk sub-problems for document from {url[:60]}...")
        return chunk_nodes
    
    @staticmethod
    def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Split text into chunks with overlap.
        :param text: Text to chunk.
        :param chunk_size: Size of each chunk.
        :param chunk_overlap: Overlap between chunks.
        :returns: List of text chunks.
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + 1, end - chunk_overlap)
        
        return chunks
    
    def _detect_chunk_dependencies(self, graph: IdeaDag, candidate_ids: List[str]) -> bool:
        """
        Detect if candidates are chunk nodes that need sequential processing.
        :param graph: IdeaDag instance.
        :param candidate_ids: List of candidate node IDs to check.
        :returns: True if chunk dependencies detected.
        """
        chunk_nodes = {}
        for node_id in candidate_ids:
            node = graph.get_node(node_id)
            if not node:
                continue
            
            chunk_index = node.details.get(DetailKey.CHUNK_INDEX.value)
            total_chunks = node.details.get(DetailKey.TOTAL_CHUNKS.value)
            source_node_id = None
            requires_data = node.details.get(DetailKey.REQUIRES_DATA.value)
            if isinstance(requires_data, dict):
                source_node_id = requires_data.get("source_node_id")
            
            if chunk_index is not None and total_chunks is not None and source_node_id:
                if source_node_id not in chunk_nodes:
                    chunk_nodes[source_node_id] = []
                chunk_nodes[source_node_id].append((node_id, chunk_index))
        
        for source_id, chunks in chunk_nodes.items():
            if len(chunks) > 1:
                chunks_sorted = sorted(chunks, key=lambda x: x[1] or 0)
                for i, (node_id, idx) in enumerate(chunks_sorted):
                    if i > 0 and idx is not None and chunks_sorted[i-1][1] is not None:
                        if idx <= chunks_sorted[i-1][1]:
                            return True
        
        return False
    
    def _is_chunk_node(self, graph: IdeaDag, node_id: str) -> bool:
        """
        Check if a node is a chunk-based search node.
        :param graph: IdeaDag instance.
        :param node_id: Node ID to check.
        :returns: True if node is a chunk search node.
        """
        node = graph.get_node(node_id)
        if not node:
            return False
        
        return node.details.get(DetailKey.CHUNK_CONTENT.value) is not None
    
    def _get_pending_executable_nodes(self, graph: IdeaDag) -> List[IdeaNode]:
        """
        Find all nodes that have actions but haven't been executed yet.
        :param graph: IdeaDag instance.
        :returns: List of pending executable nodes.
        """
        pending = []
        for node_id, node in graph._nodes.items():
            action = NodeDetailsExtractor.get_action(node.details)
            if action and not NodeDetailsExtractor.is_merge_action(node.details):
                has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None
                if not has_result and node.status not in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                    pending.append(node)
        return pending
    
    def _is_leaf_node(self, graph: IdeaDag, node, step_index: int) -> bool:
        """
        Determine if a node is a leaf (has action ready, no children, or all children complete).
        :param graph: IdeaDag instance.
        :param node: IdeaNode instance.
        :param step_index: Current step index.
        :returns: True if leaf node.
        """
        # If node has an action and is ready, it's a leaf
        action_type = node.details.get(DetailKey.ACTION.value)
        if action_type and self._is_action_ready(node, step_index):
            # Check if action hasn't been executed yet
            if node.details.get(DetailKey.ACTION_RESULT.value) is None:
                return True
        
        # If node has children, check if all are complete
        if node.children:
            all_complete = all(
                (child := graph.get_node(child_id)) and 
                child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.BLOCKED, IdeaNodeStatus.SKIPPED)
                for child_id in node.children
            )
            if all_complete:
                return True
        
        return False

    @staticmethod
    def _sanitize_action_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize action result to ensure JSON serializability.
        :param result: Action result dictionary.
        :returns: Sanitized result dictionary.
        """
        if not isinstance(result, dict):
            return {"error": f"Invalid result type: {type(result)}"}
        
        sanitized = {}
        for key, value in result.items():
            if value is None:
                sanitized[key] = None
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = IdeaDagEngine._sanitize_action_result(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [
                    IdeaDagEngine._sanitize_action_result(item) if isinstance(item, dict) else str(item)
                    for item in value
                ]
            else:
                sanitized[key] = str(value)
        return sanitized

    def _is_action_ready(self, node, step_index: int) -> bool:
        """
        Determine whether a node is ready for execution.
        :param node: IdeaNode instance.
        :param step_index: Current step index.
        :returns: True if ready.
        """
        if node.status in (IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
            return False
        cooldown = node.details.get(DetailKey.ACTION_COOLDOWN_UNTIL.value)
        if isinstance(cooldown, int) and step_index < cooldown:
            return False
        return True

    def _check_and_create_merge_nodes(self, graph: IdeaDag, node_id: str, step_index: int) -> None:
        """
        Check if children are ready to merge and create merge nodes if needed.
        Recursively checks up the tree.
        :param graph: IdeaDag instance.
        :param node_id: Node identifier to check.
        :param step_index: Current step index.
        :returns: None
        """
        node = graph.get_node(node_id)
        if not node:
            return
        
        # Check if this node's children are ready to merge
        if self.merge.should_create_merge_node(graph, node_id):
            merge_node_id = self.merge.create_merge_node(graph, node_id)
            if merge_node_id:
                self._logger.debug(f"[STEP {step_index}] Created merge node {merge_node_id} for parent {node_id}")
        
        # Recursively check parent
        if node.parent_id:
            self._check_and_create_merge_nodes(graph, node.parent_id, step_index)

    def _select_best_global(self, graph: IdeaDag, min_score: float, allow_unscored: bool) -> tuple[Optional[Any], Optional[str]]:
        """
        Select the best scored node across the graph.
        :param graph: IdeaDag instance.
        :param min_score: Minimum score threshold.
        :param allow_unscored: Allow unscored nodes.
        :returns: Tuple of node and parent_id.
        """
        best = None
        for node in graph.iter_depth_first():
            if node.parent_id is None:
                continue
            if node.details.get(DetailKey.ACTION_RESULT.value) is not None and node.status == IdeaNodeStatus.DONE:
                continue
            if not self._is_action_ready(node, self._step_index):
                continue
            if node.score is None and not allow_unscored:
                continue
            if node.score is not None and node.score < min_score:
                continue
            if best is None or (node.score or 0.0) > (best.score or 0.0):
                best = node
        if not best:
            return None, None
        parent_id = best.parent_id
        if parent_id is None and best.parent_ids:
            parent_id = best.parent_ids[0]
        return best, parent_id

    def _maybe_log_dag(self, graph: IdeaDag, step_index: int, force: bool = False) -> None:
        """
        Optionally log DAG ASCII representation.
        :param graph: IdeaDag instance.
        :param step_index: Current step index.
        :param force: Force logging.
        :returns: None
        """
        if not self.settings.get("log_dag_ascii"):
            return
        interval = int(self.settings.get("log_dag_step_interval", 0))
        if not force and interval <= 0:
            return
        if not force and step_index % interval != 0:
            return
        try:
            from agent.app.idea_dag_log import idea_dag_to_ascii
            self._logger.info("\n%s", idea_dag_to_ascii(graph))
        except Exception:
            return

    @staticmethod
    def _memo_namespace(mandate: str) -> str:
        """
        Build a memoization namespace for a mandate.
        :param mandate: Mandate text.
        :returns: Namespace string.
        """
        digest = hashlib.sha256(mandate.encode("utf-8")).hexdigest()[:10]
        return f"idea_dag:{digest}"
