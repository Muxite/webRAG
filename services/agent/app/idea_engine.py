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
from agent.app.got_operations import GoTOperations


class IdeaDagEngine:
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
        self._got: Optional[GoTOperations] = None

    async def run(self, mandate: str, max_steps: int = 50) -> Dict[str, Any]:
        mandate_short = mandate.split("\n\nTask Statement")[0] if "\n\nTask Statement" in mandate else mandate[:100]
        self._logger.info(f"[RUN] Starting idea DAG engine with mandate: {mandate_short}..., max_steps={max_steps}")
        namespace = self._memo_namespace(mandate)
        self.settings[DetailKey.MEMO_NAMESPACE.value] = namespace
        self._memory_manager = MemoryManager(
            connector_chroma=self.io.connector_chroma,
            namespace=namespace,
        )
        self._got = GoTOperations(
            settings=self.settings,
            io=self.io,
            memory_manager=self._memory_manager,
        )
        self._current_mandate = mandate
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

            if self._got and steps % 5 == 0:
                prune_ids = self._got.identify_prune_candidates(graph)
                if prune_ids:
                    self._got.prune_nodes(graph, prune_ids)

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
            
            
            if steps == 3:
                action_count = sum(1 for n in graph.iter_depth_first() if NodeDetailsExtractor.get_action(n.details))
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
        
        final_payload = await build_final_payload(
            self.io, self.settings, graph, mandate, self.model_name,
            memory_manager=self._memory_manager,
        )
        final_payload["graph"] = graph.to_dict()
        final_payload["pending_nodes_count"] = len(pending_nodes) if pending_nodes else 0
        if pending_nodes:
            final_payload["warning"] = f"Finalized with {len(pending_nodes)} pending nodes - execution incomplete"

        if self._got:
            pruned_count = sum(
                1 for n in graph.iter_depth_first()
                if n.details.get("_got_pruned")
            )
            improved_count = sum(
                1 for n in graph.iter_depth_first()
                if n.details.get("_got_improve_iterations", 0) > 0
            )
            final_payload["got_stats"] = {
                "dead_ends_detected": self._got.dead_end_count,
                "nodes_pruned": pruned_count,
                "nodes_improved": improved_count,
            }

        self._logger.info(f"[RUN] Final payload created, graph has {graph.node_count()} nodes, {len(pending_nodes) if pending_nodes else 0} pending")
        self._maybe_log_dag(graph, steps, force=True)
        return final_payload

    async def step(self, graph: IdeaDag, current_id: str, step_index: int) -> Optional[str]:
        self._logger.info(f"[STEP {step_index}] Starting step with current_id={current_id}, node_count={graph.node_count()}")
        if graph.node_count() >= int(self.settings.get("max_total_nodes", 500)):
            self._logger.warning(f"[STEP {step_index}] Max nodes reached, stopping")
            return None
        
        node = graph.get_node(current_id)
        if not node:
            self._logger.warning(f"[STEP {step_index}] Node {current_id} not found")
            return None
        
        self._logger.info(f"[STEP {step_index}] Processing node: {node.title[:80]}... (status={node.status.value}, children={len(node.children)}, action={NodeDetailsExtractor.get_action(node.details)})")
        
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
        
        if NodeDetailsExtractor.is_merge_action(node.details):
            self._logger.info(f"[STEP {step_index}] Node is merge node, handling merge")
            return await self._handle_merge_node(graph, current_id, step_index, None)
        
        is_leaf = node.details.get(DetailKey.IS_LEAF.value, False)
        action = NodeDetailsExtractor.get_action(node.details)
        has_action = action and not NodeDetailsExtractor.is_merge_action(node.details)
        if is_leaf or has_action:
            self._logger.info(f"[STEP {step_index}] Node is leaf node (is_leaf={is_leaf}, has_action={has_action}, action={action}), executing action")
            return await self._handle_leaf_node(graph, current_id, step_index, None)
        
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
        
        leaf_children = []
        merge_children = []
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            if NodeDetailsExtractor.is_merge_action(child.details):
                merge_children.append(child_id)
            else:
                leaf_children.append(child_id)
        
        all_terminal = all(
            (child := graph.get_node(cid)) and
            child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED)
            for cid in node.children
        )
        if all_terminal and merge_children:
            self._logger.info(f"[STEP {step_index}] All children complete (incl. merge), marking parent done")
            node.status = IdeaNodeStatus.DONE
            if node.parent_id:
                return node.parent_id
            return None
        
        all_leaves_complete = leaf_children and all(
            (child := graph.get_node(cid)) and
            child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED)
            for cid in leaf_children
        )
        if all_leaves_complete and not merge_children:
            branch_pair = find_branch_pair(graph, current_id)
            leaf_statuses = {cid: graph.get_node(cid).status.value for cid in leaf_children if graph.get_node(cid)}
            self._logger.info(f"[STEP {step_index}] ALL LEAVES COMPLETE ({len(leaf_children)}): {leaf_statuses} — creating merge")
            if branch_pair and branch_pair.needs_merge():
                return await self._handle_merge_creation(graph, current_id, step_index, branch_pair)
        elif not all_leaves_complete and leaf_children:
            pending = {cid: graph.get_node(cid).status.value for cid in leaf_children
                       if graph.get_node(cid) and graph.get_node(cid).status not in
                       (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED)}
            self._logger.debug(f"[STEP {step_index}] MERGE GATE: {len(pending)} leaf children still pending: {pending}")
        
        if all_leaves_complete and merge_children:
            for merge_cid in merge_children:
                merge_child = graph.get_node(merge_cid)
                if merge_child and merge_child.status not in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                    self._logger.info(f"[STEP {step_index}] All leaves done, executing merge child {merge_cid}")
                    return merge_cid
        
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
        
        return await self._handle_intermediate_node(graph, current_id, step_index, None)
    
    async def _handle_merge_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        node = graph.get_node(node_id)
        if not node:
            return None
        
        if node.details.get(DetailKey.ACTION_RESULT.value) is None:
            if self._is_action_ready(node, step_index):
                self._logger.info(f"[STEP {step_index}] Executing merge action for node {node_id}")
                result = await self._execute_action(graph, node.parent_id or graph.root_id(), node_id)
                if result is not None:
                    self._handle_action_result(graph, node_id, step_index)
                else:
                    self._logger.warning(f"[STEP {step_index}] Merge action for {node_id} returned None; marking DONE")
                    node.status = IdeaNodeStatus.DONE
        
        parent_id = node.parent_id or graph.root_id()
        if node.status == IdeaNodeStatus.DONE:
            self._logger.info(f"[STEP {step_index}] MERGE COMPLETE for {node_id}: returning to parent {parent_id}")
            return parent_id
        
        if node.status == IdeaNodeStatus.FAILED:
            self._logger.warning(f"[STEP {step_index}] MERGE FAILED for {node_id}: returning to parent {parent_id}")
            return parent_id
        
        return node_id
    
    async def _handle_leaf_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        node = graph.get_node(node_id)
        if not node:
            return None
        
        if not self._has_required_data(graph, node):
            self._logger.warning(f"[STEP {step_index}] Node {node_id} missing required data - cannot execute yet")
            return node.parent_id if node.parent_id else None
        
        has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None
        is_blocked_ready = node.status == IdeaNodeStatus.BLOCKED and self._is_action_ready(node, step_index)
        
        if not has_result or is_blocked_ready:
            if self._is_action_ready(node, step_index):
                result = await self._execute_action(graph, node.parent_id or graph.root_id(), node_id)
                if result is not None:
                    self._handle_action_result(graph, node_id, step_index)
        
        if node.status == IdeaNodeStatus.DONE:
            should_chunk = self._should_chunk_document(graph, node)
            if should_chunk:
                self._logger.info(f"[STEP {step_index}] Large document detected - creating chunk sub-problems")
                chunk_nodes = await self._create_chunk_subproblems(graph, node)
                if chunk_nodes:
                    return chunk_nodes[0] if chunk_nodes else node.parent_id
        
        if node.status == IdeaNodeStatus.DONE and node.parent_id:
            return node.parent_id
        
        return node_id
    
    def _has_required_data(self, graph: IdeaDag, node: IdeaNode) -> bool:
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
    
    async def _handle_expansion_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        node = graph.get_node(node_id)
        if not node:
            return None
        
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
            
            n_internal = int(self.settings.get("expansion_chroma_internal", 5))
            n_observations = int(self.settings.get("expansion_chroma_observations", 5))
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

            if self._got:
                hybrid_extras = await self._got.hybrid_retrieve(
                    graph, node_id, query, n_results=3,
                )
                seen_ids = {m.get("id") for m in memories if m.get("id")}
                for extra in hybrid_extras:
                    if extra.get("id") not in seen_ids:
                        memories.append(extra)
                        seen_ids.add(extra.get("id"))
            self._logger.info(
                f"[STEP {step_index}] EXPANSION: Retrieved {len(split_memories['internal_thoughts'])} internal thoughts, "
                f"{len(split_memories['observations'])} observations from vector DB"
            )
            if split_memories["observations"]:
                obs_preview = "\n".join([str(obs.get("content", "") if isinstance(obs, dict) else obs)[:200] for obs in split_memories["observations"][:3]])
                self._logger.info(f"[STEP {step_index}] Observations preview:\n{obs_preview}")
        
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
        
        filtered = [
            c for c in candidates
            if c.get("details", {}).get(DetailKey.ACTION.value) != IdeaActionType.MERGE.value
            and c.get("action") != IdeaActionType.MERGE.value
        ]
        if len(filtered) < len(candidates):
            self._logger.info(
                f"[STEP {step_index}] EXPANSION: Stripped {len(candidates) - len(filtered)} "
                f"LLM-generated merge candidates (merge is system-managed)"
            )
        candidates = filtered or candidates[:1]

        if self._got:
            candidates = await self._got.filter_duplicate_candidates(candidates, graph)

        if self._got:
            max_branching = self._got.compute_dynamic_beam_width(graph)
        else:
            max_branching = int(self.settings.get("max_branching", len(candidates)))
        hard_cap = int(self.settings.get("max_branching", 500))
        max_branching = min(max_branching, hard_cap)
        graph.expand(node_id, candidates[:max_branching])
        
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
            
            child.details[DetailKey.PARENT_GOAL.value] = parent_goal
            child_goal = (
                child.details.get(DetailKey.GOAL.value)
                or child.details.get(DetailKey.ORIGINAL_GOAL.value)
                or child.title
            )
            if child_goal:
                if not child.details.get(DetailKey.GOAL.value):
                    child.details[DetailKey.GOAL.value] = child_goal
                if not child.details.get(DetailKey.ORIGINAL_GOAL.value):
                    child.details[DetailKey.ORIGINAL_GOAL.value] = child_goal
            
            if parent_justification:
                child.details[DetailKey.PARENT_JUSTIFICATION.value] = parent_justification
            
            child_justification = NodeDetailsExtractor.get_justification(child.details)
            if not child_justification and parent_justification:
                child.details[DetailKey.WHY_THIS_NODE.value] = f"Parent goal: {parent_goal}. {parent_justification}"
            
            if NodeDetailsExtractor.get_action(child.details) and not NodeDetailsExtractor.is_merge_action(child.details):
                child.details[DetailKey.IS_LEAF.value] = True
        
        self._logger.info(f"[STEP {step_index}] EXPANSION: {len(candidates)} candidates -> {min(len(candidates), max_branching)} children (total nodes: {graph.node_count()})")
        
        self._enforce_visit_nodes_for_mandate_urls(graph, node_id, step_index)
        self._enforce_mandate_visit_requirements(graph, node_id, step_index)

        if self._got:
            await self._got.embed_children(graph, node_id)

        return node_id
    
    def _enforce_visit_nodes_for_mandate_urls(self, graph: IdeaDag, node_id: str, step_index: int) -> None:
        import re
        node = graph.get_node(node_id)
        if not node:
            return

        mandate = node.details.get("mandate") or ""
        if not mandate:
            root = graph.get_node(graph.root_id())
            if root and root.node_id == node_id:
                mandate = root.title or ""
            elif root:
                mandate = root.details.get("mandate") or root.title or ""
        if not mandate:
            return

        raw_urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', mandate)
        mandate_urls = [self._clean_extracted_url(u) for u in raw_urls]
        mandate_urls = [u for u in mandate_urls if u]
        if not mandate_urls:
            return

        covered_urls = set()
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            child_action = NodeDetailsExtractor.get_action(child.details)
            if child_action == IdeaActionType.VISIT.value:
                child_url = (
                    child.details.get(DetailKey.URL.value)
                    or child.details.get("optional_url")
                    or ""
                )
                if child_url:
                    covered_urls.add(child_url.rstrip("/"))

        missing_urls = [
            u for u in mandate_urls if u.rstrip("/") not in covered_urls
        ]
        if not missing_urls:
            return

        for url in missing_urls:
            visit_node = graph.add_child(
                parent_id=node_id,
                title=f"Visit {url[:60]}",
                details={
                    DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                    DetailKey.URL.value: url,
                    "optional_url": url,
                    DetailKey.IS_LEAF.value: True,
                    DetailKey.JUSTIFICATION.value: f"Mandate requires visiting this URL",
                    DetailKey.GOAL.value: f"Visit and extract information from {url}",
                },
            )
            self._logger.info(
                f"[STEP {step_index}] ENFORCE: Injected visit node {visit_node.node_id} "
                f"for mandate URL {url[:60]}"
            )

    def _enforce_mandate_visit_requirements(self, graph: IdeaDag, node_id: str, step_index: int) -> None:
        """
        Enforce that mandates explicitly requiring visits/search must create those actions.
        Checks for phrases like "must visit", "you must visit", "must search", etc.
        """
        node = graph.get_node(node_id)
        if not node:
            return

        mandate = node.details.get("mandate") or ""
        if not mandate:
            root = graph.get_node(graph.root_id())
            if root and root.node_id == node_id:
                mandate = root.title or ""
            elif root:
                mandate = root.details.get("mandate") or root.title or ""
        if not mandate:
            return

        mandate_lower = mandate.lower()
        
        requires_visit = any(phrase in mandate_lower for phrase in [
            "must visit", "you must visit", "must visit the", "required to visit",
            "need to visit", "should visit", "visit the url", "visit the page"
        ])
        
        requires_search = any(phrase in mandate_lower for phrase in [
            "must search", "you must search", "search for", "find and visit"
        ])
        
        if not requires_visit and not requires_search:
            return

        has_visit = False
        has_search = False
        
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            child_action = NodeDetailsExtractor.get_action(child.details)
            if child_action == IdeaActionType.VISIT.value:
                has_visit = True
            elif child_action == IdeaActionType.SEARCH.value:
                has_search = True

        search_node_id = None
        if requires_search and not has_search:
            self._logger.warning(
                f"[STEP {step_index}] ENFORCE: Mandate requires search but no search node created. "
                f"Injecting search node."
            )
            search_node = graph.add_child(
                parent_id=node_id,
                title="Search for required information",
                details={
                    DetailKey.ACTION.value: IdeaActionType.SEARCH.value,
                    DetailKey.QUERY.value: mandate[:200],
                    DetailKey.IS_LEAF.value: True,
                    DetailKey.JUSTIFICATION.value: "Mandate explicitly requires search action",
                    DetailKey.GOAL.value: f"Search as required by mandate: {mandate[:100]}",
                },
            )
            search_node_id = search_node.node_id
            has_search = True
            self._logger.info(
                f"[STEP {step_index}] ENFORCE: Injected search node {search_node_id} "
                f"for mandate requirement"
            )
        elif has_search:
            for child_id in node.children:
                child = graph.get_node(child_id)
                if child and NodeDetailsExtractor.get_action(child.details) == IdeaActionType.SEARCH.value:
                    search_node_id = child_id
                    break

        if requires_visit and not has_visit:
            self._logger.warning(
                f"[STEP {step_index}] ENFORCE: Mandate requires visit but no visit node created. "
                f"Injecting visit node."
            )
            visit_node = graph.add_child(
                parent_id=node_id,
                title="Visit required URL",
                details={
                    DetailKey.ACTION.value: IdeaActionType.VISIT.value,
                    DetailKey.IS_LEAF.value: True,
                    DetailKey.JUSTIFICATION.value: "Mandate explicitly requires visit action - will extract URL from search results or use link_idea",
                    DetailKey.GOAL.value: f"Visit URL as required by mandate: {mandate[:100]}",
                    "link_idea": "URL from search results or mandate",
                    "link_count": 1,
                },
            )
            if search_node_id:
                visit_node.details[DetailKey.REQUIRES_DATA.value] = {
                    "type": "urls_from_search",
                    "source_node_id": search_node_id
                }
                self._logger.info(
                    f"[STEP {step_index}] ENFORCE: Visit node {visit_node.node_id} "
                    f"depends on search node {search_node_id}"
                )
            self._logger.info(
                f"[STEP {step_index}] ENFORCE: Injected visit node {visit_node.node_id} "
                f"for mandate requirement"
            )

    @staticmethod
    def _clean_extracted_url(url: str) -> str:
        """
        Strip trailing punctuation from a URL while preserving balanced parentheses.
        Wikipedia URLs like https://en.wikipedia.org/wiki/Python_(programming_language)
        must keep the closing paren when the opening paren is present in the path.
        :param url: Raw extracted URL.
        :returns: Cleaned URL.
        """
        strip_chars = ".,;:!?"
        url = url.rstrip(strip_chars)
        while url.endswith(")") and url.count("(") < url.count(")"):
            url = url[:-1]
        url = url.rstrip(strip_chars)
        return url

    async def _handle_merge_creation(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        if not self.merge.should_create_merge_node(graph, node_id):
            return node_id
        
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
                if merge_node.details.get("merge_should_skip", False):
                    self._logger.warning(f"[STEP {step_index}] MERGE SKIPPED: Goal not achieved for node {merge_node_id}")
                    merge_node.status = IdeaNodeStatus.SKIPPED
                    merge_node.details["merge_skipped_reason"] = "Goal not achieved according to evaluation"
                    return node_id
                
                result = await self._execute_action(graph, node_id, merge_node_id)
                if result is not None:
                    self._handle_action_result(graph, merge_node_id, step_index)
                
                goal_achieved = merge_node.details.get(DetailKey.GOAL_ACHIEVED.value, False)
                if not goal_achieved and merge_node.details.get("merge_should_skip", False):
                    self._logger.warning(f"[STEP {step_index}] MERGE INCOMPLETE: Goal not achieved, marking as incomplete")
                    merge_node.status = IdeaNodeStatus.SKIPPED
                    merge_node.details["merge_skipped_reason"] = merge_node.details.get("goal_evaluation", "Goal not achieved")
                    return node_id
                
                if merge_node.status == IdeaNodeStatus.DONE:
                    completion_path = get_completion_path(graph, merge_node_id)
                    if len(completion_path) > 1:
                        next_id = completion_path[1]
                        self._logger.info(f"[STEP {step_index}] MERGE COMPLETE: Progressing toward completion via {next_id}")
                        return next_id
        
        return merge_node_id or node_id
    
    async def _handle_intermediate_node(self, graph: IdeaDag, node_id: str, step_index: int, branch_pair: Optional[BranchPair]) -> Optional[str]:
        node = graph.get_node(node_id)
        if not node or not node.children:
            self._logger.warning(f"[STEP {step_index}] No children to evaluate, stopping")
            return None

        eligible = []
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            if child.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                continue
            if not self._is_action_ready(child, step_index):
                continue
            eligible.append(child_id)

        if not eligible:
            self._logger.warning(f"[STEP {step_index}] No eligible children")
            return None

        meta = node.details.get(DetailKey.EXPANSION_META.value) or {}
        execute_all = bool(meta.get(DetailKey.EXECUTE_ALL_CHILDREN.value)) if isinstance(meta, dict) else False

        has_dependencies = self._detect_state_dependencies(graph, eligible)
        has_chunk_dependencies = self._detect_chunk_dependencies(graph, eligible)

        if has_dependencies or has_chunk_dependencies:
            self._logger.info(f"[STEP {step_index}] DEPENDENCY DETECTED: Forcing sequential execution")
            execute_all = False

        chunk_nodes = [nid for nid in eligible if self._is_chunk_node(graph, nid)]
        if chunk_nodes and not has_dependencies:
            execute_all = True

        if execute_all and bool(self.settings.get("allow_execute_all_children", True)) and not has_dependencies:
            self._logger.info(f"[STEP {step_index}] PARALLEL: Executing all {len(eligible)} children (skipping evaluation)")
            for child_id in eligible:
                child = graph.get_node(child_id)
                if not child or not self._is_action_ready(child, step_index):
                    continue
                result = await self._execute_action(graph, node_id, child_id)
                if result is not None:
                    self._handle_action_result(graph, child_id, step_index)
            return node_id

        needs_evaluation = any(
            (child := graph.get_node(child_id)) and child.score is None
            for child_id in eligible
        )

        if needs_evaluation:
            self._logger.debug(f"[STEP {step_index}] Evaluating {len(eligible)} children")
            if hasattr(self.evaluation, "evaluate_batch"):
                await self.evaluation.evaluate_batch(graph, node_id, list(eligible))
            else:
                for child_id in eligible:
                    await self.evaluation.evaluate(graph, child_id)

        min_score = float(self.settings.get("min_score_threshold", 0.0))
        allow_unscored = bool(self.settings.get("allow_unscored_selection", True))
        scored_eligible = []
        for child_id in eligible:
            child = graph.get_node(child_id)
            if not child:
                continue
            if NodeDetailsExtractor.is_merge_action(child.details):
                scored_eligible.append(child_id)
                continue
            if child.status == IdeaNodeStatus.BLOCKED:
                scored_eligible.append(child_id)
                continue
            if child.score is None and not allow_unscored:
                continue
            if child.score is not None and child.score < min_score:
                continue
            scored_eligible.append(child_id)

        if not scored_eligible:
            self._logger.warning(f"[STEP {step_index}] No eligible children after evaluation")
            return None

        self._logger.debug(f"[STEP {step_index}] Found {len(scored_eligible)} eligible children")
        original_children = list(node.children)
        node.children = scored_eligible
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

        best_to_execute = self._reorder_for_sequential(graph, selected, scored_eligible, step_index)
        if best_to_execute and best_to_execute.node_id != selected.node_id:
            self._logger.info(f"[STEP {step_index}] SEQUENTIAL REORDER: {selected.title[:40]}... -> {best_to_execute.title[:40]}...")
            selected = best_to_execute

        self._logger.info(f"[STEP {step_index}] SEQUENTIAL: Executing best child: {selected.title[:50]}...")
        result = await self._execute_action(graph, parent_id or node_id, selected.node_id)
        if result is not None:
            self._handle_action_result(graph, selected.node_id, step_index)

        if bool(self.settings.get("sequential_prune_siblings", False)):
            for cid in scored_eligible:
                if cid == selected.node_id:
                    continue
                sibling = graph.get_node(cid)
                if sibling and sibling.status not in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                    sibling.status = IdeaNodeStatus.SKIPPED
            return selected.node_id

        return node_id

    async def _execute_action(self, graph: IdeaDag, parent_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        node = graph.get_node(node_id)
        if not node:
            return None
        action_type = node.details.get(DetailKey.ACTION.value)
        if not action_type:
            return None
        
        existing_node_id = graph.has_executed_action(str(action_type), node.details)
        if existing_node_id and existing_node_id != node_id:
            existing_node = graph.get_node(existing_node_id)
            if existing_node and existing_node.status == IdeaNodeStatus.DONE:
                self._logger.info(f"[ACTION] Skipping duplicate action (already executed by node {existing_node_id})")
                existing_result = existing_node.details.get(DetailKey.ACTION_RESULT.value)
                if existing_result:
                    graph.update_details(node_id, {DetailKey.ACTION_RESULT.value: existing_result})
                    node.status = IdeaNodeStatus.DONE
                    return existing_result
        
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
        sanitized_result = self._sanitize_action_result(result) if result else None
        graph.update_details(node_id, {DetailKey.ACTION_RESULT.value: sanitized_result})
        
        if self._memory_manager and result:
            await self._memory_manager.write_node_result(
                node_id=node_id,
                node_title=node.title,
                action_type=str(action_type),
                result=result,
            )
        
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        if result and ActionResultExtractor.is_success(result):
            graph.mark_action_executed(node_id, str(action_type), node.details)
        
        return result

    def _handle_action_result(self, graph: IdeaDag, node_id: str, step_index: int) -> str:
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
            action = NodeDetailsExtractor.get_action(node.details)
            
            if action == IdeaActionType.VISIT.value:
                url = result.get(ActionResultKey.URL.value) or result.get("url") or ""
                content = result.get(ActionResultKey.CONTENT.value) or result.get(ActionResultKey.CONTENT_FULL.value) or result.get("content") or ""
                content_total_chars = result.get(ActionResultKey.CONTENT_TOTAL_CHARS.value) or len(content) if content else 0
                
                if not content or len(content.strip()) == 0:
                    self._logger.error(
                        f"[VISIT_VALIDATION] Visit marked as success but has no content! "
                        f"Node {node_id}, URL: {url[:80] if url else 'unknown'}, "
                        f"result keys: {list(result.keys())}"
                    )
                    result[ActionResultKey.SUCCESS.value] = False
                    result[ActionResultKey.ERROR.value] = "Visit succeeded but no content was retrieved - this indicates a validation failure"
                    result[ActionResultKey.ERROR_TYPE.value] = "ValidationError"
                    result[ActionResultKey.RETRYABLE.value] = True
                    success = False
                else:
                    self._logger.info(
                        f"[VISIT_SUCCESS] Visit succeeded: Node {node_id}, "
                        f"URL: {url[:80] if url else 'unknown'}, "
                        f"Content: {content_total_chars} chars, "
                        f"Status: DONE"
                    )
                    node.details[DetailKey.PROVIDES_DATA.value] = {"type": "urls_from_visit"}
                    node.details["visit_content_length"] = content_total_chars
                    node.details["visit_url"] = url

            if action == IdeaActionType.SEARCH.value:
                node.details[DetailKey.PROVIDES_DATA.value] = {"type": "urls_from_search"}
                self._logger.debug(f"[DATA_FLOW] Node {node_id} (search) now provides URLs")

            node.status = IdeaNodeStatus.DONE
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
        selected_action = NodeDetailsExtractor.get_action(selected.details) or ""
        # If a visit node has no explicit URL, it often depends on a sibling search node
        # to provide URLs. In sequential mode, enforce search-before-visit regardless
        # of score so we don't execute a visit prematurely and fail with "missing URL".
        if selected_action.lower() == "visit":
            url = (
                selected.details.get("optional_url")
                or selected.details.get(DetailKey.URL.value)
                or selected.details.get(DetailKey.LINK.value)
                or selected.details.get("url")
                or selected.details.get("link")
            )
            has_url = isinstance(url, str) and url.startswith(("http://", "https://"))
            if not has_url:
                search_candidates: List[IdeaNode] = []
                for nid in eligible:
                    if nid == selected.node_id:
                        continue
                    child = graph.get_node(nid)
                    if not child or child.status.value in ("done", "failed", "skipped"):
                        continue
                    child_action = NodeDetailsExtractor.get_action(child.details) or ""
                    if child_action.lower() == "search":
                        search_candidates.append(child)
                if search_candidates:
                    # Prefer the highest-scored search (or first if unscored).
                    best_search = max(search_candidates, key=lambda n: n.score if n.score is not None else float("-inf"))
                    return best_search

        data_consuming = {"think", "save", "merge"}
        if selected_action.lower() not in data_consuming:
            return None
        
        data_producing_candidates: List[IdeaNode] = []
        for nid in eligible:
            if nid == selected.node_id:
                continue
            child = graph.get_node(nid)
            if not child or child.status.value == "done":
                continue
            child_action = NodeDetailsExtractor.get_action(child.details) or ""
            if child_action.lower() == "search":
                data_producing_candidates.append(child)
            elif child_action.lower() == "visit":
                url = child.details.get("optional_url") or child.details.get("url") or child.details.get("link") or ""
                has_url = isinstance(url, str) and url.startswith(("http://", "https://"))
                has_link_idea = bool(child.details.get("link_idea"))
                if has_url or has_link_idea:
                    data_producing_candidates.append(child)
        
        if not data_producing_candidates:
            return None
        
        for candidate in data_producing_candidates:
            url = candidate.details.get("optional_url") or candidate.details.get("url") or candidate.details.get("link") or ""
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                return candidate
        
        return data_producing_candidates[0]
    
    def _detect_state_dependencies(self, graph: IdeaDag, candidate_ids: List[str]) -> bool:
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
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor, ActionResultKey, ActionResultExtractor
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
        chunk_threshold = int(self.settings.get("document_chunk_threshold", 200000))
        
        if content_total > chunk_threshold:
            self._logger.info(f"[CHUNKING] Document size {content_total} chars exceeds threshold {chunk_threshold}")
            return True
        
        return False
    
    async def _create_chunk_subproblems(self, graph: IdeaDag, visit_node: IdeaNode) -> Optional[List[str]]:
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
        node = graph.get_node(node_id)
        if not node:
            return False
        
        return node.details.get(DetailKey.CHUNK_CONTENT.value) is not None
    
    def _get_pending_executable_nodes(self, graph: IdeaDag) -> List[IdeaNode]:
        pending = []
        for node in graph.iter_depth_first():
            action = NodeDetailsExtractor.get_action(node.details)
            if action and not NodeDetailsExtractor.is_merge_action(node.details):
                has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None
                if not has_result and node.status not in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                    pending.append(node)
        return pending
    
    def _is_leaf_node(self, graph: IdeaDag, node, step_index: int) -> bool:
        action_type = node.details.get(DetailKey.ACTION.value)
        if action_type and self._is_action_ready(node, step_index):
            if node.details.get(DetailKey.ACTION_RESULT.value) is None:
                return True
        
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
        if node.status in (IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
            return False
        cooldown = node.details.get(DetailKey.ACTION_COOLDOWN_UNTIL.value)
        if isinstance(cooldown, int) and step_index < cooldown:
            return False
        return True

    def _check_and_create_merge_nodes(self, graph: IdeaDag, node_id: str, step_index: int) -> None:
        node = graph.get_node(node_id)
        if not node:
            return
        
        if self.merge.should_create_merge_node(graph, node_id):
            merge_node_id = self.merge.create_merge_node(graph, node_id)
            if merge_node_id:
                self._logger.debug(f"[STEP {step_index}] Created merge node {merge_node_id} for parent {node_id}")
        
        if node.parent_id:
            self._check_and_create_merge_nodes(graph, node.parent_id, step_index)

    def _select_best_global(self, graph: IdeaDag, min_score: float, allow_unscored: bool) -> tuple[Optional[Any], Optional[str]]:
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
        digest = hashlib.sha256(mandate.encode("utf-8")).hexdigest()[:10]
        return f"idea_dag:{digest}"
