from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode

from agent.app.idea_policies.base import MergePolicy, DetailKey, IdeaActionType, IdeaNodeStatus
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.alternative_branch import (
    RACE_COMPLETED_STEP,
    RACE_GROUPS,
    RACE_GROUPS_INFERRED,
    RACE_LOSER,
    is_alternative_pending,
)


class SimpleMergePolicy(MergePolicy):
    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        super().__init__(settings=settings)
        self._cfg = IdeaConfig.from_settings(self.settings)
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
            # A parked A->B fallback is BLOCKED by design and may never run at all, so it
            # is not work this merge is waiting on. Without this, a primary that succeeds
            # cleanly leaves its unused fallback BLOCKED forever and the parent can never
            # merge. Absent marker (every existing arm) => unchanged.
            if is_alternative_pending(child):
                continue
            # For regular action nodes: BLOCKED means still retrying -> not ready
            if child.status not in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                return False
        
        return True

    def should_create_merge_node(self, graph: IdeaDag, node_id: str) -> bool:
        if not self._cfg.policy.enable_recursive_merge:
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

    def select_winner(
        self, graph: IdeaDag, node_id: str, race_group: str
    ) -> Optional[str]:
        """Resolve one race group under ``node_id`` to a single winning member id.

        Group membership is read off ``node.details["race_groups"][race_group]`` — the
        registry ``alternative_branch.link_race_groups`` writes onto the shared ancestor (plus
        the structurally inferred one, only when that second registry is explicitly opted into;
        see :meth:`_race_registry`) — NOT off graph topology. The alternative was to mint the merge node as a multi-parent
        child of the race leaves via ``IdeaDag.merge_nodes``, and two existing consumers still
        assume single-parent topology: :meth:`merge`'s recursive-upward step reads
        ``node.parent_id`` (``None`` on a ``parent_ids``-only node, so the walk silently
        stops), and :meth:`should_create_merge_node`'s dedup check only scans the common
        ancestor's direct children, so it would never see such a merge node and would mint a
        second, duplicate one. Tagging the ancestor sidesteps both without touching either.

        The chain is mechanical end to end — no LLM judgment at any step, deliberately more
        conservative than adding a judge-score tier, because this repo's own F33/F35 work
        found step-confidence judge scores anti-calibrated:

        1. A DONE member whose step contract verified a measurable DATUM wins outright.
        2. Ties among those go to the earliest completion. The graph records no per-node
           completion timestamp, so the key is (recorded step index, declared order) — within
           one parallel batch every member shares a step and declared order decides.
        3. Otherwise any DONE member beats any FAILED/BLOCKED one, first by declared order.
        4. No usable member at all -> ``None``, and :meth:`merge` aggregates everything
           exactly as it does today.

        Never mutates and never raises.
        """
        node = graph.get_node(node_id)
        if not node:
            return None
        registry = self._race_registry(node)
        member_ids = registry.get(race_group)
        if not isinstance(member_ids, list) or len(member_ids) < 2:
            return None

        root = graph.get_node(graph.root_id())
        mandate = ""
        if root and isinstance(root.details, dict):
            mandate = str(root.details.get("mandate") or "")

        datum_verified: List[Any] = []
        any_done: List[Any] = []
        for order, member_id in enumerate(member_ids):
            member = graph.get_node(member_id)
            if member is None or member.status != IdeaNodeStatus.DONE:
                continue
            any_done.append((order, member_id))
            if self._datum_verified(member, mandate):
                step = member.details.get(RACE_COMPLETED_STEP)
                step = step if isinstance(step, int) else 1 << 30
                datum_verified.append((step, order, member_id))

        if datum_verified:
            return min(datum_verified)[2]
        if any_done:
            return min(any_done)[1]
        return None

    @staticmethod
    def _datum_verified(node: IdeaNode, mandate: str) -> bool:
        """Did this member actually verify a measurable datum (F35's distinction)?

        A satisfied-on-subject-tokens-only contract proves the member opened a page matching
        the words of its own goal, which every wrong-but-plausible route also manages — so it
        is not enough to win a race. Fails closed: an unevaluable member simply does not win
        step 1 and falls through to the DONE-beats-the-rest tier.
        """
        try:
            from agent.app.idea_policies.contract_satisfaction import evaluate_step_contract

            verdict = evaluate_step_contract(node, mandate)
        except Exception:  # noqa: BLE001. A merge-time filter must never crash a run
            return False
        return bool(verdict.applicable and verdict.satisfied and verdict.datum_verified)

    def _race_registry(self, node: IdeaNode) -> Dict[str, Any]:
        """Race groups under ``node`` this policy is willing to resolve.

        The authored registry alone unless
        ``merge_race_winner_selection_includes_inferred_groups_enabled`` is on, in which case
        the structurally inferred one (``race_groups_inferred``) is unioned in behind it —
        authored membership wins any label collision, being the better evidence. Default OFF
        keeps inferred groups pure instrumentation: they are populated, logged and visible in
        report captures while ``merge()`` stays byte-identical.
        """
        registry = node.details.get(RACE_GROUPS)
        registry = dict(registry) if isinstance(registry, dict) else {}
        if not self._cfg.merge.race_winner_selection_includes_inferred_groups_enabled:
            return registry
        inferred = node.details.get(RACE_GROUPS_INFERRED)
        if isinstance(inferred, dict):
            for label, member_ids in inferred.items():
                registry.setdefault(label, member_ids)
        return registry

    def _race_excluded_ids(self, graph: IdeaDag, node_id: str) -> set:
        """Race-group members under ``node_id`` that lost, so ``merge`` can drop them.

        No-op (empty set) unless ``merge_race_winner_selection_enabled`` is on AND this node
        carries a race registry, which is why every existing arm's ``merge()`` is unchanged.
        Losers are marked SKIPPED here rather than cancelled mid-flight: cancelling would mean
        restructuring the ``asyncio.gather``/``Semaphore`` dispatch loop, so a race cell pays
        for its discarded loser. That cost is real and is reported, not averaged away.
        """
        if not self._cfg.merge.race_winner_selection_enabled:
            return set()
        node = graph.get_node(node_id)
        if not node:
            return set()
        registry = self._race_registry(node)
        if not registry:
            return set()

        excluded = set()
        for label, member_ids in registry.items():
            if not isinstance(member_ids, list):
                continue
            winner = self.select_winner(graph, node_id, label)
            if winner is None:
                continue
            for member_id in member_ids:
                if member_id == winner:
                    continue
                excluded.add(member_id)
                loser = graph.get_node(member_id)
                if loser is None:
                    continue
                loser.details[RACE_LOSER] = True
                if loser.status not in (IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                    loser.status = IdeaNodeStatus.SKIPPED
            self._logger.info(
                f"[RACE] group {label!r} under {node_id[:8]} resolved to winner "
                f"{winner[:8]}; {len(member_ids) - 1} losing route(s) dropped from synthesis"
            )
        return excluded

    def merge(self, graph: IdeaDag, node_id: str, recursive: bool = True) -> Dict[str, Any]:
        node = graph.get_node(node_id)
        if not node:
            return {}

        # Winner-selection pre-filter: replace each race group's N members with just its
        # winner BEFORE the existing aggregation runs, so synthesis sees one route rather
        # than N partly-contradictory ones. Empty unless the flag is on and this node owns a
        # race registry; non-race siblings are never touched.
        race_excluded = self._race_excluded_ids(graph, node_id)

        merged: List[Dict[str, Any]] = []
        success_count = 0
        failed_count = 0
        blocked_count = 0
        skipped_count = 0
        
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            if child_id in race_excluded:
                continue
            # A fallback that was never needed carries no result and describes work the run
            # deliberately did not do; feeding it to synthesis would only add a phantom
            # "missing" branch to the summary.
            if is_alternative_pending(child):
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
                    # The join's half of the resolved-value channel: each branch hands the
                    # merge the structured datum it discovered, instead of the merge having
                    # to re-read it out of a concatenated result blob. None whenever the
                    # channel is off, or the branch produced no waypoint.
                    "waypoint": (
                        child.details.get(DetailKey.WAYPOINT.value)
                        if self._cfg.engine.resolved_value_channel_enabled else None
                    ),
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
