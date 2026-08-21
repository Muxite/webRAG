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
    RACE_GROUPS_INFERRED_TIERS,
    RACE_LOSER,
    is_alternative_pending,
)


#: Result fields every action echoes back whatever it found: the request parameters plus
#: bookkeeping. A child whose result holds only these has produced nothing to synthesize.
#: ``count`` is SEARCH's *requested* result count (non-zero even when the provider returned
#: nothing) and ``url``/``query``/``intent`` are the node's own inputs read back, so none of
#: them is evidence of a finding. Booleans are ignored wholesale for the same reason.
_ECHOED_RESULT_KEYS = frozenset({
    "action",
    "success",
    "error",
    "error_type",
    "retryable",
    "context",
    "node_id",
    "timestamp",
    "query",
    "intent",
    "count",
    "url",
    "urls_visited",
    "link",
    "title",
    "content_total_chars",
    "content_is_truncated",
})


#: Fraction of the goal's content words a candidate payload must repeat before it counts as
#: evidence that the goal was addressed. Containment, not Jaccard: page content dwarfs the goal
#: phrase, so a union denominator would drive every real page below any usable threshold.
_GOAL_RELEVANCE_MIN_OVERLAP = 0.5

#: Substantive-child count stamped onto a merge node when it is minted, so a later retry can
#: tell "new evidence arrived since that merge was skipped" from "same evidence, same answer".
#: Only written while ``merge_retry_after_skip_enabled`` is on; absent => no retry.
SUBSTANTIVE_CHILD_COUNT_AT_CREATION = "substantive_child_count_at_creation"

#: Result-item fields a SEARCH hit carries its finding in. ``description`` is what
#: ``connector_search`` normalizes every provider's snippet field to; ``snippet`` is the raw
#: provider spelling, kept for payloads that reach here unnormalized.
_RESULT_ITEM_TEXT_KEYS = ("title", "description", "snippet")


def _goal_tokens(text: Any) -> set:
    """Content words of ``text``, lowercased, punctuation-split, stop-word-ish shorts dropped."""
    raw = "".join(ch if ch.isalnum() else " " for ch in str(text or "").lower())
    return {t for t in raw.split() if len(t) > 2}


def _relevance_overlap(goal_tokens: set, candidate_tokens: set) -> float:
    """How much of the goal the candidate covers, in [0, 1]. Empty goal => 0.0."""
    if not goal_tokens:
        return 0.0
    return len(goal_tokens & candidate_tokens) / len(goal_tokens)


#: Where a merge's goal-addressing evidence came from -- see :func:`goal_evidence_provenance`.
GOAL_EVIDENCE_NONE = "none"
GOAL_EVIDENCE_SNIPPET = "snippet"
GOAL_EVIDENCE_DIRECT = "direct"


def _direct_text(payload: Any, _depth: int = 0) -> str:
    """Everything a result says IN ITS OWN VOICE, i.e. minus its SEARCH hit list.

    ``results`` is dropped because a search hit is a third party's snippet about a page this
    run never opened; the echoed request keys are dropped for the reason
    :data:`_ECHOED_RESULT_KEYS` documents (they are the node's own inputs read back, so a goal
    restated in a ``query`` would otherwise self-certify). Everything else -- ``content``, a
    tool payload, a nested merge's ``summary``/``key_findings`` -- counts.
    """
    if _depth > 3:
        return ""
    if isinstance(payload, bool):
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (list, tuple)):
        return " ".join(filter(None, (_direct_text(v, _depth + 1) for v in payload)))
    if not isinstance(payload, dict):
        return ""
    from agent.app.idea_policies.action_constants import ActionResultKey

    parts = []
    for key, value in payload.items():
        if str(key) in _ECHOED_RESULT_KEYS or str(key) == ActionResultKey.RESULTS.value:
            continue
        parts.append(_direct_text(value, _depth + 1))
    return " ".join(p for p in parts if p)


def goal_evidence_provenance(original_goal: Any, merged_results: Any) -> str:
    """Was the goal-relevant evidence FETCHED, or only glimpsed in unvisited search snippets?

    The 2026-08-21 A/B watched qwen2.5:14b elect a survivor from search-result snippets alone
    -- right entity, keystone datum never obtained, ``goal_achieved: true`` anyway, in both
    prompt-ordering arms. The distinguishing signal already exists in the merged payloads and
    costs nothing to read: a snippet lives inside a SEARCH result's ``results`` list, while
    fetched content lives in the result's own fields.

    Three-way, deliberately: ``direct`` (something the run actually obtained addresses the
    goal), ``snippet`` (only unvisited hits do), ``none`` (nothing clears
    :data:`_GOAL_RELEVANCE_MIN_OVERLAP`). Only ``snippet`` is a positive detection -- ``none``
    is equally consistent with a correct paraphrase-only answer that the overlap bar missed,
    so callers must not read it as a negative verdict.

    Fails open toward ``direct``: a nested merge hands up a synthesis rather than the pages
    behind it, so its provenance is unknowable here and is credited rather than blamed.
    """
    goal_tokens = _goal_tokens(original_goal)
    if not goal_tokens or not isinstance(merged_results, (list, tuple)):
        return GOAL_EVIDENCE_NONE

    from agent.app.idea_policies.action_constants import ActionResultKey

    found_snippet = False
    for item in merged_results:
        if not isinstance(item, dict):
            continue
        result = item.get("result")
        if not result:
            continue
        if SimpleMergePolicy._addresses_goal(goal_tokens, _direct_text(result)):
            return GOAL_EVIDENCE_DIRECT
        if not isinstance(result, dict):
            continue
        hits = result.get(ActionResultKey.RESULTS.value)
        if isinstance(hits, (list, tuple)) and any(
            SimpleMergePolicy._result_item_addresses_goal(goal_tokens, hit) for hit in hits
        ):
            found_snippet = True
    return GOAL_EVIDENCE_SNIPPET if found_snippet else GOAL_EVIDENCE_NONE


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
        
        # An existing merge node under this parent normally settles the question: the
        # synthesis either already ran or was deliberately skipped, and a second one would
        # re-pay for the same children. The single exception is a skipped merge that new
        # evidence has since outdated -- see :meth:`_merge_retry_allowed`.
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor
        existing_merges = [
            child
            for child in (graph.get_node(child_id) for child_id in node.children)
            if child and NodeDetailsExtractor.is_merge_action(child.details)
        ]
        if existing_merges and not self._merge_retry_allowed(graph, node, existing_merges):
            return False


        # Check if all children are ready
        if not self.are_children_ready_to_merge(graph, node_id):
            return False

        if self._cfg.merge.require_substantive_children_enabled:
            substantive = self._substantive_child_count(graph, node)
            if substantive < 2:
                self._logger.info(
                    f"[MERGE] no synthesis for {node_id[:8]}: {len(node.children)} children "
                    f"but only {substantive} carry content -- nothing to combine"
                )
                return False
        return True

    def _merge_retry_allowed(
        self, graph: IdeaDag, node: IdeaNode, existing_merges: List[IdeaNode]
    ) -> bool:
        """May this parent mint ANOTHER merge node after an earlier one was skipped?

        :meth:`should_create_merge_node` used to answer no unconditionally -- both arms of its
        ``merge_should_skip`` test returned ``False`` -- so the first merge node a parent ever
        got locked that parent out of merging for the rest of the run. Including a merge node
        skipped precisely BECAUSE the goal was judged unmet, which is the one case where the
        parent has a reason to try again once a sibling brings back more.

        Retry is permitted only when every merge under this parent was skipped AND the
        substantive-child count has GROWN since the most recent of those skips: re-running the
        same synthesis over the same evidence can only re-derive the same verdict at the price
        of another LLM call. The baseline is the count stamped at creation time; a merge node
        carrying no stamp (anything minted while the flag was off) fails closed and keeps
        today's lockout, so this never turns into a blanket unconditional retry.

        Nothing here mutates: the skipped node keeps its status and details as the run's audit
        record of what was tried, and a permitted retry mints a fresh sibling merge node
        through the ordinary :meth:`create_merge_node` path.
        """
        if not self._cfg.merge.retry_after_skip_enabled:
            return False

        baseline = -1
        for child in existing_merges:
            skipped = bool(child.details.get("merge_should_skip", False)) or (
                child.status == IdeaNodeStatus.SKIPPED
            )
            if not skipped:
                return False
            recorded = child.details.get(SUBSTANTIVE_CHILD_COUNT_AT_CREATION)
            if not isinstance(recorded, int) or isinstance(recorded, bool):
                return False
            baseline = max(baseline, recorded)

        current = self._substantive_child_count(graph, node)
        if current <= baseline:
            return False
        self._logger.info(
            f"[MERGE] retrying synthesis for {node.node_id[:8]}: {current} substantive children "
            f"now vs {baseline} when the skipped merge was created -- new evidence to combine"
        )
        return True

    def _substantive_child_count(self, graph: IdeaDag, node: IdeaNode) -> int:
        """How many of ``node``'s children actually produced something to synthesize.

        Structural child count is the wrong question for "is a merge worth an LLM call?".
        The 2026-08-21 diagnostic found 4 of 7 real merge calls synthesizing a SINGLE
        content-bearing child, the other "child" being a Serper-403 search the connector
        recorded as ``success: true, content: "", url: None`` -- success-shaped, empty, and
        therefore invisible to every status check. Combining one source with nothing is
        strictly worse than using that source directly, and the merge's ``goal_achieved``
        can terminate the branch on it.

        A parked A->B fallback is skipped for the same reason :meth:`merge` skips it: it
        describes work the run deliberately did not do.
        """
        from agent.app.idea_policies.action_constants import NodeDetailsExtractor

        count = 0
        for child_id in node.children:
            child = graph.get_node(child_id)
            if not child or is_alternative_pending(child):
                continue
            if NodeDetailsExtractor.is_merge_action(child.details):
                continue
            result = child.details.get(DetailKey.ACTION_RESULT.value)
            if result is None:
                result = child.details.get(DetailKey.ACTION_RESULTS.value)
            if self._payload_is_substantive(result):
                count += 1
        return count

    @classmethod
    def _payload_is_substantive(cls, result: Any) -> bool:
        """Does this action result carry any finding, as opposed to only its own inputs?

        Deliberately generous -- it answers "is there ANY non-echoed, non-empty field" rather
        than naming the finding keys of each action type, so an action shape this gate has
        never seen (or a future one) counts as substantive and merges exactly as today.
        """
        if isinstance(result, (list, tuple)):
            return any(cls._payload_is_substantive(item) for item in result)
        if not isinstance(result, dict):
            return bool(result)
        for key, value in result.items():
            if str(key) in _ECHOED_RESULT_KEYS or isinstance(value, bool):
                continue
            if isinstance(value, str):
                if value.strip():
                    return True
            elif isinstance(value, (list, tuple, dict, set)):
                if len(value):
                    return True
            elif value:
                return True
        return False

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
        # Only stamped while retry is enabled, so a flag-off run's node details stay exactly
        # as they were and a flag-on run never reads a baseline it did not itself record.
        if self._cfg.merge.retry_after_skip_enabled:
            merge_details[SUBSTANTIVE_CHILD_COUNT_AT_CREATION] = self._substantive_child_count(
                graph, parent
            )
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

        **Tier 2 inferred groups are never consumable here**, whatever that flag says. The
        2026-08-21 probe audit measured tier 2's live precision at 50% — half its registrations
        were a search and its own following visit, one chain, not a race — and consuming a
        wrong group DISCARDS correct findings from synthesis. Tier 2 keeps being written to
        ``race_groups_inferred`` for its instrumentation value; earning merge consumption back
        is a future cycle's job, gated on a probe that shows precision it has not shown yet.
        """
        registry = node.details.get(RACE_GROUPS)
        registry = dict(registry) if isinstance(registry, dict) else {}
        if not self._cfg.merge.race_winner_selection_includes_inferred_groups_enabled:
            return registry
        inferred = node.details.get(RACE_GROUPS_INFERRED)
        tiers = node.details.get(RACE_GROUPS_INFERRED_TIERS)
        tiers = tiers if isinstance(tiers, dict) else {}
        if isinstance(inferred, dict):
            for label, member_ids in inferred.items():
                if tiers.get(label) != 1:
                    continue
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
        """Mechanical pre-check: did the children gather anything that even ADDRESSES the goal?

        This runs before any merge node's own LLM call and, because :meth:`merge` recurses to
        root, its verdict lands on the ROOT's ``goal_achieved`` -- which ``idea_finalize`` reads
        FIRST, ahead of the LLM-verified merge verdicts. A false positive here therefore
        short-circuits finalization for the whole run.

        It used to produce them two ways: ``goal in content`` demanded the page repeat the
        goal's exact phrasing verbatim (real pages essentially never do, so the check was
        near-unsatisfiable), while ``len(results) > 0`` accepted ANY non-empty search-result
        list, off-topic hits included -- a tautology on the far side. Both are now the same
        containment-overlap test on the goal's content words, so an unrelated payload fails and
        a paraphrase of the goal passes.
        """
        original_goal = node.details.get(DetailKey.GOAL.value) or node.details.get(DetailKey.ORIGINAL_GOAL.value) or node.details.get(DetailKey.INTENT.value)
        if not original_goal:
            return True

        from agent.app.idea_policies.action_constants import ActionResultKey

        goal_tokens = _goal_tokens(original_goal)
        if not goal_tokens:
            # A goal with no content words (punctuation, digits, one-letter tokens) gives the
            # overlap test nothing to measure; fall back to the old permissive answer rather
            # than failing every merge under it.
            return True

        has_relevant_content = False
        for result_item in merged_results:
            result = result_item.get("result")
            if not isinstance(result, dict):
                continue

            content = result.get(ActionResultKey.CONTENT.value) or ""
            query = result.get(ActionResultKey.QUERY.value) or ""
            results = result.get(ActionResultKey.RESULTS.value) or []

            if self._addresses_goal(goal_tokens, content) or self._addresses_goal(goal_tokens, query):
                has_relevant_content = True
                break

            if isinstance(results, list) and any(
                self._result_item_addresses_goal(goal_tokens, item) for item in results
            ):
                has_relevant_content = True
                break

        return has_relevant_content

    @staticmethod
    def _addresses_goal(goal_tokens: set, text: Any) -> bool:
        if not text:
            return False
        return _relevance_overlap(goal_tokens, _goal_tokens(text)) >= _GOAL_RELEVANCE_MIN_OVERLAP

    @classmethod
    def _result_item_addresses_goal(cls, goal_tokens: set, item: Any) -> bool:
        """One SEARCH hit, held to the same overlap bar as page content.

        :meth:`_payload_is_substantive` is deliberately NOT reused as a pre-gate here: it reads
        ``title``/``url`` as echoed request parameters, which is right for an action result and
        wrong for a search hit, where the title IS the finding. An empty hit scores 0 overlap
        and fails anyway, so the pre-gate would only misclassify.
        """
        if not isinstance(item, dict):
            return cls._addresses_goal(goal_tokens, item)
        text = " ".join(str(item.get(key) or "") for key in _RESULT_ITEM_TEXT_KEYS)
        return cls._addresses_goal(goal_tokens, text)
