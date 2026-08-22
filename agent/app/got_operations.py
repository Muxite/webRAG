from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaDag, IdeaNode
    from agent.app.agent_io import AgentIO
    from agent.app.idea_memory import MemoryManager

from agent.app.idea_policies.base import DetailKey, IdeaNodeStatus
from agent.app.idea_policies.config import IdeaConfig
from agent.app.idea_policies.confidence_early_exit import load_rule as load_early_exit_rule
from agent.app.plan_library.retrieval import similarity_from_distance

_logger = logging.getLogger(__name__)


# Reason-before-answer variant of the shipped ``got_reexpand_followup_system_prompt``,
# selected under ``got_reexpand_followup_reason_first_enabled``. Only the
# ``needs_followup`` / ``reason`` pair is swapped; every other byte is identical.
#
# Copied from the SETTINGS value, NOT from the inline default a few lines below in
# ``check_needs_followup`` -- that default is a 347-char fossil that never runs, because
# ``settings.get(key, default)`` prefers the JSON's 614-char version, and the extra text
# carries real behavioural constraints. Deriving the variant from the fossil would have
# shipped a prompt the engine has never sent.
#
# promptbench v2 (2026-08-19): this is the largest measured effect in the run --
# pooled A2-A1 = +0.196, CI [+0.080, +0.312], permutation p = 0.0064. Opt-in, default OFF;
# end-to-end transfer is unmeasured.
_FOLLOWUP_REASON_FIRST_SYSTEM_PROMPT = (
    "You are a follow-up detector in a Graph-of-Thought research system. A leaf task "
    "has just completed and produced a result. Decide whether that result reveals a "
    "GENUINE, concrete follow-up investigation that is required to satisfy the parent "
    "goal and is not already covered by existing sibling tasks. Only answer true when "
    "the resolved content names a specific new entity, page, or question that must be "
    "investigated next (e.g. a disambiguation survivor that points to a further "
    "target). Answer false for vague, speculative, or already-answered follow-ups. "
    "Return JSON: {\"reason\": string, \"needs_followup\": boolean}."
)


class GoTOperations:

    def __init__(self, settings: Dict[str, Any], io: AgentIO, memory_manager: Optional[MemoryManager] = None):
        self.settings = settings
        self._cfg = IdeaConfig.from_settings(settings)
        self.io = io
        self.memory_manager = memory_manager
        self._dead_end_count = 0
        self._early_exit_count = 0

    async def embed_thought(
        self,
        node_id: str,
        title: str,
        goal: str,
        action_type: Optional[str] = None,
        parent_id: Optional[str] = None,
        depth: int = 0,
    ) -> bool:
        if not self._cfg.got.embed_on_create:
            return False
        if not self.memory_manager:
            return False

        content_parts = [f"Thought: {title}"]
        if goal:
            content_parts.append(f"Goal: {goal}")
        if action_type:
            content_parts.append(f"Action: {action_type}")
        content = "\n".join(content_parts)

        metadata = {
            "memory_type": "internal_thought",
            "step_type": "thought_node",
            "depth": str(depth),
        }
        if parent_id:
            metadata["parent_id"] = parent_id

        return await self.memory_manager.write_memory(
            content=content,
            node_id=node_id,
            node_title=title,
            action_type=action_type,
            metadata=metadata,
            memory_type="internal_thought",
        )

    async def embed_children(self, graph: IdeaDag, parent_id: str) -> int:
        if not self._cfg.got.embed_on_create:
            return 0
        parent = graph.get_node(parent_id)
        if not parent:
            return 0

        count = 0
        for child_id in parent.children:
            child = graph.get_node(child_id)
            if not child:
                continue
            action = child.details.get(DetailKey.ACTION.value)
            goal = (
                child.details.get(DetailKey.GOAL.value)
                or child.details.get(DetailKey.ORIGINAL_GOAL.value)
                or child.title
            )
            depth = graph.depth(child_id)
            ok = await self.embed_thought(
                node_id=child_id,
                title=child.title,
                goal=goal,
                action_type=action,
                parent_id=parent_id,
                depth=depth,
            )
            if ok:
                count += 1

        if count > 0:
            _logger.debug(f"[GoT:EMBED] Embedded {count} child thoughts for parent {parent_id}")
        return count

    async def check_needs_followup(
        self,
        graph: IdeaDag,
        node_id: str,
        model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Ask whether a completed leaf's resolved content reveals a genuine follow-up.

        Returns a structured verdict the engine reads to decide whether to
        re-expand the leaf into new children (rather than rewriting the node in
        place).
        Returns ``{"needs_followup": bool, "reason": str}`` (or ``None`` on
        error / when the flag is off). Gated by ``got.reexpand_enabled`` so the
        default-off path never issues an LLM call.
        """
        if not self._cfg.got.reexpand_enabled:
            return None

        node = graph.get_node(node_id)
        if not node:
            return None

        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            return None
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        if not ActionResultExtractor.is_success(result):
            return None

        # Compact the resolved content so the check stays cheap.
        content = (
            result.get("content")
            or result.get("content_full")
            or ""
        )
        if isinstance(content, str) and len(content) > 3000:
            content = content[:3000] + "... [truncated]"
        results_summary = result.get("results")
        if isinstance(results_summary, list):
            results_summary = results_summary[:5]

        root = graph.get_node(graph.root_id())
        mandate = ""
        if root and isinstance(root.details, dict):
            mandate = str(root.details.get("mandate") or "")[:1500]

        goal = (
            node.details.get(DetailKey.GOAL.value)
            or node.details.get(DetailKey.ORIGINAL_GOAL.value)
            or node.title
        )
        parent_goal = node.details.get(DetailKey.PARENT_GOAL.value) or ""

        sibling_titles: List[str] = []
        if node.parent_id:
            parent = graph.get_node(node.parent_id)
            if parent:
                for cid in parent.children:
                    if cid == node_id:
                        continue
                    sib = graph.get_node(cid)
                    if sib:
                        sibling_titles.append(sib.title[:80])

        system_prompt = self.settings.get(
            "got_reexpand_followup_system_prompt",
            "You are a follow-up detector in a Graph-of-Thought research system. A leaf "
            "task has just completed and produced a result. Decide whether that result "
            "reveals a GENUINE, concrete follow-up investigation required to satisfy the "
            "parent goal and not already covered by existing sibling tasks. Return JSON: "
            "{\"needs_followup\": boolean, \"reason\": string}.",
        )
        # Opt-in reason-before-answer ordering. Already doubly dormant: this whole path
        # is gated by got_reexpand_enabled.
        if self._cfg.got.reexpand_followup_reason_first_enabled:
            system_prompt = _FOLLOWUP_REASON_FIRST_SYSTEM_PROMPT
        user_content = json.dumps({
            "mandate": mandate,
            "parent_goal": parent_goal,
            "completed_task_title": node.title,
            "completed_task_goal": goal,
            "action": node.details.get(DetailKey.ACTION.value),
            "resolved_content": content,
            "resolved_results": results_summary,
            "existing_sibling_tasks": sibling_titles,
        }, ensure_ascii=True, default=str)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        check_model = model_name or self._cfg.evaluation.model or None
        temperature = self._cfg.got.reexpand_temperature

        payload = self.io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=check_model,
            temperature=temperature,
        )

        try:
            response = await self.io.query_llm_with_fallback(
                payload,
                model_name=check_model,
                fallback_model=self._cfg.generation.fallback_model,
                timeout_seconds=self._cfg.timeouts.llm,
            )
            if not response:
                return None
            data = json.loads(response)
            verdict = {
                "needs_followup": bool(data.get("needs_followup", False)),
                "reason": str(data.get("reason", "")),
            }
            _logger.info(
                f"[GoT:REEXPAND] Follow-up check for node {node_id}: "
                f"needs_followup={verdict['needs_followup']} ({verdict['reason'][:80]})"
            )
            return verdict
        except Exception as exc:
            _logger.warning(f"[GoT:REEXPAND] Follow-up check failed for node {node_id}: {exc}")
            return None

    async def judge_step_confidence(
        self,
        graph: IdeaDag,
        node_id: str,
        model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Emit a decorrelated per-step LLM-judge confidence for a completed leaf.

        Opt-in instrumentation (gated by ``got.step_confidence_judge_enabled``): a
        lightweight judge estimates, from ONLY what is visible at this point in the
        trajectory (the mandate + this node's resolved content), how confident it is
        that the step's output is correct and on-track. It is deliberately never shown
        the grep validators nor the ground-truth answer, so the resulting score is a
        genuinely *decorrelated* per-step signal — a real substrate for the E-valuator
        sequential-stopping pilot, unlike ``validation.grep_validations`` (which is what
        *computes* the final pass/fail label). This method never mutates the graph and
        never raises; it returns ``{"confidence": float, "reason": str}`` or ``None``.
        """
        if not self._cfg.got.step_confidence_judge_enabled:
            return None

        node = graph.get_node(node_id)
        if not node:
            return None

        result = node.details.get(DetailKey.ACTION_RESULT.value)
        if not isinstance(result, dict):
            return None
        from agent.app.idea_policies.action_constants import ActionResultExtractor
        if not ActionResultExtractor.is_success(result):
            return None

        # Compact the resolved content so the judge stays cheap.
        content = (
            result.get("content")
            or result.get("content_full")
            or ""
        )
        if isinstance(content, str) and len(content) > 3000:
            content = content[:3000] + "... [truncated]"
        results_summary = result.get("results")
        if isinstance(results_summary, list):
            results_summary = results_summary[:5]

        # Some leaf kinds (merge/think/verify/save) write their real output under keys this
        # judge doesn't read (``synthesized``, ``thinking_content``, ``verdict``, ``count``, ...).
        # Judging "nothing visible" produced a confidently-wrong signal instead of no signal —
        # see CONFIDENCE_JUDGE_MISCALIBRATION.md. Decline rather than guess from an empty prompt.
        if not content and not results_summary:
            return None

        root = graph.get_node(graph.root_id())
        mandate = ""
        if root and isinstance(root.details, dict):
            mandate = str(root.details.get("mandate") or "")[:1500]

        goal = (
            node.details.get(DetailKey.GOAL.value)
            or node.details.get(DetailKey.ORIGINAL_GOAL.value)
            or node.title
        )

        system_prompt = (
            "You are a step-level verifier in a Graph-of-Thought research system. A single "
            "sub-task has just produced an output. Estimate, on a 0-1 scale, how confident you "
            "are that this step's output is CORRECT and ON-TRACK toward the overall task, using "
            "ONLY what is visible below. You do NOT know the ground-truth answer and must not "
            "assume one; judge from the resolved content alone. Return JSON: "
            "{\"confidence\": number between 0 and 1, \"reason\": string}."
        )
        user_content = json.dumps({
            "task": mandate,
            "sub_task_goal": goal,
            "sub_task_title": node.title,
            "action": node.details.get(DetailKey.ACTION.value),
            "resolved_content": content,
            "resolved_results": results_summary,
        }, ensure_ascii=True, default=str)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        judge_model = model_name or self._cfg.got.step_confidence_judge_model or self._cfg.evaluation.model or None
        temperature = self._cfg.got.step_confidence_judge_temperature

        payload = self.io.build_llm_payload(
            messages=messages,
            json_mode=True,
            model_name=judge_model,
            temperature=temperature,
        )

        try:
            response = await self.io.query_llm_with_fallback(
                payload,
                model_name=judge_model,
                fallback_model=self._cfg.generation.fallback_model,
                timeout_seconds=self._cfg.timeouts.llm,
            )
            if not response:
                return None
            data = json.loads(response)
            raw = data.get("confidence")
            try:
                confidence = float(raw)
            except (TypeError, ValueError):
                return None
            confidence = max(0.0, min(1.0, confidence))
            verdict = {"confidence": confidence, "reason": str(data.get("reason", ""))}
            _logger.info(
                f"[GoT:STEPCONF] Step confidence for node {node_id}: "
                f"{confidence:.3f} ({verdict['reason'][:80]})"
            )
            return verdict
        except Exception as exc:
            _logger.warning(f"[GoT:STEPCONF] Confidence judge failed for node {node_id}: {exc}")
            return None

    def _adaptive_dedup_threshold(self, graph: IdeaDag) -> float:
        """
        Pick a similarity cutoff based on graph density. Sparse graphs tolerate
        more variety (lower cutoff → less aggressive dedup); dense ones tighten
        the cutoff to keep growth in check. Clamped to [0.75, 0.92].

        :param graph: Current DAG.
        :returns: Threshold in [0.75, 0.92].
        """
        if not self._cfg.got.adaptive_policies:
            return self._cfg.got.dedup_similarity_threshold
        floor = self._cfg.got.dedup_threshold_min
        ceil = self._cfg.got.dedup_threshold_max
        # Use sibling fanout as a density proxy: max children across non-leaf nodes.
        fanout = 0
        for n in graph.iter_depth_first():
            if n.children:
                fanout = max(fanout, len(n.children))
        # 0 siblings → loose (floor); >=8 siblings → tight (ceil); linear in between.
        ratio = min(1.0, fanout / 8.0)
        return round(floor + ratio * (ceil - floor), 3)

    async def is_duplicate_thought(
        self,
        candidate_title: str,
        candidate_goal: str,
        graph: IdeaDag,
    ) -> Tuple[bool, Optional[str]]:
        if not self._cfg.got.dedup_enabled:
            return False, None
        if not self.memory_manager:
            return False, None

        threshold = self._adaptive_dedup_threshold(graph)
        n_query = self._cfg.got.dedup_max_query

        query = f"{candidate_title} {candidate_goal}"
        try:
            memories = await self.memory_manager.retrieve_relevant_memories(
                query=query,
                n_results=n_query,
                memory_type="internal_thought",
            )
            if not memories:
                return False, None

            space = getattr(self.memory_manager, "distance_space", "cosine")
            for mem in memories:
                distance = mem.get("distance", 1.0)
                if isinstance(distance, (int, float)):
                    # BEHAVIOUR CHANGE (2026-08-20, ASSUMPTION_AUDIT.md T1-1): this used to be
                    # a bare ``1.0 - distance``. The ``mem_*`` collections were created with no
                    # metadata, so chroma's default ``l2`` space returned the SQUARED euclidean
                    # distance (``2 - 2cos`` for unit-norm embeddings) and the computed value was
                    # ``2s - 1``, not the cosine ``s``: the shipped 0.85 threshold really demanded
                    # cosine 0.925. Converting correctly makes dedup FIRE MORE OFTEN, so any
                    # measurement taken through this path before that date is not comparable.
                    similarity = similarity_from_distance(distance, space)
                    if similarity >= threshold:
                        existing_node_id = (mem.get("metadata") or {}).get("node_id", "unknown")
                        _logger.info(
                            f"[GoT:DEDUP] Candidate '{candidate_title[:40]}' is duplicate of node {existing_node_id} "
                            f"(similarity={similarity:.3f} >= {threshold})"
                        )
                        return True, existing_node_id

        except Exception as exc:
            _logger.warning(f"[GoT:DEDUP] Dedup check failed: {exc}")

        return False, None

    async def filter_duplicate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        graph: IdeaDag,
    ) -> List[Dict[str, Any]]:
        if not self._cfg.got.dedup_enabled:
            return candidates
        if not self.memory_manager:
            return candidates

        filtered = []
        dedup_count = 0

        for candidate in candidates:
            title = candidate.get("title", "")
            details = candidate.get("details", {})
            goal = (
                details.get(DetailKey.GOAL.value)
                or details.get(DetailKey.ORIGINAL_GOAL.value)
                or title
            )

            is_dup, existing_id = await self.is_duplicate_thought(title, goal, graph)
            if is_dup:
                dedup_count += 1
                continue
            filtered.append(candidate)

        if dedup_count > 0:
            _logger.info(f"[GoT:DEDUP] Filtered {dedup_count} duplicate candidates out of {len(candidates)}")

        if filtered:
            return filtered
        # BEHAVIOUR CHANGE (2026-08-20, Cycle 18): the all-flagged batch used to fall back to
        # ``candidates[:1]``, collapsing a whole sibling batch to one candidate. The live A/B
        # measured that as -0.157 on overall_score (p=0.0007, n=47), driven by chains whose
        # multi-hop plan the truncation destroyed: 58% of firing batches flagged EVERY
        # candidate, which is a maximally-uncertain dedup verdict rather than evidence that the
        # plan is redundant. Flagging everything now means "this similarity judgement is not
        # usable here", so the batch passes through untouched. Partial flags still filter.
        _logger.info(
            f"[GoT:DEDUP] All {len(candidates)} candidates flagged; keeping the batch unfiltered"
        )
        return candidates

    @staticmethod
    def _beam_pool_score(node: "IdeaNode") -> float:
        """The value the beam's spread should measure for ``node``.

        ``got_beam_spread_uses_raw_score_enabled`` only. ``node.score`` is the post-cap number
        every other consumer reads and is left alone; here the judge's own ``raw_score`` is
        preferred where one exists, so a pool that clusters on
        ``evaluation_no_action_result_score_cap`` does not read as convergence. A graph
        recorded before 9b710b91, an evaluation that errored, and the base-score shortcut (which
        returns without calling the judge, so ``raw_score`` is None) all fall back to
        ``node.score`` rather than inventing an opinion.
        """
        evaluation = node.details.get(DetailKey.EVALUATION.value)
        if isinstance(evaluation, dict):
            raw = evaluation.get("raw_score")
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return float(raw)
        return float(node.score)

    def compute_dynamic_beam_width(self, graph: IdeaDag) -> int:
        if not self._cfg.got.dynamic_beam_enabled:
            return self._cfg.engine.max_branching

        beam_min = self._cfg.got.beam_min
        beam_max = self._cfg.got.beam_max

        prefer_raw = self._cfg.got.beam_spread_uses_raw_score_enabled
        scores: List[float] = []
        for node in graph.iter_depth_first():
            if node.score is not None and node.parent_id is not None:
                scores.append(self._beam_pool_score(node) if prefer_raw else float(node.score))

        if not scores:
            return beam_max

        adaptive = self._cfg.got.adaptive_policies
        if adaptive and len(scores) >= 4:
            # Beam widens when scores are spread (uncertain) and narrows when they
            # cluster (converged). Use p25/p75 spread relative to a target band.
            ordered = sorted(scores)
            p25 = ordered[max(0, int(len(ordered) * 0.25) - 1)]
            p75 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.75))]
            spread = max(0.0, p75 - p25)  # 0..1 in practice
            target_spread = self._cfg.got.beam_target_spread
            ratio = min(1.0, spread / target_spread) if target_spread > 0 else 0.0
            beam = beam_min + int(round(ratio * (beam_max - beam_min)))
            beam = max(beam_min, min(beam_max, beam))
            _logger.debug(
                f"[GoT:BEAM] adaptive p25={p25:.3f} p75={p75:.3f} spread={spread:.3f} -> beam={beam}"
            )
            return beam

        score_high = self._cfg.got.beam_score_high
        score_low = self._cfg.got.beam_score_low
        avg_score = sum(scores) / len(scores)
        if avg_score >= score_high:
            beam = beam_min
        elif avg_score <= score_low:
            beam = beam_max
        else:
            ratio = (score_high - avg_score) / (score_high - score_low)
            beam = beam_min + int(ratio * (beam_max - beam_min))
        beam = max(beam_min, min(beam_max, beam))
        _logger.debug(f"[GoT:BEAM] legacy avg_score={avg_score:.3f} -> beam_width={beam}")
        return beam

    def identify_prune_candidates(self, graph: IdeaDag) -> List[str]:
        if not self._cfg.got.prune_enabled:
            return []

        min_nodes = self._cfg.got.prune_min_nodes_before_prune
        if graph.node_count() < min_nodes:
            return []

        scored: List[float] = []
        for node in graph.iter_depth_first():
            if node.score is not None and node.parent_id is not None:
                scored.append(float(node.score))

        adaptive = self._cfg.got.adaptive_policies
        if adaptive and len(scored) >= 5:
            mean = sum(scored) / len(scored)
            variance = sum((s - mean) ** 2 for s in scored) / len(scored)
            stddev = variance ** 0.5
            stddev_factor = self._cfg.got.prune_stddev_factor
            threshold = max(0.0, mean - stddev_factor * stddev)
        else:
            threshold = self._cfg.got.prune_score_threshold

        prune_ids = []
        for node in graph.iter_depth_first():
            if node.node_id == graph.root_id():
                continue
            if node.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED, IdeaNodeStatus.SKIPPED):
                continue
            if node.score is not None and node.score < threshold:
                has_result = node.details.get(DetailKey.ACTION_RESULT.value) is not None
                if not has_result:
                    prune_ids.append(node.node_id)

        if prune_ids:
            _logger.info(
                f"[GoT:PRUNE] Identified {len(prune_ids)} low-score nodes for pruning "
                f"(threshold={threshold:.3f}, adaptive={adaptive})"
            )
        return prune_ids

    def prune_nodes(self, graph: IdeaDag, node_ids: List[str]) -> int:
        pruned = 0
        for node_id in node_ids:
            node = graph.get_node(node_id)
            if not node:
                continue
            if node.status in (IdeaNodeStatus.DONE, IdeaNodeStatus.FAILED):
                continue
            node.status = IdeaNodeStatus.SKIPPED
            node.details["_got_pruned"] = True
            node.details["_got_prune_reason"] = f"Score {node.score} below threshold"
            pruned += 1

        if pruned > 0:
            _logger.info(f"[GoT:PRUNE] Pruned {pruned} low-score nodes")
        return pruned

    def should_backtrack(self, graph: IdeaDag, current_id: str) -> bool:
        if not self._cfg.got.backtrack_enabled:
            return False

        dead_end_limit = self._cfg.got.backtrack_dead_end_threshold
        low_score = self._cfg.got.backtrack_low_score_threshold

        node = graph.get_node(current_id)
        if not node:
            return False

        consecutive_low = 0
        path = graph.path_to_root(current_id)
        for path_node in path:
            if path_node.score is not None and path_node.score < low_score:
                consecutive_low += 1
            else:
                break

        if consecutive_low >= dead_end_limit:
            _logger.info(
                f"[GoT:BACKTRACK] Dead-end detected: {consecutive_low} consecutive low-score nodes "
                f"at node {current_id} (threshold={dead_end_limit}, low_score<{low_score})"
            )
            self._dead_end_count += 1
            return True

        return False

    def should_exit_early(
        self, graph: IdeaDag, step_confidences: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """A6: has the run earned a calibrated high-confidence early exit?

        The third outcome at the loop's decide-the-next-move point, alongside "keep going"
        and ``should_backtrack``: stop expanding new nodes and finalize with what we have.
        Mirrors ``should_backtrack``'s shape — flag-gated, pure, no LLM call, no mutation —
        but reads the run's accumulated step-confidence history rather than node scores.

        The bar is NOT hand-picked. ``confidence_early_exit.load_rule`` reads the versioned
        calibration artifact (thresholds derived from held-out ``(confidence-sequence,
        eventual-label)`` pairs under a certified false-stop rate; see
        ``idea_policies/confidence_early_exit.py``). An absent, unparseable or
        nothing-certified artifact yields no rule, and no rule means never stop — the same
        fail-safe direction as E-valuator's ``c_α = ∞``.

        ``got.confidence_early_exit_margin`` is added on top of the calibrated threshold
        (extra conservatism), and ``got.confidence_early_exit_min_judged_steps`` is a hard
        floor on how few judged steps may justify stopping at all.
        """
        if not self._cfg.got.confidence_early_exit_enabled:
            return False
        confidences = [
            float(entry["confidence"])
            for entry in (step_confidences or [])
            if isinstance(entry, dict) and isinstance(entry.get("confidence"), (int, float))
        ]
        if len(confidences) < max(1, int(self._cfg.got.confidence_early_exit_min_judged_steps)):
            return False
        rule = load_early_exit_rule()
        if rule is None:
            return False
        decision = rule.decide(confidences, margin=self._cfg.got.confidence_early_exit_margin)
        if decision.stop:
            _logger.info(
                f"[GoT:EARLY-EXIT] Calibrated high-confidence stop after "
                f"{decision.timestep} judged step(s) ({graph.node_count()} nodes): {decision.reason}"
            )
            self._early_exit_count += 1
            return True
        _logger.debug(f"[GoT:EARLY-EXIT] not stopping: {decision.reason}")
        return False

    def find_backtrack_target(self, graph: IdeaDag, current_id: str) -> Optional[str]:
        low_score = self._cfg.got.backtrack_low_score_threshold
        path = graph.path_to_root(current_id)
        for path_node in path:
            if path_node.node_id == graph.root_id():
                continue
            if path_node.score is not None and path_node.score >= low_score:
                if path_node.parent_id:
                    _logger.info(f"[GoT:BACKTRACK] Backtracking from {current_id} to {path_node.parent_id}")
                    return path_node.parent_id

        return graph.root_id()

    async def hybrid_retrieve(
        self,
        graph: IdeaDag,
        node_id: str,
        query: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.memory_manager:
            return []

        vector_results = await self.memory_manager.retrieve_relevant_memories(
            query=query,
            n_results=n_results,
        )

        node = graph.get_node(node_id)
        if not node:
            return vector_results

        path = graph.path_to_root(node_id)
        graph_context = []
        for path_node in path[:5]:
            ar = path_node.details.get(DetailKey.ACTION_RESULT.value)
            if ar and isinstance(ar, dict):
                from agent.app.idea_policies.action_constants import ActionResultExtractor
                if ActionResultExtractor.is_success(ar):
                    content = ar.get("content", "") or ""
                    if content and len(content) > 50:
                        graph_context.append({
                            "content": content[:500],
                            "metadata": {
                                "node_id": path_node.node_id,
                                "node_title": path_node.title,
                                "source": "graph_path",
                            },
                            "distance": 0.0,
                        })

        seen_ids = {m.get("id") for m in vector_results if m.get("id")}
        combined = list(vector_results)
        for gc in graph_context:
            gc_id = gc.get("metadata", {}).get("node_id")
            if gc_id not in seen_ids:
                combined.append(gc)
                seen_ids.add(gc_id)

        return combined[:n_results + len(graph_context)]

    def get_model_for_operation(self, operation: str, default_model: Optional[str] = None) -> Optional[str]:
        if not self._cfg.got.telemetry_routing_enabled:
            return default_model

        if operation in ("score", "evaluate", "evaluation"):
            override = self._cfg.got.telemetry_routing_score_model
            if override:
                return override
            return default_model

        if operation in ("generate", "expand", "expansion"):
            override = self._cfg.got.telemetry_routing_generate_model
            if override:
                return override
            return self._select_cheaper_model(default_model)

        return default_model

    @staticmethod
    def _select_cheaper_model(current_model: Optional[str]) -> Optional[str]:
        from agent.app.model_costs import MODEL_PRICING

        if not current_model or current_model not in MODEL_PRICING:
            return current_model

        current_output_cost = MODEL_PRICING[current_model]["output_per_million"]
        cheaper = None
        cheaper_cost = current_output_cost

        for model_name, pricing in MODEL_PRICING.items():
            if pricing["output_per_million"] < cheaper_cost:
                cheaper = model_name
                cheaper_cost = pricing["output_per_million"]

        if cheaper and cheaper != current_model:
            _logger.debug(f"[GoT:ROUTING] Downgraded {current_model} -> {cheaper} for generate operation")
            return cheaper
        return current_model

    @property
    def dead_end_count(self) -> int:
        return self._dead_end_count

    @dead_end_count.setter
    def dead_end_count(self, value: int) -> None:
        self._dead_end_count = int(value)

    @property
    def early_exit_count(self) -> int:
        """How many times ``should_exit_early`` fired this run (0 unless A6 is armed)."""
        return self._early_exit_count
